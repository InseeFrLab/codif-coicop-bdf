"""
Décision finale du code COICOP par LLM (module réutilisable).

Ce module est invoqué depuis main.py via la sous-commande ``decide-coicop``.
Il peut aussi être utilisé en import direct dans des notebooks ou scripts.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import openai
import pandas as pd
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field

# ── Logger ────────────────────────────────────────────────────────────────────

logger = logging.getLogger("coicop_llm")

# ── Configuration ─────────────────────────────────────────────────────────────

CONFIDENCE_LABELS = {
    1: "Très faible — le code proposé est incertain",
    2: "Faible — doute significatif sur le classement",
    3: "Moyen — classement vraisemblable mais non certain",
    4: "Élevé — le code est probablement correct",
    5: "Très élevé — le code est certain",
}

CHECKPOINT_EVERY = 50  # sauvegardes intermédiaires toutes les N observations

# ── Schéma de réponse structurée ──────────────────────────────────────────────


class DecisionCoicop(BaseModel):
    coicop_code: str = Field(
        description=(
            "Code COICOP retenu, exactement tel qu'il figure dans la nomenclature "
            "fournie (ex. '01.1.2.3'). Doit être un code existant."
        )
    )
    libelle: str = Field(
        max_length=150,
        description="Libellé COICOP correspondant au code retenu, copié de la nomenclature.",
    )
    explication: str = Field(
        max_length=400,
        description=(
            "Justification du choix en français en 1 phrase maximum : "
            "pourquoi ce code est retenu, comment les prédictions des modèles "
            "ont été prises en compte, et les éventuelles ambiguïtés."
        ),
    )
    confiance: int = Field(
        ge=1,
        le=5,
        description=(
            "Score de confiance de 1 (très faible) à 5 (très élevé) "
            "sur la décision prise."
        ),
    )


# ── I/O helpers (local + S3) ──────────────────────────────────────────────────


def _configure_s3(con: duckdb.DuckDBPyConnection) -> None:
    """Configure DuckDB S3 secret from environment variables."""
    con.execute(f"""
        CREATE SECRET secret_ls3 (
            TYPE S3,
            KEY_ID '{os.environ["AWS_ACCESS_KEY_ID"]}',
            SECRET '{os.environ["AWS_SECRET_ACCESS_KEY"]}',
            ENDPOINT '{os.environ["AWS_S3_ENDPOINT"]}',
            SESSION_TOKEN '{os.environ["AWS_SESSION_TOKEN"]}',
            REGION 'us-east-1',
            URL_STYLE 'path',
            SCOPE 's3://'
        );
    """)


def _read_parquet(path: str | Path) -> pd.DataFrame:
    """Read a parquet file from a local path or S3 URL."""
    path = str(path)
    if path.startswith("s3://"):
        con = duckdb.connect()
        _configure_s3(con)
        return con.execute(f"SELECT * FROM '{path}'").df()
    return pd.read_parquet(path)


def _read_csv(path: str | Path) -> pd.DataFrame:
    """Read a semicolon-separated CSV from a local path or S3 URL."""
    path = str(path)
    if path.startswith("s3://"):
        con = duckdb.connect()
        _configure_s3(con)
        return con.execute(f"SELECT * FROM '{path}'").df()
    return pd.read_csv(path, sep=";", encoding="utf-8")


def _write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to a parquet file locally or on S3."""
    path = str(path)
    if path.startswith("s3://"):
        con = duckdb.connect()
        _configure_s3(con)
        con.register("__output", df)
        con.execute(f"COPY __output TO '{path}' (FORMAT PARQUET)")
        return
    pd.DataFrame(df).to_parquet(path, index=False)


# ── Chargement des données ────────────────────────────────────────────────────


def load_nomenclature(path: Path | str) -> pd.DataFrame:
    """Charge la nomenclature COICOP sous forme de DataFrame (Code, Libelle)."""
    return _read_csv(path)


def _nomen_to_str(nomen: pd.DataFrame) -> str:
    return "\n".join(f"{r['Code']} — {r['Libelle']}" for _, r in nomen.iterrows())


def _get_libelle(code: str, nomen: pd.DataFrame) -> str:
    """Retourne le libellé COICOP d'un code, ou le code lui-même si introuvable."""
    row = nomen[nomen["Code"] == code]
    return str(row.iloc[0]["Libelle"]) if not row.empty else code


def try_consensus_decision(
    obs: dict,
    nomenclature: pd.DataFrame,
    threshold: float = 0.90,
) -> "DecisionCoicop | None":
    """Retourne une décision par consensus si tous les modèles s'accordent
    sur le même code et que la confiance TTC top-1 est suffisamment élevée.

    Évite un appel LLM pour les cas « faciles » : économise du temps et des tokens.
    Retourne None si les conditions ne sont pas réunies.
    """
    ttc_conf = obs.get("ttc_conf_1")
    if ttc_conf is None or (isinstance(ttc_conf, float) and pd.isna(ttc_conf)):
        return None
    if float(ttc_conf) < threshold:
        return None

    ttc_code = obs.get("ttc_code_1")
    if ttc_code is None or (isinstance(ttc_code, float) and pd.isna(ttc_code)):
        return None
    ttc_code = str(ttc_code)

    # Collect all non-null codes from the other two models
    other_codes = []
    for key in ("lcs_code", "rag_code"):
        val = obs.get(key)
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            other_codes.append(str(val))

    # Every available code must match TTC top-1
    if not all(c == ttc_code for c in other_codes):
        return None

    n_models = 1 + len(other_codes)
    libelle = _get_libelle(ttc_code, nomenclature)
    explication = (
        f"Consensus automatique : les {n_models} modèles s'accordent sur le code "
        f"{ttc_code} avec une confiance TTC de {float(ttc_conf):.0%}. "
        "Décision prise sans arbitrage LLM."
    )
    logger.debug(
        "[%s] Consensus détecté : code=%s conf_ttc=%.0f%%",
        obs.get("id"),
        ttc_code,
        float(ttc_conf) * 100,
    )
    return DecisionCoicop(
        coicop_code=ttc_code,
        libelle=libelle,
        explication=explication,
        confiance=5,
    )


def filter_nomenclature(nomen: pd.DataFrame, obs: dict, full: bool = False) -> str:
    """Retourne la nomenclature filtrée aux sections pertinentes pour cette observation.

    Strategy : collecte les préfixes L2 (ex. "01.1") de toutes les prédictions
    disponibles, puis garde :
      - les codes de section L1 pour le contexte
      - tous les codes dont le Code commence par l'un de ces préfixes L2

    Si aucune prédiction ou full=True, renvoie la nomenclature complète.
    Réduit typiquement ~700 lignes à ~50-150 lignes (gain ×4-10 sur les tokens).
    """
    if full:
        logger.debug("Nomenclature complète (%d codes)", len(nomen))
        return _nomen_to_str(nomen)

    predicted_codes = [
        obs.get("lcs_code"),
        obs.get("rag_code"),
        obs.get("ttc_code_1"),
        obs.get("ttc_code_2"),
        obs.get("ttc_code_3"),
    ]
    predicted_codes = [
        str(c)
        for c in predicted_codes
        if c is not None and not (isinstance(c, float) and pd.isna(c))
    ]

    if not predicted_codes:
        logger.debug("[%s] Aucune prédiction — nomenclature complète", obs.get("id"))
        return _nomen_to_str(nomen)

    # L1 prefixes (ex. "01") and L2 prefixes (ex. "01.1")
    l1 = {c.split(".")[0] for c in predicted_codes}
    l2 = {".".join(c.split(".")[:2]) for c in predicted_codes if "." in c}

    def is_relevant(code: str) -> bool:
        parts = code.split(".")
        if len(parts) == 1:  # section header (ex. "01")
            return parts[0] in l1
        if len(parts) == 2:  # sub-section (ex. "01.1")
            return parts[0] in l1
        return any(code.startswith(p) for p in l2)  # deeper codes

    mask = nomen["Code"].apply(is_relevant)
    filtered = nomen[mask]

    logger.debug(
        "[%s] Nomenclature filtrée : %d/%d codes (L1=%s, L2=%s)",
        obs.get("id"),
        len(filtered),
        len(nomen),
        sorted(l1),
        sorted(l2),
    )
    return _nomen_to_str(filtered)


def load_all_observations(
    lcs_path: Path | str,
    rag_path: Path | str,
    ttc_path: Path | str,
) -> pd.DataFrame:
    """Fusionne les trois fichiers parquet et retourne le DataFrame complet."""
    lcs_raw = _read_parquet(lcs_path)
    rag_raw = _read_parquet(rag_path)
    ttc_raw = _read_parquet(ttc_path)

    lcs = lcs_raw[
        [
            "id",
            "raw_product",
            "l_pr_product",
            "shop",
            "shop_type_name",
            "budget",
            "code",
            "predict_code",
            "distance",
            "common_substring",
            "prop_in_s2",
        ]
    ].rename(
        columns={
            "predict_code": "lcs_code",
            "distance": "lcs_distance",
            "common_substring": "lcs_substring",
            "prop_in_s2": "lcs_prop",
        }
    )

    rag = rag_raw[["id", "code_predict", "confidence", "codable"]].rename(
        columns={
            "code_predict": "rag_code",
            "confidence": "rag_confidence",
        }
    )

    ttc_cols = ["id", "predicted_code", "confidence"]
    for i in range(2, 4):
        ttc_cols += [f"predicted_code_top{i}", f"confidence_top{i}"]
    ttc = ttc_raw[ttc_cols].rename(
        columns={
            "predicted_code": "ttc_code_1",
            "confidence": "ttc_conf_1",
            "predicted_code_top2": "ttc_code_2",
            "confidence_top2": "ttc_conf_2",
            "predicted_code_top3": "ttc_code_3",
            "confidence_top3": "ttc_conf_3",
        }
    )

    return lcs.merge(rag, on="id", how="left").merge(ttc, on="id", how="left")


def get_observation(df: pd.DataFrame, obs_id: str) -> dict:
    row = df[df["id"] == obs_id]
    if row.empty:
        raise KeyError(f"Identifiant '{obs_id}' introuvable.")
    return row.iloc[0].to_dict()


# ── Construction du prompt ────────────────────────────────────────────────────


def build_prompt(
    obs: dict, nomenclature: pd.DataFrame, full_nomen: bool = False
) -> str:
    def fmt(val, fmt_spec=None):
        if val is None:
            return "N/A"
        try:
            if pd.isna(val):
                return "N/A"
        except (TypeError, ValueError):
            pass
        if fmt_spec:
            if isinstance(val, str):
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    return val
            return format(val, fmt_spec)
        return str(val)

    lcs_block = (
        f"  Code prédit  : {fmt(obs.get('lcs_code'))}\n"
        f"  Sous-chaîne  : {fmt(obs.get('lcs_substring'))} "
        f"(proportion dans référence : {fmt(obs.get('lcs_prop'), '.0%')})\n"
        f"  Distance     : {fmt(obs.get('lcs_distance'), '.3f')}"
    )

    rag_conf = obs.get("rag_confidence")
    rag_conf_str = (
        f"{rag_conf:.0%}" if rag_conf is not None and not pd.isna(rag_conf) else "N/A"
    )
    rag_block = (
        f"  Code prédit  : {fmt(obs.get('rag_code'))}\n"
        f"  Confiance    : {rag_conf_str}\n"
        f"  Codable      : {fmt(obs.get('codable'))}"
    )

    ttc_lines = []
    for rank in range(1, 4):
        code = fmt(obs.get(f"ttc_code_{rank}"))
        conf = obs.get(f"ttc_conf_{rank}")
        conf_str = f"{conf:.0%}" if conf is not None and not pd.isna(conf) else "N/A"
        ttc_lines.append(f"  Top {rank} : {code} (confiance {conf_str})")
    ttc_block = "\n".join(ttc_lines)

    return f"""\
Tu es un expert en comptabilité nationale et en statistiques de consommation.
Ta mission est de déterminer le code COICOP le plus approprié pour un produit acheté,
en t'appuyant sur le contexte d'achat et les prédictions de trois modèles automatiques.

═══════════════════════════════════════
PRODUIT
═══════════════════════════════════════
Libellé brut       : {fmt(obs.get("raw_product"))}
Libellé normalisé  : {fmt(obs.get("l_pr_product"))}
Enseigne           : {fmt(obs.get("shop"))}
Type de magasin    : {fmt(obs.get("shop_type_name"))}
Montant (€)        : {fmt(obs.get("budget"), ".2f")}
Code de référence  : {fmt(obs.get("code"))}

═══════════════════════════════════════
PRÉDICTIONS DES MODÈLES
═══════════════════════════════════════

[LCS — Longest Common Substring]
{lcs_block}

[RAG — Retrieval-Augmented Generation]
{rag_block}

[TTC — Classificateur par apprentissage profond]
{ttc_block}

═══════════════════════════════════════
NOMENCLATURE COICOP (codes valides)
═══════════════════════════════════════
{filter_nomenclature(nomenclature, obs, full=full_nomen)}

═══════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════
1. Choisis UN code COICOP parmi ceux listés dans la nomenclature ci-dessus.
2. Le code doit correspondre au niveau le plus précis qui te semble justifié.
3. Explique en 1 phrase maximum comment tu as pesé les prédictions des trois modèles
   et tout autre indice (enseigne, type de magasin, montant).
4. Attribue un score de confiance de 1 (très faible) à 5 (très élevé).
"""


# ── Appel LLM (synchrone — mode observation unique) ──────────────────────────

SYSTEM_MSG = (
    "Tu es un expert COICOP. "
    "Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant ni après, "
    "respectant exactement ce schéma :\n"
    '{"coicop_code": "<code exact de la nomenclature>", '
    '"libelle": "<libellé copié de la nomenclature>", '
    '"explication": "<1 phrase max>", '
    '"confiance": <entier 1-5>}'
)


def call_llm_sync(prompt: str, model: str, client: OpenAI) -> DecisionCoicop:
    logger.debug(
        "Request: model=%s, prompt_chars=%d (~%d tokens)",
        model,
        len(prompt),
        len(prompt) // 4,
    )
    t0 = time.perf_counter()
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=512,
        )
    except openai.AuthenticationError as exc:
        logger.error("Authentification échouée (vérifiez LLMLAB_API_KEY) : %s", exc)
        raise
    except openai.RateLimitError as exc:
        logger.warning("Rate limit après %.2fs : %s", time.perf_counter() - t0, exc)
        raise
    except openai.APIStatusError as exc:
        logger.error(
            "APIStatusError status=%d après %.2fs : %s",
            exc.status_code,
            time.perf_counter() - t0,
            exc.message,
        )
        raise
    except openai.APIConnectionError as exc:
        logger.error(
            "Connexion impossible après %.2fs : %s", time.perf_counter() - t0, exc
        )
        raise

    latency = time.perf_counter() - t0
    usage = completion.usage
    finish = completion.choices[0].finish_reason
    logger.info(
        "Response: finish_reason=%s | tokens prompt=%d completion=%d total=%d | %.2fs",
        finish,
        usage.prompt_tokens if usage else -1,
        usage.completion_tokens if usage else -1,
        usage.total_tokens if usage else -1,
        latency,
    )

    if finish == "length":
        raw = completion.choices[0].message.content
        logger.error("Réponse tronquée (finish_reason=length) : %s", raw)
        raise ValueError(f"Response truncated before JSON was complete. Raw: {raw}")

    raw = completion.choices[0].message.content or ""
    try:
        result = DecisionCoicop.model_validate_json(raw)
    except Exception as exc:
        logger.error("Parsing JSON échoué. Réponse brute : %s — %s", raw, exc)
        raise ValueError(f"JSON parsing failed: {exc}. Raw: {raw}") from exc

    logger.debug(
        "Parsed: coicop_code=%s confiance=%d", result.coicop_code, result.confiance
    )
    return result


# ── Appel LLM (asynchrone — mode batch) ──────────────────────────────────────


async def call_llm_async(
    obs: dict,
    nomenclature: pd.DataFrame,
    model: str,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    full_nomen: bool = False,
) -> dict:
    """Traite une observation et retourne un dict résultat (erreur capturée)."""
    async with semaphore:
        obs_id = obs["id"]
        base = {
            "id": obs_id,
            "llm_model": model,
            "llm_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Fast path: skip LLM when all models agree with high TTC confidence
        consensus = try_consensus_decision(obs, nomenclature)
        if consensus is not None:
            logger.info(
                "[%s] ✓ Consensus (LLM ignoré) : code=%s conf_ttc=%.0f%%",
                obs_id,
                consensus.coicop_code,
                float(obs.get("ttc_conf_1", 0)) * 100,
            )
            return {
                **base,
                "llm_model": "consensus",
                "llm_code": consensus.coicop_code,
                "llm_libelle": consensus.libelle,
                "llm_explication": consensus.explication,
                "llm_confiance": consensus.confiance,
                "llm_prompt_tokens": 0,
                "llm_completion_tokens": 0,
                "llm_latency_s": 0.0,
                "llm_error": None,
            }

        prompt = build_prompt(obs, nomenclature, full_nomen=full_nomen)
        logger.debug(
            "[%s] Request: model=%s, prompt_chars=%d (~%d tokens)",
            obs_id,
            model,
            len(prompt),
            len(prompt) // 4,
        )
        t0 = time.perf_counter()
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=512,
            )
            latency = time.perf_counter() - t0
            usage = completion.usage
            finish = completion.choices[0].finish_reason

            logger.info(
                "[%s] ✓ finish=%s | tokens prompt=%d completion=%d total=%d | %.2fs",
                obs_id,
                finish,
                usage.prompt_tokens if usage else -1,
                usage.completion_tokens if usage else -1,
                usage.total_tokens if usage else -1,
                latency,
            )

            if finish == "length":
                raw = completion.choices[0].message.content
                raise ValueError(f"Response truncated before JSON was complete. Raw: {raw}")

            raw = completion.choices[0].message.content or ""
            try:
                decision = DecisionCoicop.model_validate_json(raw)
            except Exception as parse_exc:
                logger.error(
                    "[%s] Parsing JSON échoué. Réponse brute : %s — %s",
                    obs_id,
                    raw,
                    parse_exc,
                )
                raise ValueError(f"JSON parsing failed: {parse_exc}. Raw: {raw}") from parse_exc

            logger.debug(
                "[%s] Parsed: coicop_code=%s confiance=%d",
                obs_id,
                decision.coicop_code,
                decision.confiance,
            )

            return {
                **base,
                "llm_code": decision.coicop_code,
                "llm_libelle": decision.libelle,
                "llm_explication": decision.explication,
                "llm_confiance": decision.confiance,
                "llm_prompt_tokens": usage.prompt_tokens if usage else None,
                "llm_completion_tokens": usage.completion_tokens if usage else None,
                "llm_latency_s": round(latency, 3),
                "llm_error": None,
            }

        except openai.RateLimitError as exc:
            latency = time.perf_counter() - t0
            logger.warning("[%s] RateLimitError après %.2fs : %s", obs_id, latency, exc)
            return {
                **base,
                "llm_code": None,
                "llm_libelle": None,
                "llm_explication": None,
                "llm_confiance": None,
                "llm_prompt_tokens": None,
                "llm_completion_tokens": None,
                "llm_latency_s": round(latency, 3),
                "llm_error": f"RateLimitError: {exc}",
            }

        except openai.APIStatusError as exc:
            latency = time.perf_counter() - t0
            logger.error(
                "[%s] APIStatusError status=%d après %.2fs : %s",
                obs_id,
                exc.status_code,
                latency,
                exc.message,
            )
            return {
                **base,
                "llm_code": None,
                "llm_libelle": None,
                "llm_explication": None,
                "llm_confiance": None,
                "llm_prompt_tokens": None,
                "llm_completion_tokens": None,
                "llm_latency_s": round(latency, 3),
                "llm_error": f"APIStatusError({exc.status_code}): {exc.message}",
            }

        except openai.APIConnectionError as exc:
            latency = time.perf_counter() - t0
            logger.error(
                "[%s] APIConnectionError après %.2fs : %s", obs_id, latency, exc
            )
            return {
                **base,
                "llm_code": None,
                "llm_libelle": None,
                "llm_explication": None,
                "llm_confiance": None,
                "llm_prompt_tokens": None,
                "llm_completion_tokens": None,
                "llm_latency_s": round(latency, 3),
                "llm_error": f"APIConnectionError: {exc}",
            }

        except Exception as exc:
            latency = time.perf_counter() - t0
            logger.error(
                "[%s] Erreur inattendue après %.2fs : %s: %s",
                obs_id,
                latency,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return {
                **base,
                "llm_code": None,
                "llm_libelle": None,
                "llm_explication": None,
                "llm_confiance": None,
                "llm_prompt_tokens": None,
                "llm_completion_tokens": None,
                "llm_latency_s": round(latency, 3),
                "llm_error": f"{type(exc).__name__}: {exc}",
            }


# ── Mode batch ────────────────────────────────────────────────────────────────


def _save_parquet(
    records: list[dict],
    path: Path | str,
    source_df: pd.DataFrame | None = None,
) -> None:
    out = pd.DataFrame(records)
    if source_df is not None:
        # Merge input columns (left) with LLM columns (right).
        # how="right" keeps only processed rows (subset at checkpoint time).
        out = source_df.merge(out, on="id", how="right")
    _write_parquet(out, path)


async def run_batch(
    df: pd.DataFrame,
    nomenclature: pd.DataFrame,
    model: str,
    concurrency: int,
    output_file: Path | str,
    full_nomen: bool = False,
) -> None:
    import sys

    from tqdm import tqdm

    # ── Reprise sur checkpoint ──
    done_ids: set[str] = set()
    results: list[dict] = []

    output_file_str = str(output_file)
    try:
        existing = _read_parquet(output_file_str)
        done_ids = set(existing["id"].tolist())
        # Keep only LLM columns — input columns are re-joined from df at save time
        llm_cols = ["id"] + [c for c in existing.columns if c.startswith("llm_")]
        results = existing[llm_cols].to_dict(orient="records")
        logger.info(
            "Reprise : %d observations déjà traitées, %d restantes.",
            len(done_ids),
            len(df) - len(done_ids),
        )
    except Exception:
        pass  # output file does not exist yet

    pending = [row for row in df.to_dict(orient="records") if row["id"] not in done_ids]

    if not pending:
        logger.info("Toutes les observations ont déjà été traitées.")
        return

    logger.info(
        "Démarrage batch : %d observations, concurrence=%d, modèle=%s",
        len(pending),
        concurrency,
        model,
    )

    # ── Client async ──
    api_key = os.environ["LLMLAB_API_KEY"]
    kwargs: dict = {"api_key": api_key}
    if base_url := os.environ.get("LLMLAB_URL"):
        kwargs["base_url"] = base_url
        logger.debug("LLMLAB_URL=%s", base_url)
    client = AsyncOpenAI(**kwargs)

    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    n_errors = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    batch_t0 = time.perf_counter()

    if not output_file_str.startswith("s3://"):
        Path(output_file_str).parent.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=len(pending), desc="Traitement LLM", unit="obs", file=sys.stderr)

    async def process_and_save(obs: dict) -> None:
        nonlocal n_errors, total_prompt_tokens, total_completion_tokens
        result = await call_llm_async(
            obs, nomenclature, model, client, semaphore, full_nomen
        )
        async with lock:
            results.append(result)
            if result["llm_error"]:
                n_errors += 1
            else:
                total_prompt_tokens += result.get("llm_prompt_tokens") or 0
                total_completion_tokens += result.get("llm_completion_tokens") or 0
            if len(results) % CHECKPOINT_EVERY == 0:
                _save_parquet(results, output_file, source_df=df)
                elapsed = time.perf_counter() - batch_t0
                rate = len(results) / elapsed if elapsed > 0 else 0
                logger.info(
                    "Checkpoint %d/%d — %.1f obs/min — tokens cumulés prompt=%d completion=%d — %d erreurs",
                    len(results),
                    len(pending) + len(done_ids),
                    rate * 60,
                    total_prompt_tokens,
                    total_completion_tokens,
                    n_errors,
                )
                pbar.set_postfix(
                    tokens=total_prompt_tokens + total_completion_tokens,
                    errors=n_errors,
                )
        pbar.update(1)

    tasks = [process_and_save(obs) for obs in pending]
    await asyncio.gather(*tasks)
    pbar.close()

    _save_parquet(results, output_file, source_df=df)

    elapsed = time.perf_counter() - batch_t0
    n_ok = len(results) - n_errors
    total_tokens = total_prompt_tokens + total_completion_tokens
    logger.info(
        "Batch terminé : %d succès, %d erreurs | tokens prompt=%d completion=%d total=%d | "
        "durée=%.1fs (%.1f obs/min) → %s",
        n_ok,
        n_errors,
        total_prompt_tokens,
        total_completion_tokens,
        total_tokens,
        elapsed,
        len(results) / elapsed * 60 if elapsed > 0 else 0,
        output_file,
    )


# ── Affichage (mode observation unique) ───────────────────────────────────────


def print_result(obs: dict, decision: DecisionCoicop) -> None:
    ref = obs.get("code")
    ref_str = str(ref) if ref is not None and not pd.isna(ref) else "N/A"
    correct = (
        str(decision.coicop_code).startswith(ref_str[: len(decision.coicop_code)])
        if ref_str != "N/A"
        else None
    )
    match_str = ""
    if correct is not None:
        match_str = (
            "  ✓ correspond au code de référence"
            if correct
            else "  ✗ diffère du code de référence"
        )

    conf_label = CONFIDENCE_LABELS.get(decision.confiance, "")

    print()
    print("══════════════════════════════════════════════════")
    print("  DÉCISION LLM")
    print("══════════════════════════════════════════════════")
    print(f"  Produit       : {obs.get('raw_product', 'N/A')}")
    print(f"  Enseigne      : {obs.get('shop', 'N/A')}")
    print()
    print(f"  Code COICOP   : {decision.coicop_code}")
    print(f"  Libellé       : {decision.libelle}")
    print(f"  Référence     : {ref_str}{match_str}")
    print()
    print(f"  Confiance     : {decision.confiance}/5 — {conf_label}")
    print()
    print("  Explication :")
    for line in decision.explication.split(". "):
        line = line.strip()
        if line:
            print(f"    {line}.")
    print("══════════════════════════════════════════════════")
    print()

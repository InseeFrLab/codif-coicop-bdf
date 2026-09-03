"""Validation des colonnes aux frontières entre étapes.

Le pipeline s'échange des parquet sans que personne ne déclare ce qu'il attend
ni ce qu'il produit. Une colonne qui disparaît en amont se manifeste très loin
de sa cause : le rapport plante au milieu de son rendu, l'accuracy chute sans
un mot, ou — pire — un chiffre faux s'affiche comme sain.

Pas de bibliothèque de validation ici, et c'est délibéré : les quinze
sélections en dur du dépôt portent toutes sur des **noms de colonnes**, jamais
sur des types. `pandera` contraindrait `pandas`, précisément le point sensible
documenté dans le `pyproject.toml` racine. Le patron suivi est celui de
`reconcile-sirus/src/scorer.py`, déjà en place et déjà éprouvé : manquant →
erreur, ordre différent → simple avertissement.
"""

import logging
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)


class SchemaError(ValueError):
    """Une frontière n'a pas le schéma annoncé."""


def _fmt(cols: Iterable[str]) -> str:
    cols = list(cols)
    return ", ".join(sorted(cols)) if cols else "(aucune)"


def require_columns(
    df,
    columns: Sequence[str],
    *,
    step: str,
    artifact: str,
) -> None:
    """Vérifie qu'un DataFrame lu en entrée porte les colonnes attendues.

    Le message nomme l'étape, l'artefact, les colonnes manquantes **et celles
    qui sont présentes** : sans cette dernière liste, diagnostiquer demande
    d'aller ouvrir le parquet à la main.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SchemaError(
            f"[{step}] colonnes absentes de « {artifact} » : {_fmt(missing)}.\n"
            f"  Colonnes présentes : {_fmt(df.columns)}\n"
            f"  L'étape amont a probablement changé la forme de sa sortie."
        )


def declare_output(
    df,
    columns: Sequence[str],
    *,
    step: str,
    artifact: str,
    strict: bool = False,
) -> None:
    """Vérifie une sortie **avant** de l'écrire.

    C'est ce qui manque totalement au dépôt aujourd'hui : rien ne contrôle ce
    qu'une étape produit, donc un livrable amputé part sur S3 et n'est
    découvert que par l'étape suivante, ou pas du tout.

    `strict=True` refuse en plus les colonnes non déclarées — à réserver aux
    livrables, où une colonne interne qui fuit est un défaut ; ailleurs, les
    étapes propagent légitimement les colonnes qu'elles reçoivent.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SchemaError(
            f"[{step}] refuse d'écrire « {artifact} » : colonnes manquantes "
            f"{_fmt(missing)}.\n  Colonnes produites : {_fmt(df.columns)}"
        )
    if strict:
        extra = [c for c in df.columns if c not in columns]
        if extra:
            raise SchemaError(
                f"[{step}] refuse d'écrire « {artifact} » : colonnes non "
                f"déclarées {_fmt(extra)}."
            )
    if list(df.columns[: len(columns)]) != list(columns):
        # Un ordre différent ne casse rien (tout se lit par nom) mais signale
        # souvent qu'une étape a été modifiée sans que le contrat suive.
        logger.warning(
            "[%s] « %s » : colonnes présentes mais dans un ordre inattendu.",
            step, artifact,
        )


def is_empty(df) -> bool:
    """Vrai si le DataFrame n'a aucune ligne.

    Écrit une sortie vide n'est pas toujours une erreur — `reconcile-sirus` le
    fait volontairement sur un run sans observation — d'où une fonction séparée
    plutôt qu'un contrôle imposé dans `declare_output`.
    """
    return len(df) == 0

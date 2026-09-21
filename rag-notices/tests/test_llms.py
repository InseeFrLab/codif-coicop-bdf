"""
Routines de test pour valider la connexion et la génération
sur un LLM hébergé sur llm.lab (Ollama, accès via OLLAMA_*).
"""
import os
import sys
from typing import Optional
from pydantic import BaseModel
from openai import OpenAI


# ---------------------------------------------------------------------------
# Schéma de réponse attendu du RAG
# ---------------------------------------------------------------------------
class ReponseFormat(BaseModel):
    codable: bool
    code_predict: Optional[str] = None
    confidence: float


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
def make_client() -> OpenAI:
    return OpenAI(
        base_url=os.environ["OLLAMA_URL"],
        api_key=os.environ["OLLAMA_API_KEY"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_list_models(client: OpenAI) -> list[str]:
    """Vérifie que le serveur répond et liste les modèles disponibles."""
    models = client.models.list()
    ids = [m.id for m in models.data]
    assert ids, "Aucun modèle disponible sur le serveur"
    print(f"[OK] Modèles disponibles : {ids}")
    return ids


def test_completion(client: OpenAI, model: str) -> str:
    """Vérifie qu'une requête simple aboutit."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Réponds uniquement avec le mot 'ok'."}],
        max_tokens=10,
        temperature=0,
    )
    content = response.choices[0].message.content
    assert content, "Réponse vide"
    print(f"[OK] Complétion simple : {content!r}")
    return content


def test_json_schema(client: OpenAI, model: str) -> ReponseFormat:
    """Vérifie que le serveur respecte un json_schema Pydantic."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Produit : lait entier bio 1L. "
                    "Peux-tu lui attribuer un code COICOP ? "
                    "Réponds en JSON."
                ),
            }
        ],
        max_tokens=256,
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "ReponseFormat",
                "schema": ReponseFormat.model_json_schema(),
                "strict": True,
            },
        },
    )
    raw = response.choices[0].message.content
    parsed = ReponseFormat.model_validate_json(raw)
    print(f"[OK] json_schema — codable={parsed.codable}, code={parsed.code_predict!r}, confidence={parsed.confidence:.2f}")
    return parsed


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_all(model: str = "gpt-oss:120b") -> None:
    client = make_client()

    print(f"\n=== Test llm.lab — modèle : {model} ===\n")

    available = test_list_models(client)
    if model not in available:
        print(f"[WARN] '{model}' absent du serveur. Modèles dispo : {available}")

    print(f"\n=== Test completion ===\n")
    test_completion(client, model)
    test_json_schema(client, model)

    print("\n=== Tous les tests sont passés ===\n")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "gemma4-31b"
    run_all(model)

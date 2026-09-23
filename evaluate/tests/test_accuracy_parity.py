"""Les mesures internes des RAG comptent-elles comme le reste du rapport ?

La section « Décomposition interne des classifieurs » du rapport est calculée par
les bibliothèques des modules RAG, pas par `codif_common.metrics`. Elle portait
donc un avertissement : leur dénominateur est constant, alors que la convention
stricte du reste du rapport écartait les vérités peu profondes.

Le passage à la règle unique — tronquer les deux codes à `k`, comparer — supprime
cet écart : `accuracy_by_level` faisait déjà exactement cela. Ce test le prouve
plutôt que de le supposer, et c'est lui qui autorise le retrait de
l'avertissement. S'il venait à échouer, c'est l'avertissement qu'il faut
remettre, pas ce fichier qu'il faut ajuster.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codif_common.metrics import CANONICAL_LEVELS, accuracy  # noqa: E402
from rag_annotations.eval import accuracy_by_level  # noqa: E402

# Un cas de chaque forme que la comparaison doit traiter. Les vérités peu
# profondes (`01.4`, `03.2`) sont le cœur du sujet : ce sont elles que l'ancienne
# convention retirait du dénominateur au niveau 4.
RECORDS = [
    # vérité profonde, prédiction juste
    {"code": "01.1.1.3", "code_predict": "01.1.1.3"},
    # vérité profonde, prédiction trop courte
    {"code": "01.1.1.3", "code_predict": "01.1.1"},
    # vérité peu profonde, prédiction identique : juste à tous les niveaux
    {"code": "01.4", "code_predict": "01.4"},
    # vérité peu profonde, prédiction PLUS FINE : juste au niveau 2, fausse au 4
    {"code": "01.4", "code_predict": "01.4.3.1"},
    # vérité peu profonde, prédiction fausse dès le niveau 2
    {"code": "03.2", "code_predict": "04.1.1.1"},
    # abstentions et sentinelles : des erreurs, jamais des exclusions
    {"code": "02.1.1.1", "code_predict": None},
    {"code": "02.1.1.1", "code_predict": "N/A"},
    {"code": "02.1.1.1", "code_predict": ""},
    # vérité absente sous ses deux formes : hors du calcul des deux côtés
    {"code": None, "code_predict": "05.1.1.1"},
    {"code": float("nan"), "code_predict": "05.1.1.1"},
]


def test_the_rag_libraries_now_score_like_codif_common():
    rag = accuracy_by_level(RECORDS, range(1, 5))
    frame = pd.DataFrame(RECORDS)

    for k in CANONICAL_LEVELS:
        n_ok, n, acc = accuracy(frame["code"], frame["code_predict"], k)
        assert n == rag[k]["n"], f"dénominateurs différents au niveau {k}"
        assert acc == rag[k]["accuracy"], f"accuracies différentes au niveau {k}"


def test_the_shared_fixture_actually_exercises_the_disputed_case():
    """Garde-fou : sans vérité peu profonde dans la fixture, le test ci-dessus
    passerait sous les deux conventions et ne prouverait rien."""
    frame = pd.DataFrame(RECORDS)
    # `01.4` contre `01.4.3.1` : juste au niveau 2, faux au niveau 4. C'est ce
    # que l'ancienne convention ne comptait ni d'un côté ni de l'autre.
    _, _, acc2 = accuracy(frame["code"], frame["code_predict"], 2)
    _, _, acc4 = accuracy(frame["code"], frame["code_predict"], 4)
    assert acc2 > acc4

    # Et les dix lignes ne se réduisent pas aux huit étiquetées par accident.
    _, n, _ = accuracy(frame["code"], frame["code_predict"], 4)
    assert n == 8

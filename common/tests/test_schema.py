"""Tests de la validation aux frontières.

Ce qu'on vérifie n'est pas qu'une erreur est levée — un `KeyError` pandas le
faisait déjà — mais qu'elle **dit quoi faire** : quelle étape, quel artefact,
quelles colonnes manquent, et lesquelles sont là. C'est cette dernière liste qui
évite d'aller ouvrir le parquet à la main.
"""

import logging

import pandas as pd
import pytest

from codif_common.schema import (
    SchemaError,
    declare_output,
    is_empty,
    require_columns,
)


def df(**cols):
    return pd.DataFrame({k: [v] for k, v in cols.items()})


class TestRequireColumns:
    def test_passes_when_all_present(self):
        require_columns(df(id=1, code="01.1"), ["id", "code"], step="s", artifact="a")

    def test_extra_columns_are_fine(self):
        """Une étape propage légitimement ce qu'elle reçoit."""
        require_columns(df(id=1, code="01.1", bonus=2), ["id"], step="s", artifact="a")

    def test_message_names_step_artifact_and_both_lists(self):
        with pytest.raises(SchemaError) as e:
            require_columns(
                df(id=1, libelle="café"), ["id", "code", "budget"],
                step="reconcile-llm", artifact="s3://.../raw_test_LCS.parquet",
            )
        msg = str(e.value)
        assert "reconcile-llm" in msg
        assert "raw_test_LCS.parquet" in msg
        assert "budget" in msg and "code" in msg      # manquantes
        assert "libelle" in msg                        # présentes
        assert "amont" in msg                          # oriente vers la cause

    def test_reports_all_missing_at_once(self):
        """Ne pas obliger à corriger une colonne, relancer, en découvrir une autre."""
        with pytest.raises(SchemaError) as e:
            require_columns(df(id=1), ["id", "a", "b", "c"], step="s", artifact="a")
        for col in ("a", "b", "c"):
            assert col in str(e.value)


class TestDeclareOutput:
    def test_refuses_to_write_an_incomplete_output(self):
        """Le trou que ça comble : rien ne contrôlait ce qu'une étape produit."""
        with pytest.raises(SchemaError, match="refuse d'écrire"):
            declare_output(df(id=1), ["id", "predicted_code"], step="s", artifact="a")

    def test_tolerates_extra_columns_by_default(self):
        declare_output(df(id=1, extra=2), ["id"], step="s", artifact="a")

    def test_strict_refuses_leaking_columns(self):
        """Pour un livrable : une colonne interne qui fuit est un défaut."""
        with pytest.raises(SchemaError, match="non .*déclarées"):
            declare_output(
                df(id=1, _source_input_file="x"), ["id"],
                step="export-results", artifact="livrable.csv", strict=True,
            )

    def test_wrong_order_warns_but_passes(self, caplog):
        """Tout se lit par nom : l'ordre ne casse rien, mais signale souvent
        qu'une étape a bougé sans que le contrat suive."""
        with caplog.at_level(logging.WARNING):
            declare_output(df(code="01.1", id=1), ["id", "code"], step="s", artifact="a")
        assert "ordre inattendu" in caplog.text


class TestIsEmpty:
    def test_separate_from_declare_output(self):
        """Une sortie vide n'est pas toujours une erreur : reconcile-sirus en
        écrit une volontairement sur un run sans observation."""
        empty = pd.DataFrame({"id": []})
        declare_output(empty, ["id"], step="s", artifact="a")   # ne lève pas
        assert is_empty(empty)
        assert not is_empty(df(id=1))

"""Tests de la règle de nommage des collections Qdrant.

`vector_index.py` vit dans `codif_common` et sert aux deux pipelines
d'indexation. Ce test fixe la forme du nom, identique pour les deux : ce qui
distingue deux collections, c'est le run qui les a bâties.
"""

from codif_common.vector_index import build_collection_name, manifest_uri


class TestBuildCollectionName:
    def test_notices(self):
        assert build_collection_name(
            base="coicop_notices", run_date="2026-09-02", run_id="index-notices-a7k2p"
        ) == "coicop_notices__2026-09-02__index-notices-a7k2p"

    def test_annotations_have_the_same_form(self):
        """Il y avait ici un segment de périmètre (`__full__` / `__train__`),
        du temps où la KB était un demi-jeu. La KB, ce sont désormais tous les
        produits annotés : le segment n'a plus rien à distinguer."""
        assert build_collection_name(
            base="coicop_annotations",
            run_date="2026-09-02",
            run_id="index-anno-b3x9q",
        ) == "coicop_annotations__2026-09-02__index-anno-b3x9q"

    def test_sample_size_is_visible_in_the_name(self):
        """Sans ce suffixe, un index jouet de 100 points et un index complet
        portent des noms de même forme."""
        name = build_collection_name(
            base="coicop_annotations",
            run_date="2026-09-02",
            run_id="index-anno-b3x9q",
            sample_size=100,
        )
        assert name.endswith("__sample100")

    def test_no_sample_suffix_when_unset_or_zero(self):
        for sample in (None, 0):
            name = build_collection_name(
                base="b", run_date="2026-09-02", run_id="r", sample_size=sample
            )
            assert "sample" not in name

    def test_name_is_url_path_safe(self):
        """Qdrant expose le nom de collection dans ses URLs."""
        name = build_collection_name(
            base="coicop_annotations",
            run_date="2026-09-02",
            run_id="index-anno-b3x9q",
            sample_size=50,
        )
        assert not set(name) & set(":/ ?#")

    def test_distinct_runs_never_collide(self):
        """Toute la raison d'être du changement : deux indexations ne doivent
        plus jamais écrire dans la même collection."""
        a = build_collection_name(base="b", run_date="2026-09-02", run_id="run-aaa")
        b = build_collection_name(base="b", run_date="2026-09-02", run_id="run-bbb")
        assert a != b


class TestManifestUri:
    def test_is_derived_from_the_collection_name_alone(self):
        """Connaître le nom suffit à retrouver le manifeste : pas de catalogue."""
        assert manifest_uri("s3://bucket/manifests", "coicop_notices__2026-09-02__r") == (
            "s3://bucket/manifests/coicop_notices__2026-09-02__r.json"
        )

    def test_trailing_slash_does_not_double_up(self):
        assert manifest_uri("s3://bucket/manifests/", "c") == "s3://bucket/manifests/c.json"

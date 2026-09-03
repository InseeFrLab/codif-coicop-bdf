"""Tests de la règle de nommage des collections Qdrant.

`vector_index.py` vit dans `codif_common` et sert aux deux pipelines
d'indexation. Ce test fixe la forme du nom pour les deux usages réels :
notices sans périmètre, annotations avec.
"""

from codif_common.vector_index import build_collection_name, manifest_uri


class TestBuildCollectionName:
    def test_notices_have_no_mode_segment(self):
        """Les notices dérivent d'un CSV statique : leur contenu ne dépend pas
        du mode prod/éval, donc en mettre un dans le nom serait mensonger."""
        assert build_collection_name(
            base="coicop_notices", run_date="2026-09-02", run_id="index-notices-a7k2p"
        ) == "coicop_notices__2026-09-02__index-notices-a7k2p"

    def test_annotations_carry_their_kb_scope(self):
        """Le périmètre de la KB est dans le nom parce qu'une collection `full`
        contient tous les produits annotés : c'est le champ qu'on confond en
        recopiant, et le seul qui distingue deux collections du même jour."""
        assert build_collection_name(
            base="coicop_annotations",
            run_date="2026-09-02",
            run_id="index-anno-b3x9q",
            mode="full",
        ) == "coicop_annotations__full__2026-09-02__index-anno-b3x9q"
        assert build_collection_name(
            base="coicop_annotations",
            run_date="2026-09-02",
            run_id="index-anno-b3x9q",
            mode="train",
        ) == "coicop_annotations__train__2026-09-02__index-anno-b3x9q"

    def test_sample_size_is_visible_in_the_name(self):
        """Sans ce suffixe, un index jouet de 100 points et un index complet
        portent des noms de même forme."""
        name = build_collection_name(
            base="coicop_annotations",
            run_date="2026-09-02",
            run_id="index-anno-b3x9q",
            mode="full",
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
            mode="full",
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

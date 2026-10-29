import pytest

from conftest import DEFAULT_CONFIG, MINERAL_CONFIG
from src.labels import LabelSpace


def space(**kw):
    return LabelSpace.load(MINERAL_CONFIG, **kw)


def test_default_config_uses_type_codes():
    s = LabelSpace.load(DEFAULT_CONFIG)
    assert s.answers("type") == ["A-AM", "A-CF", "A-COF", "A-CP", "A-CRO", "NA-CS", "NA-OF"]
    assert [c.asbestos for c in s.classes] == [True] * 5 + [False] * 2
    # folder / annotation spellings map without explicit aliases
    assert s.map("A-COF") == "a_cof" and s.map("a_cro") == "a_cro" and s.map("na-of") == "na_of"
    assert s.target_answer("na_cs", "binary") == "No"
    assert s.target_answer("a_cp", "binary") == "Yes"
    with pytest.raises(ValueError):
        s.map("A-XYZ")
    m = LabelSpace.load(DEFAULT_CONFIG, merges=["a_cf,a_cof=a_cf_cof:A-CF/COF"])
    assert m.map("A-COF") == "a_cf_cof" and len(m.keys) == 6


def test_aliases_and_normalisation():
    s = space()
    assert s.map("White Asbestos") == "chrysotile"
    assert s.map("chrysotile") == "chrysotile"
    assert s.map("Non-asbestos") == "non_asbestos"  # display name is an implicit alias
    assert s.map("NAD") == "non_asbestos"
    assert s.map("unknown") is None
    with pytest.raises(ValueError):
        s.map("talc")


def test_binary_and_type_targets():
    s = space()
    assert s.answers("binary") == ["No", "Yes"]
    assert s.target_answer("amosite", "binary") == "Yes"
    assert s.target_answer("non_asbestos", "binary") == "No"
    assert s.target_answer("crocidolite", "type") == "Crocidolite"
    assert s.target_index("chrysotile", "type") == 0


def test_merge_from_cli_spec_redirects_aliases():
    s = space(merges=["amosite,crocidolite=amphibole:Amphibole asbestos"])
    assert s.keys == ["chrysotile", "amphibole", "non_asbestos"]
    assert s.map("blue asbestos") == "amphibole"
    assert s.map("amosite") == "amphibole"
    assert s.by_key("amphibole").asbestos
    assert s.answers("type") == ["Chrysotile", "Amphibole asbestos", "Non-asbestos"]


def test_merge_refuses_mixed_asbestos_flags():
    with pytest.raises(ValueError):
        space(merges=["chrysotile,non_asbestos=x"])


def test_rename_and_drop():
    s = space(renames=["non_asbestos=No asbestos"], drops=["crocidolite"])
    assert s.target_answer("non_asbestos", "type") == "No asbestos"
    assert "crocidolite" not in s.keys
    assert s.map("crocidolite") is None
    assert s.map("blue_asbestos") is None


def test_roundtrip_json(tmp_path):
    s = space(merges=["amosite,crocidolite=amphibole"])
    s.to_json(tmp_path / "ls.json")
    t = LabelSpace.load(tmp_path / "ls.json")
    assert t.keys == s.keys and t.names == s.names and t.map("riebeckite") == "amphibole"


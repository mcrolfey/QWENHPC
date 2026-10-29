import pytest

from conftest import DEFAULT_CONFIG, MACRO_CONFIG, MINERAL_CONFIG, ROOT
from src.labels import LabelSpace


def space(**kw):
    return LabelSpace.load(MINERAL_CONFIG, **kw)


def test_default_config_uses_type_codes():
    s = LabelSpace.load(DEFAULT_CONFIG)
    assert s.keys == ["a_am", "a_cf", "a_cof", "a_cp", "a_cro", "na_cs", "na_of"]
    assert s.answers("type") == ["Amosite", "Chrysotile Fluffy", "Chrysotile and Organic Fiber",
                                 "Chrysotile Pointy", "Crocidolite", "Cement Sheet", "Organic Fiber"]
    assert [c.asbestos for c in s.classes] == [True] * 5 + [False] * 2
    assert s.map("Chrysotile Pointy") == "a_cp"   # descriptive names work as labels too
    codes = LabelSpace.load(ROOT / "configs" / "classes_micro_codes.yaml")
    assert codes.keys == s.keys and codes.answers("type")[0] == "A-AM"
    # folder / annotation spellings map without explicit aliases
    assert s.map("A-COF") == "a_cof" and s.map("a_cro") == "a_cro" and s.map("na-of") == "na_of"
    assert s.target_answer("na_cs", "binary") == "No"
    assert s.target_answer("a_cp", "binary") == "Yes"
    with pytest.raises(ValueError):
        s.map("A-XYZ")
    m = LabelSpace.load(DEFAULT_CONFIG, merges=["a_cf,a_cp=a_c:Chrysotile"])
    assert m.map("A-CP") == "a_c" and m.map("a-cf") == "a_c" and len(m.keys) == 6
    assert m.answers("type")[1] == "Chrysotile"


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



def test_macro_config_and_dataset_spelling():
    s = LabelSpace.load(MACRO_CONFIG)
    assert s.keys == ["a_cry", "a_cry_of", "a_cry_of_cs", "na_cs"]
    assert s.map("A-CRY-OF-CS") == "a_cry_of_cs"
    assert s.map("Crysotile Organic Fiber") == "a_cry_of"
    assert s.target_answer("a_cry_of_cs", "binary") == "Yes"
    with pytest.raises(ValueError):
        s.map("A-AM")                                  # micro-only class
    codes = LabelSpace.load(ROOT / "configs" / "classes_macro_codes.yaml")
    assert codes.keys == s.keys and codes.answers("type")[0] == "A-CRY"


def test_union_of_micro_and_macro():
    u = LabelSpace.load([DEFAULT_CONFIG, MACRO_CONFIG])
    # NA-CS shared by key; "Chrysotile and Organic Fiber" shared by answer text
    assert len(u.keys) == 7 + 4 - 2
    assert u.map("A-CRY-OF") == "a_cof" and u.map("NA-CS") == "na_cs"
    assert u.map("A-AM") == "a_am" and u.map("A-CRY") == "a_cry"
    assert u.map("crysotile") == "a_cry"
    assert [u.is_asbestos(k) for k in ("a_cry_of_cs", "na_of")] == [True, False]


def test_label_space_defaults_follow_modality():
    from types import SimpleNamespace
    from src.labels import label_space_from_args
    args = SimpleNamespace(class_config=None, merge=[], rename=[], drop_class=[])
    assert len(label_space_from_args(args, "microscopic").keys) == 7
    assert len(label_space_from_args(args, "smartphone").keys) == 4
    assert len(label_space_from_args(args, "unified").keys) == 9

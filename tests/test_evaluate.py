import numpy as np
import pandas as pd

from src.evaluate import (aggregate_record, bootstrap_ci, classification_metrics, compare_predictions,
                          holm, mcnemar_test, parse_answer)


def test_mcnemar_exact_and_chi2():
    a = np.array([1] * 10 + [1] * 8 + [0] * 2 + [0] * 5, bool)
    b = np.array([1] * 10 + [0] * 8 + [1] * 2 + [0] * 5, bool)
    r = mcnemar_test(a, b)
    assert r["table"]["a_correct_b_wrong"] == 8 and r["table"]["a_wrong_b_correct"] == 2
    assert r["method"] == "exact binomial"
    assert abs(r["p_exact"] - 0.109375) < 1e-9          # 2 * P(X<=2 | n=10, p=.5)
    assert abs(r["chi2_cc"] - 2.5) < 1e-9               # (|8-2|-1)^2 / 10


def test_fisher_exact_matches_reference():
    from src.evaluate import fisher_exact_2x2
    odds, p = fisher_exact_2x2([[8, 2], [1, 5]])
    assert abs(odds - 20.0) < 1e-9
    assert abs(p - 0.034965) < 1e-5                     # scipy.stats.fisher_exact reference


def test_holm():
    adj = holm({"binary": 0.01, "type": 0.04})
    assert adj == {"binary": 0.02, "type": 0.04}


def test_parse_answer():
    c = ["Chrysotile", "Amosite", "Crocidolite", "Non-asbestos"]
    assert parse_answer("Amosite", c) == 1
    assert parse_answer("The fibre is crocidolite.", c) == 2
    assert parse_answer("non asbestos", c) == 3
    assert parse_answer("talc", c) == -1


def test_tile_aggregation():
    asb = np.array([True, True, True, False])
    tiles = np.array([[0.0, 0.0, 0.0, 1.0]] * 5 + [[0.1, 0.8, 0.0, 0.1]])
    probs, pred = aggregate_record("type", tiles, asb, "max", 0.5, "hierarchical")
    assert pred == 1 and abs(probs.sum() - 1) < 1e-9     # one positive tile is enough
    probs, pred = aggregate_record("binary", np.array([[0.9, 0.1], [0.3, 0.7]]), asb, "max", 0.5, "hierarchical")
    assert pred == 1 and abs(probs[1] - 0.7) < 1e-9


def test_metrics_and_ci():
    yt = np.array([0, 0, 1, 1, 1, 0])
    yp = np.array([0, 1, 1, 1, 0, 0])
    m = classification_metrics(yt, yp, np.eye(2)[yp], ["No", "Yes"])
    assert abs(m["accuracy"] - 4 / 6) < 1e-9
    assert m["per_class"]["Yes"]["tp"] == 2 and m["sensitivity"] == 2 / 3
    ci = bootstrap_ci(yt, yp, np.array(list("aabbcc")), 2, n_boot=200)
    assert ci["accuracy"][0] <= m["accuracy"] <= ci["accuracy"][1]


def _preds(correct_pattern, prefix):
    rows = []
    for i, ok in enumerate(correct_pattern):
        rows.append({"record_id": f"{prefix}{i}", "sample_id": f"S{i}", "task": "binary",
                     "true_name": "Yes", "pred_name": "Yes" if ok else "No",
                     "correct": bool(ok), "p::No": 0.0 if ok else 1.0, "p::Yes": 1.0 if ok else 0.0})
    return pd.DataFrame(rows)


def test_compare_paired_by_specimen():
    a = _preds([1] * 18 + [0] * 2, "m")
    b = _preds([1] * 10 + [0] * 8 + [1] * 2, "p")
    r = compare_predictions(a, b, "auto", n_boot=100, seed=0)
    assert r["pairing"] == "sample_id" and r["n_pairs"] == 20
    assert r["mcnemar"]["table"]["a_correct_b_wrong"] == 8


def test_compare_collapses_multiple_images_per_specimen():
    a = _preds([1, 1, 0, 0], "m")
    a["sample_id"] = ["S0", "S0", "S1", "S1"]
    a["p::Chrysotile"] = np.nan          # column of the other task, empty for binary rows
    b = _preds([0, 1], "p")
    r = compare_predictions(a, b, "sample_id", n_boot=0, seed=0)
    assert r["n_pairs"] == 2
    assert r["mcnemar"]["table"]["a_correct_b_wrong"] == 1   # S0: A right, B wrong
    assert r["mcnemar"]["table"]["a_wrong_b_correct"] == 1   # S1: A wrong, B right


def test_compare_unpaired_falls_back_to_fisher():
    a = _preds([1] * 18 + [0] * 2, "m")
    b = _preds([1] * 10 + [0] * 10, "p")
    b["sample_id"] = "X" + b["sample_id"]
    r = compare_predictions(a, b, "auto", n_boot=100, seed=0)
    assert r["pairing"] == "none" and "fisher_exact" in r


def test_compare_skips_tasks_with_different_class_sets(tmp_path):
    import json
    from types import SimpleNamespace
    from src.evaluate import run_compare
    a = _preds([1, 0, 1], "m")
    b = _preds([1, 1, 0], "p")
    t_a = a.assign(task="type", **{"p::No": np.nan, "p::Yes": np.nan, "p::Amosite": 1.0})
    t_b = b.assign(task="type", **{"p::No": np.nan, "p::Yes": np.nan, "p::Chrysotile": 1.0})
    pd.concat([a, t_a]).to_csv(tmp_path / "a.csv", index=False)
    pd.concat([b, t_b]).to_csv(tmp_path / "b.csv", index=False)
    args = SimpleNamespace(a=str(tmp_path / "a.csv"), b=str(tmp_path / "b.csv"), name_a="micro", name_b="macro",
                           task="both", pair_on="auto", bootstrap=0, seed=0, output=str(tmp_path / "c.json"))
    run_compare(args)
    res = json.loads((tmp_path / "c.json").read_text())
    assert set(res["tasks"]) == {"binary"} and "type" in res["skipped"]

"""Evaluation: predictions, classification metrics, confusion matrices, and
statistical comparison (McNemar's test) between models / image modalities.

Sub-commands
------------
``predict``  Run a fine-tuned adapter (or the zero-shot base model) on a split
             and write image-level predictions + metrics::

    python -m src.evaluate predict --adapter_dir $RUNS/micro \
        --data_dir $DATA --split test --modality microscopic \
        --output_dir $RESULTS/micro_on_micro

``compare``  Compare two prediction files for the same task(s)::

    python -m src.evaluate compare \
        --a $RESULTS/micro_on_micro/predictions.csv --name_a microscopic \
        --b $RESULTS/phone_on_phone/predictions.csv --name_b smartphone \
        --task both --output $RESULTS/compare_micro_vs_phone.json

    If the files share specimens (``sample_id``) the comparison is *paired*
    and McNemar's test is used (exact binomial when there are fewer than 25
    discordant pairs, otherwise chi-square with continuity correction). If
    they share no specimens, McNemar is not applicable; Fisher's exact test on
    accuracy and bootstrap CIs on the metric differences are reported instead.
    Holm correction is applied across tasks.

Scoring
-------
``--scoring full`` (default) computes the log-likelihood of every candidate
answer, giving a proper probability distribution over classes (needed for
ROC-AUC and tile aggregation). ``first_token`` is ~C times faster and exact
when answers start with distinct tokens. ``generate`` decodes free text and
parses it (useful for zero-shot baselines; reports the invalid-answer rate).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("evaluate")


# =========================================================================
# Metrics (no torch / transformers needed)
# =========================================================================
def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray | None,
                           class_names: list[str]) -> dict:
    from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                                 cohen_kappa_score, confusion_matrix, matthews_corrcoef,
                                 multilabel_confusion_matrix, precision_recall_fscore_support,
                                 roc_auc_score)

    labels = list(range(len(class_names)))
    valid = y_pred >= 0
    y_pred_eff = np.where(valid, y_pred, -1)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred_eff, labels=labels, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred_eff, labels=labels, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred_eff, labels=labels, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred_eff, labels=labels)
    mcm = multilabel_confusion_matrix(y_true, y_pred_eff, labels=labels)
    out = {
        "n": int(len(y_true)),
        "invalid_predictions": int((~valid).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred_eff)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred_eff)) if len(set(y_true)) > 1 else None,
        "precision_macro": float(pm), "recall_macro": float(rm), "f1_macro": float(fm),
        "precision_weighted": float(pw), "recall_weighted": float(rw), "f1_weighted": float(fw),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred_eff, labels=labels)),
        "mcc": float(matthews_corrcoef(y_true, y_pred_eff)),
        "per_class": {
            name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i]),
                   "tn": int(mcm[i][0, 0]), "fp": int(mcm[i][0, 1]), "fn": int(mcm[i][1, 0]), "tp": int(mcm[i][1, 1])}
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }
    if len(class_names) == 2:
        tn, fp, fn, tp = cm.ravel()
        out["sensitivity"] = float(tp / (tp + fn)) if tp + fn else None
        out["specificity"] = float(tn / (tn + fp)) if tn + fp else None
    if probs is not None and len(set(y_true)) > 1:
        try:
            if len(class_names) == 2:
                out["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
                out["pr_auc"] = float(average_precision_score(y_true, probs[:, 1]))
            elif len(set(y_true)) == len(class_names):
                out["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, probs, multi_class="ovr", labels=labels))
        except ValueError as exc:
            log.warning("AUC not computed: %s", exc)
    return out


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, n_classes: int,
                 n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Cluster bootstrap (resampling specimens, not images) CIs for key metrics."""
    from sklearn.metrics import balanced_accuracy_score, f1_score

    if n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    labels = list(range(n_classes))
    stats = defaultdict(list)
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        yt, yp = y_true[idx], y_pred[idx]
        stats["accuracy"].append(np.mean(yt == yp))
        stats["f1_macro"].append(f1_score(yt, yp, labels=labels, average="macro", zero_division=0))
        if len(set(yt)) > 1:
            stats["balanced_accuracy"].append(balanced_accuracy_score(yt, yp))
        if n_classes == 2:
            pos, neg = yt == 1, yt == 0
            if pos.any():
                stats["sensitivity"].append(np.mean(yp[pos] == 1))
            if neg.any():
                stats["specificity"].append(np.mean(yp[neg] == 0))
    return {k: [float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))]
            for k, v in stats.items() if v}


def plot_confusion(cm: np.ndarray, names: list[str], path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, axes = plt.subplots(1, 2, figsize=(5.5 + 1.2 * len(names), 2.8 + 0.6 * len(names)))
    for ax, mat, sub, fmt in ((axes[0], cm, "counts", "{:.0f}"), (axes[1], norm, "row-normalised", "{:.2f}")):
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max() if mat.max() > 0 else 1)
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title} ({sub})", fontsize=9)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center",
                        color="white" if mat[i, j] > 0.6 * mat.max() else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =========================================================================
# Statistical comparison
# =========================================================================
def _binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p-value for p=0.5 (symmetric, so 2 x smaller tail)."""
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def _chi2_sf_1df(x: float) -> float:
    return math.erfc(math.sqrt(max(x, 0.0) / 2.0))


def fisher_exact_2x2(table: list[list[int]]) -> tuple[float, float]:
    """Two-sided Fisher's exact test (sum of tables no more likely than observed)."""
    (a, b), (c, d) = table
    r1, c1, n = a + b, a + c, a + b + c + d
    def p(x):
        return math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)
    p_obs = p(a)
    lo, hi = max(0, r1 + c1 - n), min(r1, c1)
    pval = sum(p(x) for x in range(lo, hi + 1) if p(x) <= p_obs * (1 + 1e-7))
    odds = (a * d) / (b * c) if b * c else float("inf")
    return odds, min(1.0, pval)


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """McNemar's test on paired correctness vectors."""
    correct_a, correct_b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01 = int(np.sum(correct_a & ~correct_b))  # A right, B wrong
    n10 = int(np.sum(~correct_a & correct_b))  # A wrong, B right
    n11 = int(np.sum(correct_a & correct_b))
    n00 = int(np.sum(~correct_a & ~correct_b))
    disc = n01 + n10
    res = {"table": {"both_correct": n11, "a_correct_b_wrong": n01, "a_wrong_b_correct": n10, "both_wrong": n00},
           "discordant": disc}
    if disc == 0:
        res.update({"p_exact": 1.0, "p_chi2_cc": 1.0, "chi2_cc": 0.0, "p_value": 1.0, "method": "no discordant pairs"})
        return res
    p_exact = _binom_two_sided(n01, disc)
    stat = (abs(n01 - n10) - 1) ** 2 / disc
    p_chi = _chi2_sf_1df(stat)
    exact = disc < 25
    res.update({
        "p_exact": p_exact, "chi2_cc": float(stat), "p_chi2_cc": p_chi,
        "p_value": p_exact if exact else p_chi,
        "method": "exact binomial" if exact else "chi-square with continuity correction",
        "odds_ratio_discordant": float(n01 / n10) if n10 else float("inf"),
    })
    return res


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def _prob_cols(df: pd.DataFrame) -> list[str]:
    """Probability columns of the task present in ``df`` (other tasks' columns are empty)."""
    return [c for c in df.columns if c.startswith("p::") and df[c].notna().all()]


def _json_safe(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _sample_level(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Collapse to one row per pairing key (mean class probabilities)."""
    pcols = _prob_cols(df)
    rows = []
    for k, g in df.groupby(key, sort=True):
        if g["true_name"].nunique() > 1:
            log.warning("%s=%s has conflicting true labels %s; using the first", key, k, g["true_name"].unique())
        if len(g) == 1 or not pcols:
            pred = g["pred_name"].mode().iloc[0]
        else:
            pred = pcols[int(np.argmax(g[pcols].mean().to_numpy()))][3:]
        rows.append({key: k, "true_name": g["true_name"].iloc[0], "pred_name": pred})
    out = pd.DataFrame(rows)
    out["correct"] = out["true_name"] == out["pred_name"]
    return out


def _macro_f1(true: np.ndarray, pred: np.ndarray, names: list[str]) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(true, pred, labels=names, average="macro", zero_division=0))


def compare_predictions(a: pd.DataFrame, b: pd.DataFrame, pair_on: str, n_boot: int, seed: int) -> dict:
    key = pair_on
    if key == "auto":
        key = "sample_id" if set(a["sample_id"]) & set(b["sample_id"]) else "none"
    names = sorted(set(a["true_name"]) | set(b["true_name"]))
    rng = np.random.default_rng(seed)
    res: dict = {"pairing": key}

    if key != "none":
        sa, sb = _sample_level(a, key), _sample_level(b, key)
        m = sa.merge(sb, on=key, suffixes=("_a", "_b"))
        if len(m) == 0:
            raise SystemExit(f"No shared '{key}' values between the files; use --pair_on none.")
        mismatch = (m["true_name_a"] != m["true_name_b"]).sum()
        if mismatch:
            log.warning("%d paired items have different ground truth in A and B", mismatch)
        ca, cb = m["correct_a"].to_numpy(), m["correct_b"].to_numpy()
        res.update({
            "n_pairs": int(len(m)),
            "n_unpaired_dropped": {"a": int(len(sa) - len(m)), "b": int(len(sb) - len(m))},
            "accuracy_a": float(ca.mean()), "accuracy_b": float(cb.mean()),
            "accuracy_diff_a_minus_b": float(ca.mean() - cb.mean()),
            "f1_macro_a": _macro_f1(m["true_name_a"].to_numpy(), m["pred_name_a"].to_numpy(), names),
            "f1_macro_b": _macro_f1(m["true_name_b"].to_numpy(), m["pred_name_b"].to_numpy(), names),
            "mcnemar": mcnemar_test(ca, cb),
        })
        if n_boot > 0:
            diffs_acc, diffs_f1 = [], []
            ta, pa = m["true_name_a"].to_numpy(), m["pred_name_a"].to_numpy()
            tb, pb = m["true_name_b"].to_numpy(), m["pred_name_b"].to_numpy()
            for _ in range(n_boot):
                i = rng.integers(0, len(m), len(m))
                diffs_acc.append(ca[i].mean() - cb[i].mean())
                diffs_f1.append(_macro_f1(ta[i], pa[i], names) - _macro_f1(tb[i], pb[i], names))
            res["accuracy_diff_ci95"] = [float(np.quantile(diffs_acc, q)) for q in (0.025, 0.975)]
            res["f1_macro_diff_ci95"] = [float(np.quantile(diffs_f1, q)) for q in (0.025, 0.975)]
        res["p_value"] = res["mcnemar"]["p_value"]
        return res

    # Unpaired: different specimens in A and B.
    ca, cb = a["correct"].to_numpy(bool), b["correct"].to_numpy(bool)
    table = [[int(ca.sum()), int((~ca).sum())], [int(cb.sum()), int((~cb).sum())]]
    odds, p = fisher_exact_2x2(table)
    res.update({
        "note": "No shared specimens: McNemar's test requires paired data, Fisher's exact test used instead.",
        "n_a": int(len(a)), "n_b": int(len(b)),
        "accuracy_a": float(ca.mean()), "accuracy_b": float(cb.mean()),
        "accuracy_diff_a_minus_b": float(ca.mean() - cb.mean()),
        "f1_macro_a": _macro_f1(a["true_name"].to_numpy(), a["pred_name"].to_numpy(), names),
        "f1_macro_b": _macro_f1(b["true_name"].to_numpy(), b["pred_name"].to_numpy(), names),
        "fisher_exact": {"table": table, "odds_ratio": float(odds), "p_value": float(p)},
        "p_value": float(p),
    })
    if n_boot > 0:
        def cluster_idx(df):
            groups = df.groupby("sample_id").indices
            keys = list(groups)
            pick = rng.integers(0, len(keys), len(keys))
            return np.concatenate([groups[keys[j]] for j in pick])
        diffs_acc, diffs_f1 = [], []
        for _ in range(n_boot):
            ia, ib = cluster_idx(a), cluster_idx(b)
            diffs_acc.append(ca[ia].mean() - cb[ib].mean())
            diffs_f1.append(_macro_f1(a["true_name"].to_numpy()[ia], a["pred_name"].to_numpy()[ia], names)
                            - _macro_f1(b["true_name"].to_numpy()[ib], b["pred_name"].to_numpy()[ib], names))
        res["accuracy_diff_ci95"] = [float(np.quantile(diffs_acc, q)) for q in (0.025, 0.975)]
        res["f1_macro_diff_ci95"] = [float(np.quantile(diffs_f1, q)) for q in (0.025, 0.975)]
    return res


def run_compare(args) -> None:
    a_all, b_all = pd.read_csv(args.a), pd.read_csv(args.b)
    tasks = ["binary", "type"] if args.task == "both" else [args.task]
    results = {"a": {"file": args.a, "name": args.name_a}, "b": {"file": args.b, "name": args.name_b}, "tasks": {}}
    for task in tasks:
        a, b = a_all[a_all["task"] == task], b_all[b_all["task"] == task]
        if a.empty or b.empty:
            log.warning("Task '%s' missing in one of the files; skipped", task)
            continue
        if set(_prob_cols(a)) != set(_prob_cols(b)):
            msg = (f"different class sets ({sorted(c[3:] for c in _prob_cols(a))} vs "
                   f"{sorted(c[3:] for c in _prob_cols(b))}); accuracies of different tasks are not "
                   "comparable - harmonise the classes first")
            log.warning("Task '%s' skipped: %s", task, msg)
            results["skipped"] = {**results.get("skipped", {}), task: msg}
            continue
        results["tasks"][task] = compare_predictions(a.reset_index(drop=True), b.reset_index(drop=True),
                                                     args.pair_on, args.bootstrap, args.seed)
    adjusted = holm({t: r["p_value"] for t, r in results["tasks"].items()})
    for t, p in adjusted.items():
        results["tasks"][t]["p_value_holm"] = p
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_safe(results), indent=2), encoding="utf-8")
    for t, r in results["tasks"].items():
        log.info("[%s] %s acc=%.3f vs %s acc=%.3f | diff=%+.3f | p=%.4g (Holm %.4g) | pairing=%s",
                 t, args.name_a, r["accuracy_a"], args.name_b, r["accuracy_b"],
                 r["accuracy_diff_a_minus_b"], r["p_value"], r["p_value_holm"], r["pairing"])
    log.info("Wrote %s", out)


# =========================================================================
# Prediction
# =========================================================================
def aggregate_record(task: str, tile_probs: np.ndarray, asb_mask: np.ndarray, how: str,
                     threshold: float, type_decision: str) -> tuple[np.ndarray, int]:
    """Aggregate tile-level class probabilities into one image-level prediction.

    binary: P(asbestos) = max / mean / noisy-OR over tiles (max = "any tile
            positive", the natural multiple-instance rule for sparse fibres).
    type:   hierarchical - P(asbestos) aggregated as above; the asbestos type
            is the class with the largest summed probability over all tiles;
            non-asbestos mass is spread by the mean non-asbestos distribution.
    """
    def agg(p: np.ndarray) -> float:
        if how == "max":
            return float(p.max())
        if how == "mean":
            return float(p.mean())
        return float(1.0 - np.prod(1.0 - np.clip(p, 0, 1)))

    if task == "binary":
        p_pos = agg(tile_probs[:, 1])
        return np.array([1 - p_pos, p_pos]), int(p_pos >= threshold)

    if type_decision == "argmax" or asb_mask.all() or not asb_mask.any():
        probs = tile_probs.mean(axis=0)
        return probs, int(np.argmax(probs))
    p_asb = agg(tile_probs[:, asb_mask].sum(axis=1))
    asb_dist = tile_probs[:, asb_mask].sum(axis=0)
    asb_dist = asb_dist / max(asb_dist.sum(), 1e-12)
    neg_dist = tile_probs[:, ~asb_mask].mean(axis=0)
    neg_dist = neg_dist / max(neg_dist.sum(), 1e-12)
    probs = np.zeros(tile_probs.shape[1])
    probs[asb_mask] = p_asb * asb_dist
    probs[~asb_mask] = (1 - p_asb) * neg_dist
    idx_asb, idx_neg = np.flatnonzero(asb_mask), np.flatnonzero(~asb_mask)
    pred = idx_asb[np.argmax(probs[idx_asb])] if p_asb >= threshold else idx_neg[np.argmax(probs[idx_neg])]
    return probs, int(pred)


def parse_answer(text: str, candidates: list[str]) -> int:
    """Map free-text output to a candidate index (-1 if invalid)."""
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    t = norm(text)
    for i, c in enumerate(candidates):
        if t == norm(c):
            return i
    hits = [(len(norm(c)), i) for i, c in enumerate(candidates) if re.search(rf"\b{re.escape(norm(c))}\b", t)]
    return max(hits)[1] if hits else -1


class ScoringCollator:
    """Builds model batches for the chosen scoring mode inside DataLoader workers."""

    def __init__(self, processor, space, mode: str) -> None:
        from src.dataset import VLMCollator
        self.space = space
        self.mode = mode
        self.inner = VLMCollator(processor, "score" if mode == "full" else "generate")

    def __call__(self, items: list[dict]):
        if self.mode == "full":
            expanded, counts = [], []
            for it in items:
                cands = self.space.answers(it["task"])
                expanded += [{**it, "answer": c} for c in cands]
                counts.append(len(cands))
            return self.inner(expanded), items_meta(items), counts
        return self.inner(items), items_meta(items), None


def _maps(space, label: str) -> bool:
    """True if the label is a class (or an explicitly dropped label) of the space."""
    try:
        space.map(label)
        return True
    except ValueError:
        return False


def items_meta(items: list[dict]) -> list[dict]:
    return [{"index": it["index"], "task": it["task"], "target": it["target"]} for it in items]


def run_predict(args) -> None:
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from src.dataset import AsbestosVLMDataset, DataConfig
    from src.labels import DEFAULT_CONFIGS, LabelSpace
    from src.manifest import read_manifest
    from src.model import DEFAULT_MODEL, load_model_for_inference, load_processor

    adapter = None if str(args.adapter_dir).lower() in ("none", "") else Path(args.adapter_dir)
    train_cfg = {}
    if adapter and (adapter / "train_config.json").exists():
        train_cfg = json.loads((adapter / "train_config.json").read_text(encoding="utf-8"))
    targs = train_cfg.get("args", {})

    # Label space: explicit config > adapter's saved label space.
    if args.class_config:
        space = LabelSpace.load(args.class_config, args.merge, args.rename, args.drop_class)
    elif adapter and (adapter / "label_space.json").exists():
        space = LabelSpace.load(adapter / "label_space.json", args.merge, args.rename, args.drop_class)
    else:
        raise SystemExit("--class_config is required when the adapter has no label_space.json")
    log.info("\n%s", space.describe())

    cfg_dict = dict(train_cfg.get("data_config", {}))
    cfg_dict.update({"tile_label_source": "image", "online_aug": False, "shuffle_options": False,
                     "max_tiles_per_image": 0, "exclude_augmented": True})
    if args.modality:
        cfg_dict["modality"] = args.modality
    cfg_dict.setdefault("modality", "unified")
    if args.task:
        cfg_dict["tasks"] = ["binary", "type"] if args.task == "both" else [args.task]
    if args.tile_size is not None:
        cfg_dict["tile_size"] = args.tile_size
    if args.tile_overlap is not None:
        cfg_dict["tile_overlap"] = args.tile_overlap
    cfg = DataConfig.from_dict(cfg_dict)
    log.info("Data config: %s", cfg.to_dict())

    manifest = args.manifest or str(Path(args.data_dir) / "manifest.jsonl")
    # Cross-domain: the test set may use classes the model was never trained on
    # (micro and macro datasets have different type classes). Binary detection is
    # still well defined, so evaluate it with a label space that knows the test
    # labels, and skip the type task.
    unknown = sorted({r["label"] for r in read_manifest(manifest)
                      if r.get("split") == args.split and "augmented_from" not in r
                      and cfg.modality in ("unified", r["modality"]) and not _maps(space, r["label"])})
    if unknown:
        log.warning("Test labels %s are not classes of this model: evaluating the binary task only "
                    "(type classes differ between the datasets).", unknown)
        truth = LabelSpace.load(DEFAULT_CONFIGS["unified"])
        truth.binary_positive, truth.binary_negative = space.binary_positive, space.binary_negative
        space = truth
        cfg.tasks = ("binary",)

    ds = AsbestosVLMDataset(manifest, args.data_dir, space, cfg, split=args.split, train=False)
    if args.limit:
        keep = {it.rec_idx for it in ds.items[: args.limit]}
        ds.items = [it for it in ds.items if it.rec_idx in keep]
    if len(ds) == 0:
        raise SystemExit("No evaluation examples - check --split/--modality.")
    log.info("Eval: %s", ds.summary())

    model_name = args.base_model or targs.get("model_name", DEFAULT_MODEL)
    proc_src = str(adapter) if adapter and (adapter / "preprocessor_config.json").exists() else model_name
    # Pixel budget: explicit CLI value > training value > default.
    min_px = args.min_pixels or targs.get("min_pixels", 256 * 28 * 28)
    max_px = args.max_pixels or targs.get("max_pixels", 1024 * 28 * 28)
    processor = load_processor(proc_src, min_px, max_px)
    model = load_model_for_inference(model_name, adapter, args.quant or targs.get("quant", "4bit"),
                                     args.attn_implementation)
    device = next(model.parameters()).device
    tok = processor.tokenizer

    first_ids = {}
    if args.scoring == "first_token":
        for task in cfg.tasks:
            ids = [tok.encode(a, add_special_tokens=False)[0] for a in space.answers(task)]
            if len(set(ids)) != len(ids):
                raise SystemExit(f"Answers for task '{task}' share a first token; use --scoring full.")
            first_ids[task] = torch.tensor(ids, device=device)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        collate_fn=ScoringCollator(processor, space, args.scoring))
    tile_probs: dict[int, np.ndarray] = {}
    raw_text: dict[int, str] = {}

    with torch.inference_mode():
        for batch, metas, counts in tqdm(loader, desc=f"predict[{args.scoring}]"):
            labels = batch.pop("labels", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            if args.scoring == "full":
                logits = model(**batch, use_cache=False).logits[:, :-1]
                tgt = labels[:, 1:].to(device)
                mask = tgt != -100
                rows = mask.nonzero(as_tuple=True)[0]
                tok_lp = torch.log_softmax(logits[mask].float(), dim=-1).gather(1, tgt[mask][:, None]).squeeze(1)
                seq_lp = torch.zeros(tgt.size(0), device=device).index_add_(0, rows, tok_lp)
                off = 0
                for meta, c in zip(metas, counts):
                    tile_probs[meta["index"]] = torch.softmax(seq_lp[off:off + c], 0).cpu().numpy()
                    off += c
            elif args.scoring == "first_token":
                last = torch.log_softmax(model(**batch, use_cache=False).logits[:, -1].float(), dim=-1)
                for row, meta in enumerate(metas):
                    tile_probs[meta["index"]] = torch.softmax(last[row, first_ids[meta["task"]]], 0).cpu().numpy()
            else:
                gen = model.generate(**batch, max_new_tokens=args.max_new_tokens, do_sample=False)
                texts = tok.batch_decode(gen[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
                for meta, text in zip(metas, texts):
                    cands = space.answers(meta["task"])
                    k = parse_answer(text, cands)
                    p = np.full(len(cands), 1.0 / len(cands)) if k < 0 else np.eye(len(cands))[k]
                    tile_probs[meta["index"]] = p
                    raw_text[meta["index"]] = text.strip()

    # ---------------- tile rows + aggregation to image level ----------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tile_rows, groups = [], defaultdict(list)
    for idx, it in enumerate(ds.items):
        if idx not in tile_probs:
            continue
        rec = ds.records[it.rec_idx]
        groups[(it.rec_idx, it.task)].append(idx)
        tile_rows.append({"record_id": rec["id"], "task": it.task, "tile": it.tile_idx,
                          "crop": json.dumps(it.crop) if it.crop else "",
                          "pred": int(np.argmax(tile_probs[idx])), "raw_output": raw_text.get(idx, ""),
                          **{f"p::{n}": float(v) for n, v in zip(space.answers(it.task), tile_probs[idx])}})
    pd.DataFrame(tile_rows).to_csv(out_dir / "tiles.csv", index=False)

    asb_mask = np.array([c.asbestos for c in space.classes])
    rows = []
    for (ri, task), idxs in groups.items():
        rec = ds.records[ri]
        names = space.answers(task)
        key = space.map(rec["label"])
        y_true = space.target_index(key, task)
        invalid = args.scoring == "generate" and all(parse_answer(raw_text[i], names) < 0 for i in idxs)
        probs, y_pred = aggregate_record(task, np.stack([tile_probs[i] for i in idxs]), asb_mask,
                                         args.aggregate, args.threshold, args.type_decision)
        if invalid:
            y_pred = -1
        rows.append({"record_id": rec["id"], "sample_id": rec.get("sample_id", rec["id"]),
                     "modality": rec["modality"], "task": task, "n_tiles": len(idxs),
                     "y_true": y_true, "y_pred": y_pred, "true_name": names[y_true],
                     "pred_name": names[y_pred] if y_pred >= 0 else "INVALID",
                     "correct": bool(y_pred == y_true),
                     **{f"p::{n}": float(v) for n, v in zip(names, probs)}})
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    # ---------------------------- metrics ----------------------------------
    metrics = {"adapter": str(adapter) if adapter else None, "model": model_name, "split": args.split,
               "scoring": args.scoring, "aggregate": args.aggregate, "threshold": args.threshold,
               "type_decision": args.type_decision, "data_config": cfg.to_dict(), "tasks": {}}
    for task in cfg.tasks:
        df = pred_df[pred_df["task"] == task]
        if df.empty:
            continue
        names = space.answers(task)
        pcols = [f"p::{n}" for n in names]
        yt, yp = df["y_true"].to_numpy(), df["y_pred"].to_numpy()
        m = classification_metrics(yt, yp, df[pcols].to_numpy(), names)
        m["ci95"] = bootstrap_ci(yt, yp, df["sample_id"].astype(str).to_numpy(), len(names),
                                 args.bootstrap, args.seed)
        if df["modality"].nunique() > 1:
            m["by_modality"] = {}
            for mod, sub in df.groupby("modality"):
                m["by_modality"][mod] = classification_metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy(),
                                                               sub[pcols].to_numpy(), names)
        metrics["tasks"][task] = m
        cm = np.array(m["confusion_matrix"])
        pd.DataFrame(cm, index=[f"true::{n}" for n in names], columns=[f"pred::{n}" for n in names]) \
            .to_csv(out_dir / f"confusion_{task}.csv")
        pd.DataFrame(m["per_class"]).T.to_csv(out_dir / f"per_class_{task}.csv")
        plot_confusion(cm, names, out_dir / f"confusion_{task}.png", f"{task} | {cfg.modality} | {args.split}")
        log.info("[%s] n=%d acc=%.3f bal_acc=%s macroF1=%.3f kappa=%.3f invalid=%d", task, m["n"], m["accuracy"],
                 f"{m['balanced_accuracy']:.3f}" if m["balanced_accuracy"] is not None else "n/a",
                 m["f1_macro"], m["cohen_kappa"], m["invalid_predictions"])
    (out_dir / "metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    log.info("Wrote predictions, metrics and confusion matrices to %s", out_dir)


# =========================================================================
# CLI
# =========================================================================
def main() -> None:
    from src.labels import add_label_args

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="Predict on a split and compute metrics.")
    p.add_argument("--adapter_dir", required=True, help="Trained output_dir, or 'none' for zero-shot.")
    p.add_argument("--base_model", default=None, help="Default: model used for training.")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--modality", choices=["microscopic", "smartphone", "unified"], default=None,
                   help="Images to evaluate on (default: training modality). Use for cross-domain tests.")
    p.add_argument("--task", choices=["binary", "type", "both"], default=None)
    p.add_argument("--tile_size", type=int, default=None, help="Default: as in training.")
    p.add_argument("--tile_overlap", type=float, default=None)
    p.add_argument("--scoring", choices=["full", "first_token", "generate"], default="full")
    p.add_argument("--aggregate", choices=["max", "mean", "noisy_or"], default="max")
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold on P(asbestos).")
    p.add_argument("--type_decision", choices=["hierarchical", "argmax"], default="hierarchical")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--quant", choices=["4bit", "8bit", "none"], default=None)
    p.add_argument("--attn_implementation", default="sdpa")
    p.add_argument("--min_pixels", type=int, default=None, help="Default: as in training.")
    p.add_argument("--max_pixels", type=int, default=None, help="Default: as in training.")
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--limit", type=int, default=0, help="Debug: evaluate only the first N examples.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", required=True)
    add_label_args(p)
    p.set_defaults(class_config=None)

    c = sub.add_parser("compare", help="Statistically compare two predictions.csv files.")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--name_a", default="A")
    c.add_argument("--name_b", default="B")
    c.add_argument("--task", choices=["binary", "type", "both"], default="both")
    c.add_argument("--pair_on", default="auto", help="auto | sample_id | record_id | none")
    c.add_argument("--bootstrap", type=int, default=2000)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--output", required=True)

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_predict(args) if args.cmd == "predict" else run_compare(args)


if __name__ == "__main__":
    main()

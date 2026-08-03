"""Shared threshold sweep utilities for TC2/TC3 and TC4/TC5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

MARGIN_WARN = 0.01


@dataclass(frozen=True)
class SweepPoint:
    threshold: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class SweepResult:
    points: list[SweepPoint]
    feasible: list[float]
    t_min: float | None
    t_max: float | None
    recommended: float | None
    recommended_metrics: dict[str, float] | None
    recommended_margins: dict[str, float] | None
    min_normalized_margin: float | None
    infeasible: bool


def thresholds_01(step: float = 0.01) -> list[float]:
    n = int(round(1.0 / step))
    return [round(i * step, 2) for i in range(n + 1)]


def normalized_margins(
    metrics: dict[str, float],
    targets: dict[str, float],
) -> dict[str, float]:
    """(metric - target) / target for each constrained metric."""
    out: dict[str, float] = {}
    for name, target in targets.items():
        if target == 0:
            out[name] = float("inf") if metrics[name] >= 0 else float("-inf")
        else:
            out[name] = (metrics[name] - target) / target
    return out


def select_threshold_maximin(
    points: Sequence[SweepPoint],
    feasible: Sequence[float],
    targets: dict[str, float],
) -> tuple[float, dict[str, float], dict[str, float], float]:
    """t* = argmax_t min_i (metric_i(t) - target_i) / target_i over feasible t."""
    by_t = {p.threshold: p for p in points}
    best_t: float | None = None
    best_score = float("-inf")
    best_metrics: dict[str, float] = {}
    best_margins: dict[str, float] = {}

    for t in feasible:
        metrics = by_t[t].metrics
        margins = normalized_margins(metrics, targets)
        score = min(margins.values())
        # Tie-break: prefer higher absolute min metric, then lower threshold
        if score > best_score or (
            score == best_score
            and (
                min(metrics[n] for n in targets) > min(best_metrics.get(n, -1) for n in targets)
                or (
                    min(metrics[n] for n in targets)
                    == min(best_metrics.get(n, -1) for n in targets)
                    and (best_t is None or t < best_t)
                )
            )
        ):
            best_score = score
            best_t = t
            best_metrics = dict(metrics)
            best_margins = margins

    assert best_t is not None
    return best_t, best_metrics, best_margins, best_score


def sweep_thresholds(
    scores: Sequence[float],
    labels: Sequence[int],
    metric_fns: dict[str, Callable[[Sequence[float], Sequence[int], float], float]],
    targets: dict[str, float],
    *,
    higher_is_positive: bool = True,
    step: float = 0.01,
) -> SweepResult:
    """Sweep thresholds; a point is feasible when ALL target constraints hold.

    targets maps metric_name -> minimum required value (all are >= constraints).
    Recommended t* maximises the worst-case normalised margin among targets.
    """
    del higher_is_positive  # reserved; binary_confusion defaults to higher=positive
    points: list[SweepPoint] = []
    feasible: list[float] = []

    for t in thresholds_01(step):
        metrics = {name: fn(scores, labels, t) for name, fn in metric_fns.items()}
        points.append(SweepPoint(threshold=t, metrics=metrics))
        if all(metrics[name] >= min_val for name, min_val in targets.items()):
            feasible.append(t)

    if not feasible:
        return SweepResult(
            points=points,
            feasible=[],
            t_min=None,
            t_max=None,
            recommended=None,
            recommended_metrics=None,
            recommended_margins=None,
            min_normalized_margin=None,
            infeasible=True,
        )

    t_min, t_max = feasible[0], feasible[-1]
    recommended, rec_metrics, rec_margins, min_norm = select_threshold_maximin(
        points, feasible, targets
    )
    return SweepResult(
        points=points,
        feasible=feasible,
        t_min=t_min,
        t_max=t_max,
        recommended=recommended,
        recommended_metrics=rec_metrics,
        recommended_margins=rec_margins,
        min_normalized_margin=min_norm,
        infeasible=False,
    )


def binary_confusion(
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float,
    *,
    higher_is_positive: bool = True,
) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for score, label in zip(scores, labels, strict=True):
        pred = 1 if (score >= threshold if higher_is_positive else score <= threshold) else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def tpr(scores: Sequence[float], labels: Sequence[int], threshold: float) -> float:
    tp, _, _, fn = binary_confusion(scores, labels, threshold)
    return tp / (tp + fn) if (tp + fn) else 0.0


def fpr(scores: Sequence[float], labels: Sequence[int], threshold: float) -> float:
    _, fp, tn, _ = binary_confusion(scores, labels, threshold)
    return fp / (fp + tn) if (fp + tn) else 0.0


def precision(scores: Sequence[float], labels: Sequence[int], threshold: float) -> float:
    tp, fp, _, _ = binary_confusion(scores, labels, threshold)
    return tp / (tp + fp) if (tp + fp) else 0.0


def f1(scores: Sequence[float], labels: Sequence[int], threshold: float) -> float:
    p = precision(scores, labels, threshold)
    r = tpr(scores, labels, threshold)
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def accuracy(scores: Sequence[float], labels: Sequence[int], threshold: float) -> float:
    tp, fp, tn, fn = binary_confusion(scores, labels, threshold)
    total = tp + fp + tn + fn
    return (tp + tn) / total if total else 0.0


def write_sweep_csv(
    path: str | Path,
    points: Sequence[SweepPoint],
    metric_names: Sequence[str],
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = "threshold," + ",".join(metric_names)
    lines = [header]
    for p in points:
        cols = [f"{p.threshold:.2f}"] + [f"{p.metrics[m]:.6f}" for m in metric_names]
        lines.append(",".join(cols))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def warn_thin_margins(margins: dict[str, float] | None, *, warn_below: float = MARGIN_WARN) -> bool:
    """Print warning if any normalised margin is below warn_below. Returns True if warned."""
    if not margins:
        return False
    thin = {k: v for k, v in margins.items() if v < warn_below}
    if not thin:
        return False
    print()
    print("!" * 60)
    print(f"WARNING: normalised margin < {warn_below} at t* — field remeasure may flip")
    for k, v in thin.items():
        print(f"  {k}: margin={v:.6f}")
    print("!" * 60)
    return True

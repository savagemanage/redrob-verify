"""Build a self-contained HTML evaluation report (base64-embedded charts).

No web server. One file: results/report.html.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from harness.origin import html_banner_for_results
from harness.config_util import load_config, resolve_results_root
from harness.freeze import write_freeze


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _forgery_sweep_chart(data: dict[str, Any], *, domain: str | None = None) -> str:
    """Plot sweep for a domain payload or legacy top-level tc23 blob."""
    if domain and isinstance(data.get("domains"), dict):
        data = data["domains"].get(domain) or data.get(domain) or {}
    sweep = data.get("sweep") or []
    xs = [p["threshold"] for p in sweep]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not xs:
        ax.set_title(f"TC2/TC3 Forgery Sweep ({domain or 'gate'}) — empty")
        return _fig_to_b64(fig)
    ax.plot(xs, [p["tpr"] for p in sweep], label="TPR (TC2)", linewidth=2)
    ax.plot(xs, [p["precision"] for p in sweep], label="Precision", linewidth=1.5, linestyle="--")
    ax.plot(xs, [p["f1"] for p in sweep], label="F1 (TC3)", linewidth=2)

    interval = data.get("feasible_interval")
    if interval:
        ax.axvspan(interval[0], interval[1], alpha=0.18, color="green", label="Feasible")
    rec = data.get("recommended_threshold")
    if rec is not None:
        ax.axvline(rec, color="black", linestyle=":", linewidth=1.5, label=f"t*={rec:.2f}")

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    title = "TC2 / TC3 Forgery Threshold Sweep"
    if domain:
        title += f" — {domain}"
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return _fig_to_b64(fig)


def _face_sweep_chart(data: dict[str, Any]) -> str:
    sweep = data.get("sweep") or []
    xs = [p["threshold"] for p in sweep]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, [p["sensitivity"] for p in sweep], label="Sensitivity (TC4)", linewidth=2)
    ax.plot(xs, [p["accuracy"] for p in sweep], label="Accuracy (TC5)", linewidth=2)

    interval = data.get("feasible_interval")
    if interval:
        ax.axvspan(interval[0], interval[1], alpha=0.18, color="green", label="Feasible")
    rec = data.get("recommended_threshold")
    if rec is not None:
        ax.axvline(rec, color="black", linestyle=":", linewidth=1.5, label=f"t*={rec:.2f}")

    targets = data.get("targets") or {}
    if "sensitivity_min" in targets:
        ax.axhline(targets["sensitivity_min"], color="C0", alpha=0.35, linestyle=":")
    if "accuracy_min" in targets:
        ax.axhline(targets["accuracy_min"], color="C1", alpha=0.35, linestyle=":")

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.set_title("TC4 / TC5 Face Threshold Sweep")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    return _fig_to_b64(fig)


def _cer_hist(data: dict[str, Any]) -> str:
    values = [
        item["cer_field"]
        for item in data.get("per_item") or []
        if item.get("cer_field") is not None
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    if values:
        ax.hist(values, bins=min(20, max(5, len(values))), color="#3d5a80", edgecolor="white")
    ax.axvline(data.get("target_max", 0.21), color="#e63946", linestyle="--", label="target max")
    if data.get("cer_field") is not None:
        ax.axvline(
            data["cer_field"],
            color="#2a9d8f",
            linestyle="-",
            label=f"cer_field={data['cer_field']:.4f}",
        )
    ax.set_xlabel("cer_field")
    ax.set_ylabel("Count")
    ax.set_title("TC1 cer_field Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _fig_to_b64(fig)


def _tc6_hist(data: dict[str, Any]) -> str:
    values = data.get("latencies_seconds") or []
    fig, ax = plt.subplots(figsize=(7, 4))
    if values:
        ax.hist(values, bins=min(30, max(8, int(np.sqrt(len(values))))), color="#264653", edgecolor="white")
    ax.axvline(data.get("target_max", 60.0), color="#e63946", linestyle="--", label="target max")
    if data.get("value") is not None:
        ax.axvline(data["value"], color="#2a9d8f", linestyle="-", label=f"mean={data['value']:.4f}s")
    ax.set_xlabel("Response time (s)")
    ax.set_ylabel("Count")
    ax.set_title("TC6 Identity Response Time Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _fig_to_b64(fig)


def _fmt(v: Any, digits: int = 6) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _pass_badge(ok: bool | None) -> str:
    if ok is True:
        return '<span class="pass">PASS</span>'
    if ok is False:
        return '<span class="fail">FAIL</span>'
    return '<span class="na">N/A</span>'


def _tc1_detail_html(
    tc1: dict[str, Any] | None,
    classic: dict[str, Any] | None,
    vl: dict[str, Any] | None,
    ablation: list[Any] | None,
) -> str:
    parts: list[str] = []
    # Backend comparison
    parts.append("<h3>Backend comparison (cer_field)</h3><table><thead><tr>"
                 "<th>backend</th><th>cer_field</th><th>n</th><th>passed</th></tr></thead><tbody>")
    for label, payload in (("active", tc1), ("paddleocr_classic", classic), ("paddleocr_vl", vl)):
        if not payload:
            parts.append(f"<tr><td>{label}</td><td class='na'>—</td><td class='na'>—</td><td class='na'>—</td></tr>")
            continue
        parts.append(
            f"<tr><td>{payload.get('backend', label)}</td>"
            f"<td>{_fmt(payload.get('cer_field'))}</td>"
            f"<td>{payload.get('n')}</td>"
            f"<td>{_pass_badge(payload.get('passed'))}</td></tr>"
        )
    parts.append("</tbody></table>")

    # Script breakdown from active (or VL) payload
    script_src = tc1 or vl or classic
    if script_src and script_src.get("cer_field_by_script"):
        parts.append("<h3>cer_field by script</h3><table><thead><tr>"
                     "<th>script</th><th>cer_field</th><th>n</th></tr></thead><tbody>")
        for script, row in sorted((script_src["cer_field_by_script"] or {}).items()):
            parts.append(
                f"<tr><td>{script}</td><td>{_fmt(row.get('cer_field'))}</td>"
                f"<td>{row.get('n')}</td></tr>"
            )
        parts.append("</tbody></table>")

    if ablation:
        parts.append("<h3>Preprocess ablation (cer_field)</h3><table><thead><tr>"
                     "<th>ablation</th><th>cer_field</th><th>mean_ms</th><th>backend</th></tr></thead><tbody>")
        for row in ablation:
            parts.append(
                f"<tr><td>{row.get('ablation')}</td>"
                f"<td>{_fmt(row.get('cer_field'))}</td>"
                f"<td>{_fmt(row.get('mean_latency_ms'), 1)}</td>"
                f"<td>{row.get('backend')}</td></tr>"
            )
        parts.append("</tbody></table>")
    else:
        parts.append("<p class='missing'>No ocr_ablation.json — run tools/ablate_ocr.py</p>")
    return "\n".join(parts)


def _forgery_train_html(train: dict[str, Any] | None) -> str:
    if not train:
        return "<p class='missing'>No forgery_train.json</p>"
    rows = train.get("epochs") or []
    parts = [
        f"<p>device={train.get('device')} best_joint_width={_fmt(train.get('best_joint_interval_width'))} "
        f"checkpoint={train.get('best_checkpoint')}</p>",
        "<table><thead><tr><th>epoch</th><th>AUC</th><th>ROC gate</th>"
        "<th>TPR</th><th>FPR</th><th>joint_width</th></tr></thead><tbody>",
    ]
    for row in rows:
        gate = row.get("roc_gate") or {}
        parts.append(
            f"<tr><td>{row.get('epoch')}</td><td>{_fmt(row.get('auc'))}</td>"
            f"<td>{_pass_badge(bool(gate.get('passed')))}</td>"
            f"<td>{_fmt(gate.get('tpr'))}</td><td>{_fmt(gate.get('fpr'))}</td>"
            f"<td>{_fmt(row.get('joint_interval_width'))}</td></tr>"
        )
    parts.append("</tbody></table>")
    return "\n".join(parts)


def build_report(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    results = resolve_results_root(cfg)
    freeze = write_freeze(cfg)

    preflight = _load_json(results / "preflight.json")
    tc1 = _load_json(results / "tc1_cer.json")
    tc1_classic = _load_json(results / "tc1_cer_paddleocr_classic.json")
    tc1_vl = _load_json(results / "tc1_cer_paddleocr_vl.json")
    ablation = _load_json(results / "ocr_ablation.json")
    forgery_train = _load_json(results / "forgery_train.json")
    tc23 = _load_json(results / "tc2_tc3_forgery.json")
    tc45 = _load_json(results / "tc4_tc5_face.json")
    tc6 = _load_json(results / "tc6_identity.json")
    origin_banner = html_banner_for_results([tc1, tc23, tc45, tc6])

    def _origin_block(payload: dict[str, Any] | None, title: str) -> str:
        if not payload:
            return f"<tr><td>{title}</td><td class='na'>—</td><td class='na'>—</td></tr>\n"
        dist = payload.get("origin_distribution") or {}
        dist_txt = ", ".join(f"{k}={v}" for k, v in sorted(dist.items())) or "—"
        tta = payload.get("tta_valid")
        tta_txt = "true" if tta is True else "false" if tta is False else "—"
        return f"<tr><td>{title}</td><td>{dist_txt}</td><td>{tta_txt}</td></tr>\n"

    origin_rows = (
        _origin_block(tc1, "TC1 / 1_ocr")
        + _origin_block(tc23, "TC2–3 / 2_forgery")
        + _origin_block(tc45, "TC4–5 / 3_face")
        + _origin_block(tc6, "TC6 / 4_resume")
    )

    preflight_rows = ""
    if preflight and preflight.get("services"):
        for entry in preflight["services"]:
            meta = entry.get("meta") or {}
            preflight_rows += (
                f"<tr><td>{entry.get('endpoint')}</td>"
                f"<td>{meta.get('service')}</td>"
                f"<td>{meta.get('backend')}</td>"
                f"<td>{meta.get('version')}</td>"
                f"<td>{meta.get('model_sha256')}</td>"
                f"<td>{meta.get('git_commit')}</td>"
                f"<td>{meta.get('started_at')}</td></tr>\n"
            )
    elif preflight and preflight.get("errors"):
        preflight_rows = (
            "<tr><td colspan='7' class='fail'>PREFLIGHT FAILED: "
            + "; ".join(str(e) for e in preflight["errors"])
            + "</td></tr>\n"
        )
    else:
        preflight_rows = (
            "<tr><td colspan='7' class='na'>No preflight.json - run ./run.sh preflight</td></tr>\n"
        )
    preflight_ok = preflight.get("ok") if preflight else None

    charts: dict[str, str] = {}
    if tc23:
        charts["forgery_sweep"] = _forgery_sweep_chart(tc23)
        if tc23.get("in_domain"):
            charts["forgery_sweep_in"] = _forgery_sweep_chart(tc23, domain="in_domain")
        if tc23.get("cross_domain"):
            charts["forgery_sweep_cross"] = _forgery_sweep_chart(tc23, domain="cross_domain")
    if tc45:
        charts["face_sweep"] = _face_sweep_chart(tc45)
    if tc1:
        charts["cer_hist"] = _cer_hist(tc1)
    if tc6:
        charts["tc6_hist"] = _tc6_hist(tc6)

    targets = cfg["targets"]
    rows = [
        {
            "id": "TC1",
            "name": "서류 인식 cer_field",
            "target": f"< {targets['tc1_cer_max']}",
            "value": (_fmt(tc1.get("cer_field")) if tc1 else None),
            "ok": tc1["passed"] if tc1 else None,
        },
        {
            "id": "TC2",
            "name": "위·변조 탐지 TPR",
            "target": f">= {targets['tc2_tpr_min']}",
            "value": _fmt(tc23["tc2_tpr"]) if tc23 else None,
            "ok": (not tc23["infeasible"]) if tc23 else None,
        },
        {
            "id": "TC3",
            "name": "위·변조 탐지 F1",
            "target": f">= {targets['tc3_f1_min']}",
            "value": _fmt(tc23["tc3_f1"]) if tc23 else None,
            "ok": (not tc23["infeasible"]) if tc23 else None,
        },
        {
            "id": "TC4",
            "name": "얼굴 대조 민감도",
            "target": f">= {targets['tc4_sensitivity_min']}",
            "value": _fmt(tc45["tc4_sensitivity"]) if tc45 else None,
            "ok": (not tc45["infeasible"]) if tc45 else None,
        },
        {
            "id": "TC5",
            "name": "얼굴 대조 정확도",
            "target": f">= {targets['tc5_accuracy_min']}",
            "value": _fmt(tc45["tc5_accuracy"]) if tc45 else None,
            "ok": (not tc45["infeasible"]) if tc45 else None,
        },
        {
            "id": "TC6",
            "name": "신원 취합 시간 (s)",
            "target": f"< {targets['tc6_seconds_max']}",
            "value": _fmt(tc6["value"]) if tc6 else None,
            "ok": tc6["passed"] if tc6 else None,
        },
    ]

    table_rows = "\n".join(
        f"<tr><td>{r['id']}</td><td>{r['name']}</td><td>{r['target']}</td>"
        f"<td>{r['value'] or '-'}</td><td>{_pass_badge(r['ok'])}</td></tr>"
        for r in rows
    )

    def img(key: str, caption: str) -> str:
        if key not in charts:
            return f"<p class='missing'>Missing chart: {caption}</p>"
        return (
            f"<figure><img alt=\"{caption}\" src=\"data:image/png;base64,{charts[key]}\"/>"
            f"<figcaption>{caption}</figcaption></figure>"
        )

    infeasible_notes = []
    if tc23 and tc23.get("missing_cross_domain"):
        infeasible_notes.append(
            "<p class='infeasible'>TC2/TC3: <strong>INFEASIBLE</strong> — "
            "cross-domain (FMIDV) set missing. Pass gate is cross-domain; "
            "in-domain (gen_forgery) alone does not count.</p>"
        )
    if tc23 and tc23.get("in_domain"):
        ind = tc23["in_domain"]
        infeasible_notes.append(
            f"<p>in-domain (gen_forgery): n={ind.get('n')} "
            f"TPR={_fmt(ind.get('tc2_tpr'))} F1={_fmt(ind.get('tc3_f1'))} "
            f"{'INFEASIBLE' if ind.get('infeasible') else 'feasible'} "
            f"(diagnostic only)</p>"
        )
    if tc23 and tc23.get("cross_domain"):
        xd = tc23["cross_domain"]
        infeasible_notes.append(
            f"<p>cross-domain (FMIDV): n={xd.get('n')} "
            f"TPR={_fmt(xd.get('tc2_tpr'))} F1={_fmt(xd.get('tc3_f1'))} "
            f"<strong>{'INFEASIBLE' if xd.get('infeasible') else 'feasible'}</strong> "
            f"(pass gate)</p>"
        )
    if tc23 and tc23.get("infeasible"):
        infeasible_notes.append(
            "<p class='infeasible'>TC2/TC3 gate: <strong>INFEASIBLE</strong> — "
            "cross-domain must jointly satisfy TPR and F1 targets.</p>"
        )
    elif tc23 and tc23.get("feasible_interval"):
        a, b = tc23["feasible_interval"]
        margins = tc23.get("recommended_margins") or {}
        margin_txt = ", ".join(f"{k}={v:.4f}" for k, v in margins.items())
        infeasible_notes.append(
            f"<p>TC2/TC3 gate ({tc23.get('gate_domain', 'cross_domain')}) "
            f"feasible interval: [{a:.2f}, {b:.2f}], "
            f"recommended t*={tc23['recommended_threshold']:.2f} "
            f"(maximin; margins: {margin_txt}; "
            f"min_norm={tc23.get('min_normalized_margin')})</p>"
        )
        if any(float(v) < 0.01 for v in margins.values()):
            infeasible_notes.append(
                "<p class='warn'>TC2/TC3: thin margin (&lt;0.01) — field remeasure may flip.</p>"
            )
    if tc45 and tc45.get("under_variation_warning"):
        infeasible_notes.append(
            "<p class='warn'><strong>변이 과소, 임계값 확정 불가</strong> — "
            "MIDV photo/scan/video cross pairs (or flagged under-variation pairs) "
            "are calibration-only; do not lock face thresholds.</p>"
        )
    if tc45 and tc45.get("infeasible"):
        infeasible_notes.append(
            "<p class='infeasible'>TC4/TC5: <strong>INFEASIBLE</strong> — "
            + (
                "변이 과소, 임계값 확정 불가."
                if tc45.get("under_variation_warning")
                else "no threshold jointly satisfies sensitivity and accuracy targets."
            )
            + "</p>"
        )
    elif tc45 and tc45.get("feasible_interval"):
        a, b = tc45["feasible_interval"]
        margins = tc45.get("recommended_margins") or {}
        margin_txt = ", ".join(f"{k}={v:.4f}" for k, v in margins.items())
        infeasible_notes.append(
            f"<p>TC4/TC5 feasible interval: [{a:.2f}, {b:.2f}], "
            f"recommended t*={tc45['recommended_threshold']:.2f} "
            f"(maximin; margins: {margin_txt}; "
            f"min_norm={tc45.get('min_normalized_margin')})</p>"
        )
        if any(float(v) < 0.01 for v in margins.values()):
            infeasible_notes.append(
                "<p class='warn'>TC4/TC5: thin margin (&lt;0.01) — field remeasure may flip.</p>"
            )

    generated = datetime.now(timezone.utc).isoformat()
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8"/>
<title>redrob-verify Evaluation Report</title>
<style>
  body {{ font-family: "Segoe UI", "Noto Sans KR", sans-serif; margin: 2rem; color: #1a1a1a; background: #fafafa; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.2rem; margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }}
  table {{ border-collapse: collapse; width: 100%; max-width: 900px; background: #fff; }}
  th, td {{ border: 1px solid #ccc; padding: 0.5rem 0.75rem; text-align: left; }}
  th {{ background: #f0f0f0; }}
  .pass {{ color: #1b7f3a; font-weight: 700; }}
  .fail {{ color: #b00020; font-weight: 700; }}
  .na {{ color: #888; }}
  .infeasible {{ background: #ffe0e0; border: 2px solid #b00020; padding: 0.75rem; font-size: 1.05rem; }}
  .warn {{ background: #fff3cd; border: 1px solid #856404; padding: 0.6rem; }}
  figure {{ margin: 1rem 0; }}
  img {{ max-width: 100%; height: auto; border: 1px solid #ddd; background: #fff; }}
  figcaption {{ font-size: 0.85rem; color: #555; margin-top: 0.35rem; }}
  .freeze {{ background: #fff; border: 1px solid #ccc; padding: 1rem; max-width: 900px; font-family: ui-monospace, Consolas, monospace; font-size: 0.85rem; white-space: pre-wrap; }}
  .meta {{ color: #666; font-size: 0.9rem; margin-bottom: 1.5rem; }}
  .dev-fixture {{ background: #b00020; color: white; border: 3px solid #650012; padding: 1rem; font-size: 1.2rem; font-weight: 800; text-align: center; margin-bottom: 1rem; }}
  .non-field {{ background: #b8860b; color: #1a1a1a; border: 3px solid #8b6914; padding: 1rem; font-size: 1.1rem; font-weight: 800; text-align: center; margin-bottom: 1rem; }}
</style>
</head>
<body>
{origin_banner}
<h1>redrob-verify Evaluation Report</h1>
<p class="meta">Generated (UTC): {generated}</p>

<h2>0. Preflight {_pass_badge(preflight_ok)}</h2>
<table>
  <thead><tr>
    <th>Endpoint</th><th>service</th><th>backend</th><th>version</th>
    <th>model_sha256</th><th>git_commit</th><th>started_at</th>
  </tr></thead>
  <tbody>
{preflight_rows}
  </tbody>
</table>

<h2>1. Metrics Summary</h2>
<table>
  <thead><tr><th>ID</th><th>항목</th><th>목표</th><th>실측</th><th>판정</th></tr></thead>
  <tbody>
{table_rows}
  </tbody>
</table>
{"".join(infeasible_notes)}

<h2>1b. Origin distribution</h2>
<table>
  <thead><tr><th>Eval</th><th>origin counts</th><th>tta_valid</th></tr></thead>
  <tbody>
{origin_rows}
  </tbody>
</table>

<h2>2. Threshold Sweeps</h2>
{img("forgery_sweep", "TC2/TC3 gate sweep (pass = cross-domain when present)")}
{img("forgery_sweep_in", "TC2/TC3 in-domain (gen_forgery) — diagnostic")}
{img("forgery_sweep_cross", "TC2/TC3 cross-domain (FMIDV) — pass gate")}
{img("face_sweep", "TC4/TC5 face sweep (Sensitivity, Accuracy)")}

<h2>3. Distributions</h2>
{img("cer_hist", "TC1 cer_field per-item histogram")}
{img("tc6_hist", "TC6 response-time histogram")}

<h2>3b. TC1 backend / script / ablation</h2>
{_tc1_detail_html(tc1, tc1_classic, tc1_vl, ablation)}

<h2>3c. Forgery train</h2>
{_forgery_train_html(forgery_train)}

<h2>4. Freeze Metadata</h2>
<div class="freeze">{json.dumps(freeze, indent=2, ensure_ascii=False)}</div>
</body>
</html>
"""

    out = results / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")

    # Also print six metrics to stdout for eval-all
    print("--- Summary ---")
    for r in rows:
        badge = _pass_badge(r["ok"])
        for tag in (
            '<span class="pass">',
            '<span class="fail">',
            '<span class="na">',
            "</span>",
        ):
            badge = badge.replace(tag, "")
        print(f"{r['id']} {r['name']}: {_fmt(r['value'])}  target {r['target']}  {badge}")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate self-contained HTML report")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    build_report(cfg)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate the thesis result figures from committed evidence artifacts.

Every number plotted here is read from a file under ``results/`` or from a
TensorBoard event file under ``checkpoints/``. Nothing is hard-coded except
axis labels and the predeclared gate thresholds, which are themselves read
back from the frozen report where the report records them.

Output: vector PDFs written into the LaTeX template's ``figs/`` directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "thesis-report" / "figs"

# A single muted palette shared by every figure.
C_MAIN = "#1f4e79"
C_ALT = "#c0504d"
C_THIRD = "#7f7f7f"
C_FOURTH = "#4f8a3d"
C_GATE = "#b45f06"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def load(rel: str) -> dict:
    with (ROOT / rel).open(encoding="utf-8") as handle:
        return json.load(handle)


def save(fig: plt.Figure, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / name
    fig.savefig(out, format="pdf")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def tb_scalars(event_file: Path, tag: str) -> tuple[list[int], list[float]]:
    from tensorboard.backend.event_processing import event_accumulator

    acc = event_accumulator.EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags()["scalars"]:
        return [], []
    steps: list[int] = []
    values: list[float] = []
    for scalar in acc.Scalars(tag):
        if math.isfinite(scalar.value):
            steps.append(scalar.step)
            values.append(scalar.value)
    return steps, values


def tb_file(run_dir: str, kind: str) -> Path | None:
    candidates = sorted((ROOT / run_dir).glob(f"*.{kind}"))
    return candidates[0] if candidates else None


# --------------------------------------------------------------------------
# 1. Corpus funnel
# --------------------------------------------------------------------------
def fig_corpus_funnel() -> None:
    card = load("results/dataset_card_snapshot.json")
    export = load("results/moshi_export_report.json")
    audit = load("results/conversation_audit.json")

    inventory = float(card["csv_hours"])
    candidate = float(card["conversation_candidate_hours"])
    pair_hours = float(audit["hours"])
    export_hours = float(export["exported_hours"])
    expected = {
        "conversation_estimated_pair_hours": pair_hours,
        "moshi_export_hours": export_hours,
    }
    for key, value in expected.items():
        if not math.isclose(float(card[key]), value, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"evidence drift for {key}: {card[key]} != {value}")

    stages = [
        ("Channel caption inventory", inventory),
        ("Deterministic episode selection", candidate),
        ("Aligned response-pair source audio", pair_hours),
        ("Audited stereo training export", export_hours),
    ]

    fig, ax = plt.subplots(figsize=(6.0, 2.5))
    labels = [s[0] for s in stages][::-1]
    values = [s[1] for s in stages][::-1]
    colours = [C_FOURTH, C_MAIN, C_THIRD, C_THIRD][::-1]
    bars = ax.barh(labels, values, color=colours, height=0.6)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_width() + inventory * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,.3f} h",
            va="center",
            fontsize=8.5,
        )
    ax.axvspan(100, 200, color=C_FOURTH, alpha=0.16, zorder=0)
    ax.annotate(
        "required band 100-200 h",
        xy=(170, -0.42),
        xytext=(330, -0.62),
        ha="left",
        va="center",
        fontsize=8.2,
        color=C_FOURTH,
        arrowprops={"arrowstyle": "->", "color": C_FOURTH, "linewidth": 0.7},
    )
    ax.set_xlim(0, inventory * 1.22)
    ax.set_ylim(-0.9, 3.55)
    ax.set_xlabel("Hours of audio")
    ax.set_title("Corpus reduction from raw inventory to audited training export")
    save(fig, "corpus_funnel.pdf")


# --------------------------------------------------------------------------
# 2. Moshi v1 training and re-evaluation curves
# --------------------------------------------------------------------------
def fig_moshi_v1() -> None:
    train_file = tb_file("checkpoints/moshi_fa/tb", "train")
    steps, losses = tb_scalars(train_file, "train.loss") if train_file else ([], [])
    selection = load("results/moshi_checkpoint_selection.json")
    cand = selection["candidates"]
    csteps = [c["step"] for c in cand]
    ctotal = [c["eval_loss"] for c in cand]
    ctext = [c["text_eval_loss"] for c in cand]
    caudio = [c["audio_eval_loss"] for c in cand]

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ax = axes[0]
    if steps:
        ax.plot(
            steps,
            losses,
            color=C_MAIN,
            linewidth=0.5,
            alpha=0.30,
            label="logged every 10 steps",
        )
        window = 20
        if len(losses) > window:
            smooth_steps, smooth = [], []
            running = 0.0
            for i, value in enumerate(losses):
                running += value
                if i >= window:
                    running -= losses[i - window]
                if i >= window - 1:
                    smooth_steps.append(steps[i])
                    smooth.append(running / window)
            ax.plot(
                smooth_steps,
                smooth,
                color=C_ALT,
                linewidth=1.3,
                label=f"{window}-point moving mean",
            )
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss")
    ax.set_title("(a) Teacher-forced training loss")
    ax.xaxis.set_major_locator(MaxNLocator(5))

    ax = axes[1]
    ax.plot(csteps, ctotal, "o-", color=C_MAIN, markersize=3, label="total")
    ax.plot(csteps, caudio, "s--", color=C_ALT, markersize=3, label="audio")
    ax.plot(csteps, ctext, "^:", color=C_FOURTH, markersize=3, label="text")
    selected = selection.get("selected", {}).get("step")
    if selected:
        ax.axvline(selected, color=C_THIRD, linewidth=0.8, linestyle="-.")
        ax.annotate(
            f"selected step {selected}",
            xy=(selected, 0.55),
            xytext=(csteps[3], 0.62),
            fontsize=8,
            color=C_THIRD,
            ha="left",
            arrowprops={"arrowstyle": "->", "color": C_THIRD, "linewidth": 0.7},
        )
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Validation loss")
    ax.set_ylim(0, 1.85)
    ax.set_title("(b) Re-evaluated validation loss")
    ax.legend(frameon=False, loc="center right", fontsize=8.2)
    ax.xaxis.set_major_locator(MaxNLocator(5))

    fig.tight_layout()
    save(fig, "moshi_v1_training.pdf")


# --------------------------------------------------------------------------
# 3. Moshi v2: loss falls while autoregressive runtime keeps failing
# --------------------------------------------------------------------------
def fig_moshi_v2_loss_vs_runtime() -> None:
    selection = load("results/moshi_v2_checkpoint_selection.json")
    cand = selection["candidates"]
    steps = [c["step"] for c in cand]
    loss = [c["eval_loss"] for c in cand]
    passes = [c["runtime_panel_pass_count"] for c in cand]
    panel_n = cand[0]["runtime_panel_count"]

    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.bar(
        steps,
        passes,
        width=56,
        color=C_ALT,
        alpha=0.75,
        label=f"runtime rows passed (of {panel_n})",
        zorder=2,
    )
    ax.axhline(
        panel_n,
        color=C_GATE,
        linewidth=1.0,
        linestyle="--",
        label=f"eligibility gate = {panel_n}/{panel_n}",
    )
    ax.set_ylim(0, panel_n + 0.6)
    ax.set_ylabel(f"Runtime rows passed (of {panel_n})")
    ax.set_xlabel("Checkpoint step")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax.twinx()
    ax2.plot(steps, loss, "o-", color=C_MAIN, markersize=3.2, label="validation loss", zorder=3)
    ax2.set_ylabel("Validation loss")
    lo, hi = min(loss), max(loss)
    pad = (hi - lo) * 0.12
    ax2.set_ylim(lo - pad, hi + pad)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)

    handles = ax.get_legend_handles_labels()
    handles2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles[0] + handles2[0],
        handles[1] + handles2[1],
        frameon=False,
        loc="upper left",
        fontsize=8.2,
    )
    ax.set_title("Trial v2: validation loss falls, runtime never reaches 9/9")
    fig.tight_layout()
    save(fig, "moshi_v2_loss_vs_runtime.pdf")


# --------------------------------------------------------------------------
# 4. Moshi v6.2 in-sample capacity diagnostic
# --------------------------------------------------------------------------
def fig_moshi_v62() -> None:
    reeval = load("results/moshi_v6_text_dropout_reevaluation.json")
    runtime = load("results/moshi_v6_text_dropout_runtime.json")
    cand = reeval["candidates"]
    steps = [c["step"] for c in cand]
    total = [c["eval_loss"] for c in cand]
    text = [c["text_eval_loss"] for c in cand]
    audio = [c["audio_eval_loss"] for c in cand]

    by_step = {c.get("step"): c for c in runtime["candidates"]}
    passes = [int(by_step[s]["panel_pass_count"]) for s in steps]
    panel_n = next(
        (int(c["panel_count"]) for c in runtime["candidates"] if c.get("panel_count")),
        None,
    )
    if panel_n is None:
        raise ValueError("runtime evidence does not define panel_count")

    reduction = (text[0] - min(text[1:])) / text[0] * 100 if len(text) > 1 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ax = axes[0]
    ax.plot(steps, total, "o-", color=C_MAIN, markersize=3.5, label="total")
    ax.plot(steps, audio, "s--", color=C_ALT, markersize=3.5, label="audio")
    ax.plot(steps, text, "^:", color=C_FOURTH, markersize=3.5, label="text")
    ax.annotate(
        f"text loss $-${reduction:.2f}%\nfrom step {steps[0]}",
        xy=(steps[-1], text[-1]),
        xytext=(steps[1], text[0] + 0.15),
        fontsize=8.2,
        color=C_FOURTH,
        arrowprops={"arrowstyle": "->", "color": C_FOURTH, "linewidth": 0.7},
    )
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("In-sample loss")
    ax.set_title("(a) In-sample learning signal")
    ax.set_xticks(steps)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.bar(steps, passes, width=28, color=C_ALT, alpha=0.7, zorder=2)
    ax.scatter(steps, passes, s=26, color=C_ALT, zorder=3)
    ax.axhline(panel_n, color=C_GATE, linewidth=1.0, linestyle="--")
    ax.text(
        steps[-1],
        panel_n - 0.28,
        f"eligibility gate = {panel_n}/{panel_n}",
        fontsize=8.2,
        color=C_GATE,
        ha="right",
        va="top",
    )
    for step, value in zip(steps, passes, strict=True):
        ax.text(step, value + 0.38, f"{value}/{panel_n}", ha="center", fontsize=8.2)
    ax.set_ylim(-0.2, panel_n + 0.8)
    ax.set_xticks(steps)
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel(f"Runtime rows passed (of {panel_n})")
    ax.set_title("(b) Autoregressive runtime outcome")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    save(fig, "moshi_v62_diagnostic.pdf")


# --------------------------------------------------------------------------
# 5. Cross-trial summary
# --------------------------------------------------------------------------
def fig_moshi_trials() -> None:
    trials = load("results/eval/EVIDENCE_STATUS.json")["direct_moshi"]["trials"]
    v1_selection = load("results/moshi_checkpoint_selection.json")
    v1_min_loss = min(c["eval_loss"] for c in v1_selection["candidates"])

    labels: list[str] = []
    losses: list[float] = []
    best: list[int] = []
    panel_size = 9
    for trial in trials:
        label = str(trial.get("version", "?"))
        fixed = trial.get("fixed_validation_losses") or []
        loss = min(fixed) if fixed else (v1_min_loss if label == "v1" else None)
        if loss is None:
            continue
        counts = trial.get("runtime_panel_pass_counts") or []
        panel_size = trial.get("runtime_panel_size") or panel_size
        labels.append(label)
        losses.append(float(loss))
        best.append(max(counts) if counts else 0)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    xs = list(range(len(labels)))

    ax = axes[0]
    ax.bar(xs, losses, color=C_MAIN, width=0.55)
    for x, value in zip(xs, losses, strict=True):
        ax.text(x, value + 0.03, f"{value:.3f}", ha="center", fontsize=8.2)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Best offline loss achieved")
    ax.set_ylim(0, max(losses) * 1.25)
    ax.set_title("(a) Best offline loss per trial")

    ax = axes[1]
    ax.bar(xs, best, color=C_ALT, width=0.55, alpha=0.7, zorder=2)
    ax.scatter(xs, best, s=22, color=C_ALT, zorder=3)
    ax.axhline(panel_size, color=C_GATE, linewidth=1.0, linestyle="--")
    ax.text(
        xs[-1],
        panel_size - 0.28,
        f"eligibility gate = {panel_size}/{panel_size}",
        fontsize=8.2,
        color=C_GATE,
        ha="right",
        va="top",
    )
    for x, value in zip(xs, best, strict=True):
        ax.text(x, value + 0.18, str(value), ha="center", fontsize=8.2)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, panel_size + 0.9)
    ax.set_ylabel(f"Best runtime rows passed (of {panel_size})")
    ax.set_title("(b) Best autoregressive runtime per trial")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    save(fig, "moshi_trials_summary.pdf")


# --------------------------------------------------------------------------
# 6. Automatic semantic final test
# --------------------------------------------------------------------------
def fig_semantic_final_test() -> None:
    report = load("results/eval/qwen4b_responder_v2_final_test_proxy.json")
    agg = report["aggregate"]
    thr = report["thresholds"]

    arms = [
        ("0.5B\nbaseline", "base", C_THIRD),
        ("Qwen3-4B\nprompt-v2", "qwen4b_v2", C_MAIN),
        ("aligned\nreference", "reference", C_ALT),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))

    ax = axes[0]
    width = 0.36
    xs = range(len(arms))
    rel = [agg[key]["relevance"]["mean"] for _, key, _ in arms]
    coh = [agg[key]["coherence"]["mean"] for _, key, _ in arms]
    b1 = ax.bar([x - width / 2 for x in xs], rel, width, color=C_MAIN, label="relevance")
    b2 = ax.bar([x + width / 2 for x in xs], coh, width, color=C_FOURTH, label="coherence")
    for series_index, bars in enumerate((b1, b2)):
        label_offset = 0.06 + series_index * 0.10
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + label_offset,
                f"{bar.get_height():.3f}",
                ha="center",
                fontsize=7.5,
            )
    ax.axhline(
        thr["candidate_relevance_mean_min"],
        color=C_GATE,
        linestyle="--",
        linewidth=0.9,
    )
    ax.text(
        2.85,
        thr["candidate_relevance_mean_min"] + 0.04,
        rf"relevance $\geq$ {thr['candidate_relevance_mean_min']:.1f}",
        fontsize=7.8,
        color=C_GATE,
        ha="right",
        va="bottom",
    )
    ax.axhline(
        thr["candidate_coherence_mean_min"],
        color=C_GATE,
        linestyle=":",
        linewidth=0.9,
    )
    ax.text(
        2.85,
        thr["candidate_coherence_mean_min"] + 0.04,
        rf"coherence $\geq$ {thr['candidate_coherence_mean_min']:.1f}",
        fontsize=7.8,
        color=C_GATE,
        ha="right",
        va="bottom",
    )
    ax.set_xlim(-0.6, 2.95)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([label for label, _, _ in arms], fontsize=7.5)
    ax.set_ylim(0, 4.5)
    ax.set_ylabel("Mean judge score (0-4)")
    ax.set_title(f"(a) Blinded arm means, n={agg['rows']} rows")
    ax.legend(frameon=False, loc="upper left", fontsize=8.2)

    ax = axes[1]
    bottoms = [0.0] * len(arms)
    shades = ["#e8e8e8", "#c6d4e2", "#9db7cf", "#5b87ad", C_MAIN]
    total_calls = None
    for score in range(5):
        heights = []
        for _, key, _ in arms:
            hist = agg[key]["relevance"]["histogram"]
            total = sum(hist.values())
            total_calls = total
            heights.append(hist.get(str(score), 0) / total * 100)
        ax.bar(
            list(xs),
            heights,
            bottom=bottoms,
            color=shades[score],
            edgecolor="white",
            linewidth=0.4,
            label=f"score {score}",
        )
        bottoms = [b + h for b, h in zip(bottoms, heights, strict=True)]
    ax.set_ylim(0, 118)
    ax.set_ylabel("Share of judge calls (%)")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([label for label, _, _ in arms], fontsize=7.5)
    ax.set_title(f"(b) Relevance score distribution\n({total_calls} calls per arm)")
    ax.legend(frameon=False, fontsize=7.5, ncol=5, loc="upper center", columnspacing=0.8)

    fig.tight_layout()
    save(fig, "semantic_final_test.pdf")


# --------------------------------------------------------------------------
# 7. Barge-in detector
# --------------------------------------------------------------------------
def fig_bargein() -> None:
    proxy = load("results/eval/interrupt_recorded_proxy.json")
    sweep = proxy["threshold_selection"]["candidates"]
    selected = proxy["threshold_selection"]["selected"]
    test = proxy["heldout_test"]

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))

    ax = axes[0]
    ths = [c["threshold"] for c in sweep]
    ax.plot(
        ths,
        [c["accuracy"] * 100 for c in sweep],
        "o-",
        color=C_MAIN,
        markersize=3,
        label="accuracy",
    )
    ax.plot(
        ths,
        [c["interrupt_f1"] * 100 for c in sweep],
        "s--",
        color=C_FOURTH,
        markersize=3,
        label="interrupt F1",
    )
    ax.plot(
        ths,
        [c["far"] * 100 for c in sweep],
        "^:",
        color=C_ALT,
        markersize=3,
        label="false-accept rate",
    )
    ax.axvline(selected, color=C_THIRD, linewidth=0.9, linestyle="-.")
    ax.text(
        selected,
        4,
        f" selected\n threshold {selected}",
        fontsize=8,
        color=C_THIRD,
    )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Percent")
    ax.set_title("(a) Validation threshold sweep")
    ax.legend(frameon=False, fontsize=8.2)
    ax.set_ylim(0, 100)

    ax = axes[1]
    metrics = [
        ("Accuracy", "accuracy"),
        ("Interrupt\nF1", "interrupt_f1"),
        ("FAR", "far"),
        ("FRR", "frr"),
    ]
    width = 0.36
    xs = range(len(metrics))
    prop = [test["proposed"][key] * 100 for _, key in metrics]
    base = [test["energy_zcr_baseline"][key] * 100 for _, key in metrics]
    ci = test.get("proposed_session_block_bootstrap_ci95", {})
    err_lo, err_hi = [], []
    for _, key in metrics:
        bounds = ci.get(key)
        value = test["proposed"][key] * 100
        if bounds:
            err_lo.append(max(0.0, value - bounds[0] * 100))
            err_hi.append(max(0.0, bounds[1] * 100 - value))
        else:
            err_lo.append(0.0)
            err_hi.append(0.0)

    ax.bar(
        [x - width / 2 for x in xs],
        prop,
        width,
        color=C_MAIN,
        yerr=[err_lo, err_hi],
        capsize=2.5,
        error_kw={"linewidth": 0.7},
        label="GBDT detector",
    )
    ax.bar(
        [x + width / 2 for x in xs],
        base,
        width,
        color=C_THIRD,
        label="energy/ZCR baseline",
    )
    for x, value in zip(xs, base, strict=True):
        if value < 1.0:
            ax.text(x + width / 2, 2.2, "0", ha="center", fontsize=7.8, color=C_THIRD)
    ax.axhline(80, color=C_GATE, linestyle="--", linewidth=0.9)
    ax.text(
        len(metrics) - 0.02,
        83.5,
        "80% target",
        fontsize=8,
        color=C_GATE,
        ha="right",
        va="bottom",
    )
    ax.set_xticks(list(xs))
    ax.set_xticklabels([label for label, _ in metrics], fontsize=7.8)
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 118)
    ax.set_title(f"(b) Held-out test: {test['n']} events, {test['sessions']} sessions")
    ax.legend(frameon=False, fontsize=8.2, loc="upper left", ncol=1)

    fig.tight_layout()
    save(fig, "bargein_proxy.pdf")


# --------------------------------------------------------------------------
# 8. Cascade mechanics: turn time and intelligibility
# --------------------------------------------------------------------------
def fig_cascade_panel() -> None:
    panel = load("results/eval/cascade_real_service_validation_panel.json")
    desc = load("results/eval/cascade_validation_descriptive_analysis.json")
    intel = load("results/eval/cascade_intelligibility_proxy.json")

    chars = [s["transcript_statistics"]["characters"] for s in panel["samples"]]
    times = [s["full_turn_generation_ms"] for s in panel["samples"]]
    rows = [s["manifest_row_index"] for s in panel["samples"]]
    pearson = desc.get("transcript_length_full_turn_pearson_r")

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))

    ax = axes[0]
    ax.scatter(chars, times, s=26, color=C_MAIN, zorder=3)
    # Place the clustered near-origin labels on alternating sides and heights.
    offsets = {
        0: (8, 4, "left", "bottom"),
        16: (9, -9, "left", "top"),
        32: (7, 3, "left", "bottom"),
        48: (7, 7, "left", "bottom"),
        65: (-8, 8, "right", "bottom"),
        81: (8, -8, "left", "top"),
        97: (8, 5, "left", "bottom"),
        113: (8, -7, "left", "top"),
        130: (0, 14, "center", "bottom"),
    }
    for x, y, row in zip(chars, times, rows, strict=True):
        dx, dy, ha, va = offsets.get(int(row), (7, 5, "left", "bottom"))
        ax.annotate(
            f"{row}",
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=7.8,
            ha=ha,
            va=va,
        )
    if len(chars) > 1:
        n = len(chars)
        mx = sum(chars) / n
        my = sum(times) / n
        denom = sum((x - mx) ** 2 for x in chars)
        if denom:
            slope = sum((x - mx) * (y - my) for x, y in zip(chars, times, strict=True)) / denom
            intercept = my - slope * mx
            lo, hi = min(chars), max(chars)
            ax.plot(
                [lo, hi],
                [intercept + slope * lo, intercept + slope * hi],
                color=C_ALT,
                linewidth=0.9,
                linestyle="--",
            )
    if pearson is not None:
        ax.text(
            0.96,
            0.05,
            f"descriptive Pearson $r$ = {pearson:.3f}\n(n = {len(chars)}, not inferential)",
            transform=ax.transAxes,
            fontsize=8,
            ha="right",
            va="bottom",
        )
    ax.set_xlabel("ASR transcript length (characters)")
    ax.set_ylabel("Complete-turn generation (ms)")
    ax.set_title("(a) Full-turn time vs. input length")

    ax = axes[1]
    samples = intel["samples"]
    xs = range(len(samples))
    width = 0.38
    wer = [s["wer"] * 100 for s in samples]
    cer = [s["cer"] * 100 for s in samples]
    ax.bar([x - width / 2 for x in xs], wer, width, color=C_ALT, label="WER")
    ax.bar([x + width / 2 for x in xs], cer, width, color=C_MAIN, label="CER")
    micro_wer = intel["aggregate"]["word_error_rate"]["micro"]
    micro_cer = intel["aggregate"]["character_error_rate"]["micro"]
    ax.axhline(
        micro_wer * 100,
        color=C_ALT,
        linestyle="--",
        linewidth=0.9,
        label=f"micro WER {micro_wer * 100:.2f}%",
    )
    ax.axhline(
        micro_cer * 100,
        color=C_MAIN,
        linestyle=":",
        linewidth=0.9,
        label=f"micro CER {micro_cer * 100:.2f}%",
    )
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(s["manifest_row_index"]) for s in samples], fontsize=8)
    ax.set_xlabel("Validation manifest row")
    ax.set_ylabel("Error rate (%)")
    ax.set_ylim(0, 76)
    ax.set_title("(b) ASR round-trip intelligibility proxy")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", columnspacing=1.0)

    fig.tight_layout()
    save(fig, "cascade_panel.pdf")


FIGURES = {
    "corpus": fig_corpus_funnel,
    "v1": fig_moshi_v1,
    "v2": fig_moshi_v2_loss_vs_runtime,
    "v62": fig_moshi_v62,
    "trials": fig_moshi_trials,
    "semantic": fig_semantic_final_test,
    "bargein": fig_bargein,
    "cascade": fig_cascade_panel,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=sorted(FIGURES), default=None)
    args = parser.parse_args()

    style()
    names = args.only or sorted(FIGURES)
    for name in names:
        FIGURES[name]()


if __name__ == "__main__":
    main()

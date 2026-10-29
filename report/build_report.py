"""Build assignment figures and report.pdf from the original run logs.

Dependencies: matplotlib, numpy, reportlab. Run from any directory.
This script only writes report artifacts; it does not modify training inputs.
"""

from pathlib import Path
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/a3-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Image, PageBreak, Table, TableStyle,
)
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
PDF = ROOT / "report.pdf"
FIG.mkdir(parents=True, exist_ok=True)
PDF.parent.mkdir(parents=True, exist_ok=True)

RUNS = {
    "p2a": ("p2a_number_search_reinforce", "p2a REINFORCE (G=1)", "#2563a6"),
    "p2b": ("p2b_number_search_grpo", "p2b GRPO (G=8)", "#d5662d"),
    "p3": ("p3_number_search_async", "p3 async GRPO (G=8)", "#27836c"),
    "p41": ("p41_number_search_grpo_g2", "p41 GRPO (G=2)", "#8356ad"),
    "iw": ("p43_number_search_async_stale4_iw", "p43 iw (PPO)", "#d5662d"),
    "noiw": ("p43_number_search_async_stale4_noiw", "p43 noiw (CE)", "#27836c"),
    "p5": ("p5_textcraft_grpo", "p5 TextCraft GRPO", "#27836c"),
}
data = {}
summary = {}
for key, (name, label, color) in RUNS.items():
    rows = [json.loads(line) for line in (ROOT / "runs" / f"{name}.jsonl").read_text().splitlines()]
    train = [r for r in rows if "train/success_rate" in r]
    evaluation = [r for r in rows if "eval/success_rate" in r]
    assert [r["step"] for r in train] == list(range(1, 51))
    expected_eval_steps = [0, 10, 30, 40, 50] if key == "p5" else [0, 10, 20, 30, 40, 50]
    assert [r["step"] for r in evaluation] == expected_eval_steps
    data[key] = {"train": train, "eval": evaluation}
    skipped = [r["step"] for r in train if r.get("train/skipped")]
    manifest = json.loads((ROOT / "runs" / f"{name}.checkpoints.json").read_text())
    assert manifest["completed_steps"] == 50 - len(skipped)
    summary[key] = {
        "run_name": name,
        "baseline_eval_success": evaluation[0]["eval/success_rate"],
        "final_eval_success": evaluation[-1]["eval/success_rate"],
        "final_eval_pass_at_4": evaluation[-1]["eval/pass@4"],
        "optimizer_updates": manifest["completed_steps"],
        "skipped_steps": skipped,
        "wall_minutes_after_baseline": (evaluation[-1]["time"] - evaluation[0]["time"]) / 60,
        "final_sampler_path": manifest["final_sampler_path"],
        "mean_step_seconds": sum(r["train/step_time_s"] for r in train) / len(train),
        "first20_degenerate_fraction": sum(r["train/frac_degenerate_groups"] for r in train[:20]) / 20,
        "first20_sequences_per_step": sum(r["train/num_sequences"] for r in train[:20]) / 20,
        "groups_dropped_stale": next((r["summary/groups_dropped_stale"] for r in reversed(rows) if "summary/groups_dropped_stale" in r), None),
    }
    for metric in ["grad_norm", "importance_weight_mean", "importance_weight_min", "importance_weight_max", "importance_weight_std", "entropy_proxy"]:
        values = [r[f"train/{metric}"] for r in train if f"train/{metric}" in r]
        summary[key][metric] = {"min": min(values), "max": max(values), "final": values[-1]}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8291a2", "grid.color": "#e2e8f0",
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

def plot_metric(ax, metric, keys=("p2a", "p2b"), probability=False):
    phase = metric.split("/")[0]
    for key in keys:
        records = data[key][phase]
        if key == "p5" and phase == "eval":
            # Preserve the missing step-20 evaluation as a gap, not an interpolated result.
            records = sorted([*records, {"step": 20}], key=lambda r: r["step"])
        _, label, color = RUNS[key]
        # NaN gaps explicitly preserve missing optimizer metrics on skipped steps.
        ax.plot([r["step"] for r in records], [r.get(metric, np.nan) for r in records],
                color=color, label=label, linewidth=1.5,
                marker="o", markersize=3 if phase == "eval" else 2)
    ax.set_title(metric, loc="left", pad=9)
    ax.set_xlabel("Training step")
    ax.set_xlim(0, 50)
    ax.set_xticks(range(0, 51, 10))
    ax.grid(True, alpha=0.8)
    if probability:
        ax.set_ylim(-0.025, 1.05)
    ax.legend(loc="best", frameon=True, framealpha=0.92)

def save_figure(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=220)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.2), layout="constrained")
plot_metric(axes[0, 0], "train/success_rate", probability=True)
plot_metric(axes[0, 1], "eval/success_rate", probability=True)
ax = axes[1, 0]
for key, metric in [("p2a", "train/pass@1"), ("p2b", "train/pass@8")]:
    records = data[key]["train"]
    ax.plot([r["step"] for r in records], [r[metric] for r in records],
            label=f"{key} {metric}", color=RUNS[key][2], linewidth=1.5, marker="o", markersize=2)
ax.set_title("Training pass@k (different k)", loc="left", pad=9)
ax.set_xlabel("Training step")
ax.set_xlim(0, 50)
ax.set_xticks(range(0, 51, 10))
ax.set_ylim(-0.025, 1.05)
ax.grid(True)
ax.legend(loc="lower right")
plot_metric(axes[1, 1], "eval/pass@4", probability=True)
save_figure(fig, "part2_reward")

fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.2), layout="constrained")
for ax, metric in zip(axes.flat, ["train/grad_norm", "train/importance_weight_mean", "train/importance_weight_max", "train/importance_weight_std"]):
    plot_metric(ax, metric)
axes[0, 1].axhline(1, color="#64748b", linestyle=":", linewidth=1)
axes[0, 1].yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
save_figure(fig, "part2_optimization")

fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.2), layout="constrained")
plot_metric(axes[0], "train/frac_degenerate_groups", keys=("p2b",), probability=True)
axes[0].plot([r["step"] for r in data["p2a"]["train"]],
             [r["train/frac_degenerate_groups"] for r in data["p2a"]["train"]],
             color=RUNS["p2a"][2], linestyle="--", label="p2a logged 0 (N/A)")
axes[0].legend(loc="center left", fontsize=8)
plot_metric(axes[1], "train/entropy_proxy")
save_figure(fig, "part2_data_health")

fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.2), layout="constrained")
for ax, metric in zip(axes.flat, ["train/success_rate", "eval/success_rate", "train/pass@8", "eval/pass@4"]):
    plot_metric(ax, metric, keys=("p2b", "p3"), probability=True)
save_figure(fig, "part3_reward")

fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.2), layout="constrained")
for ax, metric in zip(axes.flat, ["train/grad_norm", "train/importance_weight_mean", "train/importance_weight_max", "train/importance_weight_std"]):
    plot_metric(ax, metric, keys=("p2b", "p3"))
axes[0, 1].axhline(1, color="#64748b", linestyle=":", linewidth=1)
axes[0, 1].yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
save_figure(fig, "part3_optimization")

fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.2), layout="constrained")
plot_metric(axes[0], "train/frac_degenerate_groups", keys=("p2b", "p3"), probability=True)
plot_metric(axes[1], "train/entropy_proxy", keys=("p2b", "p3"))
save_figure(fig, "part3_data_health")

def ablation_figures(prefix, keys):
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.2), layout="constrained")
    plot_metric(axes[0, 0], "train/success_rate", keys=keys, probability=True)
    plot_metric(axes[0, 1], "eval/success_rate", keys=keys, probability=True)
    ax = axes[1, 0]
    for key in keys:
        records = data[key]["train"]
        pass_keys = sorted({k for r in records for k in r if k.startswith("train/pass@")})
        for metric in pass_keys:
            ax.plot([r["step"] for r in records], [r.get(metric, np.nan) for r in records],
                    color=RUNS[key][2], label=f"{key} {metric}",
                    linewidth=1.5, marker="o", markersize=2)
    ax.set_title("Training pass@k", loc="left", pad=9)
    ax.set_xlabel("Training step")
    ax.set_xlim(0, 50)
    ax.set_xticks(range(0, 51, 10))
    ax.set_ylim(-0.025, 1.05)
    ax.grid(True)
    ax.legend(loc="best")
    plot_metric(axes[1, 1], "eval/pass@4", keys=keys, probability=True)
    save_figure(fig, f"{prefix}_reward")

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.2), layout="constrained")
    for ax, metric in zip(axes.flat, ["train/grad_norm", "train/importance_weight_mean", "train/importance_weight_max", "train/importance_weight_std"]):
        plot_metric(ax, metric, keys=keys)
    axes[0, 1].axhline(1, color="#64748b", linestyle=":", linewidth=1)
    axes[0, 1].yaxis.set_major_formatter(FormatStrFormatter("%.5f" if keys == ("p5",) else "%.4f"))
    save_figure(fig, f"{prefix}_optimization")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.2), layout="constrained")
    plot_metric(axes[0], "train/frac_degenerate_groups",
                keys=tuple(key for key in keys if key != "p2a"), probability=True)
    if "p2a" in keys:
        axes[0].plot([], [], linestyle="none", label="p2a REINFORCE N/A")
        axes[0].legend(loc="best")
    plot_metric(axes[1], "train/entropy_proxy", keys=keys)
    save_figure(fig, f"{prefix}_data_health")

ablation_figures("part41", ("p2a", "p41", "p2b"))
ablation_figures("part42", ("p2b", "p3"))
ablation_figures("part43", ("iw", "noiw"))
ablation_figures("part5", ("p5",))

comparison = (
    "GRPO reaches 99% evaluation success by step 20 and ends at 99%, while "
    "REINFORCE ends at 56.5% after 50 steps. "
    "The final eval/pass@4 is 100% for GRPO and 94% for REINFORCE. "
    "This is expected because GRPO compares attempts on the same task and can penalize "
    "unsuccessful attempts, while REINFORCE only reinforces successful ones. "
    "However, GRPO also uses eight times as many rollouts per step and a different "
    "loss, so this is not an equal-budget comparison."
)
reward_note = (
    "Figure 1. GRPO's training and evaluation success rise faster, and train/pass@8 "
    "saturates early. REINFORCE's train/pass@1 equals its training success. The two "
    "training pass@k curves use different k."
)
optimization_note = (
    "Figure 2. Gradient norms show no large spikes. The mean importance weight stays "
    "near 1. The maximum is below 2.4 and the standard deviation is below 0.035. "
    "Gaps mark steps without updates."
)
health_note = (
    "Figure 3. GRPO's entropy proxy falls and its degenerate fraction rises as more "
    "groups contain only successful attempts. The trainer skips 14 updates when "
    "every group in a batch has zero advantage. REINFORCE's entropy fluctuates, and "
    "its degenerate fraction is N/A because it has no group baseline."
)

speedup = summary["p2b"]["mean_step_seconds"] / summary["p3"]["mean_step_seconds"]
async_comparison = (
    "p3's final eval/success_rate and eval/pass@4 are both 100%. "
    f"The mean train/step_time_s is {summary['p3']['mean_step_seconds']:.2f} s for p3 and "
    f"{summary['p2b']['mean_step_seconds']:.2f} s for p2b, so p3 is {speedup:.2f} times as fast. "
    "p3 samples data and updates the model at the same time. It learns more slowly "
    "per iteration. At step 20, its evaluation success is 60%, while p2b's is 99%."
)
async_reward_note = (
    "Figure 4. Training success and train/pass@8 rise as learning progresses. "
    "eval/pass@4 reaches 100% before eval/success_rate."
)
async_optimization_note = (
    "Figure 5. Gradient norms remain small. p3's importance-weight means range from "
    "0.9988 to 1.0008. Individual ratios span 0.340 to 2.479, while p2b's range is "
    "0.462 to 2.397. Standard deviations are 0.017-0.033 for p3 and 0.008-0.027 "
    "for p2b. The mean alone does not show these differences between individual tokens. "
    "Gaps mark steps without updates."
)
async_health_note = (
    "Figure 6. Entropy proxy decreases and the degenerate-group fraction rises "
    "as success improves. These changes occur later than in p2b. The trainer skips "
    "three updates in p3 because all attempts in those batches succeed. "
    "p3 therefore has 47 optimizer updates across 50 iterations."
)

rollout_specs = [
    (5, "number_search.train.48", 3,
     "The environment says 321 is too low, but the agent guesses 267 and 291. "
     "It ignores the lower bound, then scans 326, 334, 342 and runs out of guesses."),
    (25, "number_search.train.241", 23,
     "It keeps guesses within the bounds and mostly bisects "
     "(645, 825, 735, 690, ...). It eventually finds 697."),
    (50, "number_search.train.459", 45,
     "It guesses 737, 652, 694, 715, 726, 720, 717 and finds the target in seven attempts. "
     "It now makes roughly midpoint jumps instead of the early rollout's slow scan."),
]
rollout_text = []
for step, task_id, version, observation in rollout_specs:
    path = ROOT / "runs" / "p3_number_search_async_trajectories" / f"step_{step:06d}" / f"train_{task_id}_rollout0.jsonl"
    header = json.loads(path.read_text().splitlines()[0])
    assert (header["step"], header["task_id"], header["policy_version"]) == (step, task_id, version)
    rollout_text.append(
        f"Step {step} (task {task_id}, policy_version {version}, rollout 0). {observation}"
    )

ablation_sections = [
    {
        "title": "Part 4.1. Group Size",
        "prefix": "part41",
        "analysis": (
            "Final evaluation success is 56.5% for REINFORCE, 89% for GRPO with G=2, "
            "and 99% for GRPO with G=8. The smaller group estimates the baseline from "
            "fewer samples, so the baseline has higher variance and adds more noise to "
            "the advantages. In the first 20 steps, 61.9% of G=2 groups are degenerate, "
            "versus 20.0% for G=8. G=2 supplies only 6.1 useful sequences per step "
            "on average over this period, versus 51.2 for G=8, which helps explain its "
            "slower learning. REINFORCE's degenerate fraction is N/A. It also uses a "
            "different advantage rule and loss, so this is not a pure group-size comparison."
        ),
        "reward": (
            "Figure 7. G=8 improves fastest, while G=2 improves mainly near the end. "
            "Both GRPO runs reach 100% eval/pass@4, versus 94% for REINFORCE. "
            "Training pass@1, pass@2, and pass@8 use different k and are not directly comparable."
        ),
        "optimization": (
            "Figure 8. G=2 has larger gradient norms than G=8. All three runs have "
            "mean importance weights near 1, maxima below 2.4, and standard deviations "
            "below 0.035. Gaps mark steps without updates."
        ),
        "health": (
            "Figure 9. G=2 has many degenerate groups early. G=8 reaches a high "
            "degenerate fraction later because most attempts succeed. Its entropy "
            "proxy also falls earlier, while G=2 and REINFORCE fluctuate more."
        ),
    },
    {
        "title": "Part 4.2. Synchronous vs. Asynchronous Training",
        "prefix": "part42",
        "analysis": (
            f"The asynchronous run is {speedup:.2f} times as fast per step. Its workers "
            "sample data while the learner updates the model, so the two stages spend "
            "less time waiting for each other. The learner accepts 400 groups and "
            "discards 62 groups that exceed the staleness limit. Its importance weights "
            "have wider extremes and larger standard deviations than the synchronous "
            "run, although their means stay near 1. It learns more slowly per iteration "
            "but reaches similar final accuracy, so these results show no clear final "
            "accuracy cost from stale data. The runs do not isolate staleness as the "
            "cause of slower learning."
        ),
        "table": [
            ["Run", "Mean step time (s)", "Final eval/success_rate"],
            ["p2b synchronous", f"{summary['p2b']['mean_step_seconds']:.2f}", "99%"],
            ["p3 asynchronous", f"{summary['p3']['mean_step_seconds']:.2f}", "100%"],
        ],
        "reward": (
            "Figure 10. p3's training and evaluation success rise later than p2b's. "
            "Both training pass@8 and evaluation pass@4 approach 100%."
        ),
        "optimization": (
            "Figure 11. Neither run shows large gradient spikes. Importance-weight "
            "means stay near 1, but p3's maximum ratio reaches 2.48 versus 2.40 for "
            "p2b, and its standard deviation reaches 0.033 versus 0.027."
        ),
        "health": (
            "Figure 12. p3's degenerate fraction rises and its entropy proxy falls "
            "later than p2b's, consistent with its slower improvement per iteration."
        ),
    },
    {
        "title": "Part 4.3. Importance Weighting under Higher Staleness",
        "prefix": "part43",
        "analysis": (
            "Both runs allow staleness up to four versions. The iw run uses PPO, while "
            "noiw uses advantage-weighted cross-entropy. At step 20, evaluation success "
            "is 92% for noiw and 59.5% for iw. Their final success rates are 99.5% and "
            "99%, respectively, and both reach 100% eval/pass@4. This experiment does "
            "not show an accuracy benefit from importance weighting, although noiw "
            "has larger importance-ratio fluctuations. One run per loss is not enough "
            "to conclude that noiw is generally better."
        ),
        "reward": (
            "Figure 13. noiw's training and evaluation success improve sooner. "
            "Training pass@8 and evaluation pass@4 reach 100% in both runs."
        ),
        "optimization": (
            "Figure 14. Gradient norms are similar and importance-weight means stay "
            "near 1. noiw's maximum ratio reaches 4.35 versus 3.96 for iw, and its "
            "standard deviation reaches 0.104 versus 0.067. These ratios are diagnostics "
            "only in noiw and do not enter its loss."
        ),
        "health": (
            "Figure 15. noiw's entropy proxy falls and its degenerate fraction rises "
            "earlier. Both runs skip updates near the end when all attempts in a batch "
            "succeed. Gaps in the other diagnostic curves correspond to these steps."
        ),
    },
]

p5_table = [["Model", "Split", "success_rate", "pass@4"]]
for model, split, filename in [
    ("Fine-tuned", "Medium", "finetuned_medium"),
    ("Base", "Medium", "base_medium"),
    ("Fine-tuned", "Hard", "finetuned_hard"),
    ("Base", "Hard", "base_hard"),
]:
    text = (ROOT / "runs" / f"p5_eval_{filename}.log").read_text()
    result = json.loads(text[text.rindex("\n{") + 1:])
    assert result["num_tasks"] == (213 if split == "Medium" else 136)
    assert result["episodes_per_task"] == 4
    assert result["model"] == ("a3" if model == "Fine-tuned" else "Qwen/Qwen3.5-4B")
    p5_table.append([model, split, f"{result['success_rate']:.2%}", f"{result['pass@4']:.2%}"])

ablation_sections.append({
    "title": "Part 5. TextCraft-Synth",
    "prefix": "part5",
    "analysis": (
        "On the full validation splits with four attempts per task, fine-tuning improves "
        "medium success from 10.92% to 65.26% and pass@4 from 28.17% to 77.46%. "
        "Both models score 0% on hard, so training on medium does not produce success "
        "on harder tasks in this experiment. The fine-tuned model averages 71.41 of "
        "75 turns on hard, which suggests difficulty with longer plans. "
        "Each medium evaluation has one abandoned episode. Hard has nine for the "
        "fine-tuned model and none for the base model. All count as failures."
    ),
    "table": p5_table,
    "reward": (
        "Figure 16. Training and evaluation success improve. These periodic evaluations "
        "use 50 medium tasks. The run resumed from step 20 after a server failure and "
        "completed 50 updates. The missing step-20 evaluation appears as a gap."
    ),
    "optimization": (
        "Figure 17. Gradient norms remain small. Mean importance weights stay near 1, "
        "maxima stay below 3.11, and standard deviations stay below 0.029."
    ),
    "health": (
        "Figure 18. The degenerate-group fraction fluctuates, while the entropy proxy "
        "generally decreases. No batch is fully degenerate, so no update is skipped."
    ),
})

md = ["# Part 2. Synchronous REINFORCE and GRPO", comparison,
      "![Reward quality](report/figures/part2_reward.png)", reward_note,
      "![Optimization](report/figures/part2_optimization.png)", optimization_note,
      "![Data health](report/figures/part2_data_health.png)", health_note,
      "# Part 3. Asynchronous GRPO", async_comparison,
      "![Reward quality](report/figures/part3_reward.png)", async_reward_note,
      *["- " + text for text in rollout_text],
      "![Optimization](report/figures/part3_optimization.png)", async_optimization_note,
      "![Data health](report/figures/part3_data_health.png)", async_health_note]
for section in ablation_sections:
    md.extend(["# " + section["title"], section["analysis"]])
    if "table" in section:
        rows = section["table"]
        md.append("\n".join([" | ".join(rows[0]), " | ".join(["---"] * len(rows[0])),
                             *[" | ".join(row) for row in rows[1:]]]))
    for figure, caption in [("reward", "reward"), ("optimization", "optimization"), ("data_health", "health")]:
        md.extend([f"![{figure}](report/figures/{section['prefix']}_{figure}.png)", section[caption]])
(ROOT / "report.md").write_text("\n\n".join(md) + "\n")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("BodyA3", fontName="Helvetica", fontSize=10, leading=14,
                         spaceAfter=12))
styles.add(ParagraphStyle("CaptionA3", parent=styles["BodyA3"], fontSize=9, leading=12,
                         spaceAfter=14))
styles.add(ParagraphStyle("TitleA3", fontName="Helvetica-Bold", fontSize=16, leading=20,
                         spaceAfter=14))

def para(text, style="BodyA3"):
    return Paragraph(text, styles[style])

def chart(name):
    path = FIG / f"{name}.png"
    width, height = ImageReader(str(path)).getSize()
    return Image(str(path), width=516, height=516 * height / width)

def page_frame(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(564, 28, str(doc.page))
    canvas.restoreState()

doc = SimpleDocTemplate(str(PDF), pagesize=(612, 792), rightMargin=48, leftMargin=48,
                        topMargin=48, bottomMargin=45, title="Assignment 3 Report",
                        author="")
story = [para("Part 2. Synchronous REINFORCE and GRPO", "TitleA3"),
         para(comparison), chart("part2_reward"), para(reward_note, "CaptionA3"),
         PageBreak(), chart("part2_optimization"), para(optimization_note, "CaptionA3"),
         chart("part2_data_health"), para(health_note, "CaptionA3"),
         PageBreak(), para("Part 3. Asynchronous GRPO", "TitleA3"),
         para(async_comparison), chart("part3_reward"), para(async_reward_note, "CaptionA3"),
         *[para(text) for text in rollout_text],
         PageBreak(), chart("part3_optimization"), para(async_optimization_note, "CaptionA3"),
         chart("part3_data_health"), para(async_health_note, "CaptionA3")]
for section in ablation_sections:
    story.extend([PageBreak(), para(section["title"], "TitleA3"), para(section["analysis"])])
    if "table" in section:
        widths = [150, 160, 206] if len(section["table"][0]) == 3 else [130, 110, 138, 138]
        table = Table(section["table"], colWidths=widths, hAlign="LEFT", spaceAfter=12)
        table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.black),
        ]))
        story.append(table)
    story.extend([
        chart(f"{section['prefix']}_reward"), para(section["reward"], "CaptionA3"),
        PageBreak(), chart(f"{section['prefix']}_optimization"),
        para(section["optimization"], "CaptionA3"),
        chart(f"{section['prefix']}_data_health"), para(section["health"], "CaptionA3"),
    ])
doc.build(story, onFirstPage=page_frame, onLaterPages=page_frame)
print(PDF)
print(ROOT / "report.md")

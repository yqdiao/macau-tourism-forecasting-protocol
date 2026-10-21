#!/usr/bin/env python3
"""Render the joint resource/accuracy figure from the released numeric plot data.

Optional dependency: matplotlib. Numeric verification uses rebuild_tables.py.
The renderer writes to paper/rebuilt and does not replace the archived PDF export.
"""
from pathlib import Path
import csv
import os

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / "rebuilt"
OUT.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "matplotlib-cache"))
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError as exc:
    raise SystemExit("Install matplotlib in your plotting environment to use this optional renderer.") from exc

with (PAPER / "data/accuracy_resource_plot_data.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                     "axes.titlesize": 10, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.facecolor": "white"})
fig, axes = plt.subplots(2, 2, figsize=(7.7, 6.8))
fig.subplots_adjust(left=.09, right=.98, bottom=.18, top=.91, wspace=.43, hspace=.55)
blue, orange, grey = "#0072B2", "#D55E00", "#6B6B6B"

for ax, panel, metric, title, ylabel, ylim, ticks in [
    (axes[0, 0], "A", "mase", "MASE and relative computation charge", "MASE(7)", (1.025, 1.75), [1.1, 1.3, 1.5, 1.7]),
    (axes[0, 1], "B", "mae", "MAE and relative computation charge", "MAE (arrivals/day)", (490, 980), [500, 650, 800, 950]),
    (axes[1, 0], "C", "rmse", "RMSE and relative computation charge", "Mean task RMSE (arrivals/day)", (740, 1250), [800, 1000, 1200]),
]:
    selected = sorted((r for r in rows if r["panel"] == panel), key=lambda r: float(r["relative_registered_charge_chronos2_1"]))
    frontier = [r for r in selected if r["conditional_nondominated"] == "True"]
    ax.plot([float(r["relative_registered_charge_chronos2_1"]) for r in frontier],
            [float(r["error"]) for r in frontier], color=blue, lw=1.3)
    for r in selected:
        x, y = float(r["relative_registered_charge_chronos2_1"]), float(r["error"])
        nondominated = r["conditional_nondominated"] == "True"
        ax.scatter([x], [y], marker="o" if nondominated else "s", s=28,
                   facecolors=blue if nondominated else "white", edgecolors=blue if nondominated else grey, zorder=3)
        offset = {"Chronos-2": (4, 9), "TimesFM": (5, -14), "Moirai": (7, 0), "Granite": (6, 1), "Lag-Llama": (-4, -17)}[r["model"]]
        ax.annotate(r["model"], (x, y), xytext=offset, textcoords="offset points", fontsize=8,
                    ha="right" if r["model"] == "Lag-Llama" else "left")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 10], ["1", "2", "4", "10"])
    ax.minorticks_off()
    ax.set_xlim(.79, 14.83)
    ax.set_ylim(*ylim)
    ax.set_yticks(ticks)
    ax.set_xlabel("Relative registered computation charge\n(Chronos-2 = 1; common GPU price)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{panel}   {title}", loc="left", y=1.16)
    ax.text(0, 1.035, "Five TSFMs; same 891 O tasks", transform=ax.transAxes, fontsize=8)

ax = axes[1, 1]
for r in (r for r in rows if r["panel"] == "D"):
    x = float(r["additional_call_hours_per1000"])
    y, low, high = (float(r[k]) for k in ["mase_difference_times1000", "ci95_lower_times1000", "ci95_upper_times1000"])
    color = blue if r["model"] == "S1" else orange
    ax.errorbar(x, y, yerr=[[y-low], [high-y]], fmt="o", color=color, capsize=3, markersize=5)
    ax.annotate(r["model"], (x, y), xytext=(6, 5), textcoords="offset points", fontsize=8, color=color)
ax.scatter([0], [0], marker="s", facecolors="white", edgecolors=grey, s=28)
ax.annotate("Equal weighting", (0, 0), xytext=(6, 8), textcoords="offset points", fontsize=8, color=grey)
ax.axhline(0, ls="--", lw=.8, color=grey)
ax.set_xlim(-.22, 3.68)
ax.set_ylim(-1.5, 1.3)
ax.set_xticks([0, 1, 2, 3])
ax.set_yticks([-1, 0, 1])
ax.set_xlabel("Additional request hours\n(per 1,000 delivered forecasts)")
ax.set_ylabel("MASE difference vs equal weighting × 1,000")
ax.set_title("D   M2: additional LLM workload", loc="left", y=1.16)
ax.text(0, 1.035, "Same candidates; apply local resource prices", transform=ax.transAxes, fontsize=8)
ax.text(.03, .03, "Negative difference favours LLM weighting", transform=ax.transAxes, fontsize=7.4)

for ax in axes.flat:
    ax.grid(axis="y", color="#e5e5e5", linewidth=.5)
    ax.set_axisbelow(True)
fig.legend(handles=[Line2D([], [], marker="o", color=blue, lw=1, label="A–C: metric-specific point-estimate frontier"),
                    Line2D([], [], marker="s", color=grey, markerfacecolor="white", lw=0, label="A–C: dominated within five TSFMs")],
           frameon=False, loc="lower center", bbox_to_anchor=(.5, .018), ncol=2, fontsize=8)
fig.text(.51, .072, "D: paired 95% intervals; request hours exclude candidate generation and unmeasured overheads.",
         ha="center", fontsize=8)
for ext in ["pdf", "png"]:
    path = OUT / f"accuracy_resources_rebuilt.{ext}"
    fig.savefig(path, dpi=300)
    print(path)
plt.close(fig)

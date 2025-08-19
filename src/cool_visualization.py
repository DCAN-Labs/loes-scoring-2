import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

warnings.filterwarnings("ignore")

plt.style.use("dark_background")
sns.set_palette("husl")

df = pd.read_csv("data/regression.csv")

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("#0a0a0a")

gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, :])

subject_scores = df.groupby("anonymized_subject_id")["loes-score"].agg(
    ["mean", "std", "count"]
)
subject_scores = subject_scores.sort_values("mean", ascending=False).head(15)

colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(subject_scores)))
bars = ax1.barh(
    range(len(subject_scores)),
    subject_scores["mean"],
    xerr=subject_scores["std"],
    color=colors,
    alpha=0.8,
    edgecolor="white",
    linewidth=1.5,
)

ax1.set_yticks(range(len(subject_scores)))
ax1.set_yticklabels(
    [s.replace("subject-", "S") for s in subject_scores.index], fontsize=9
)
ax1.set_xlabel("Mean LOES Score", fontsize=11, fontweight="bold")
ax1.set_title("Top Subjects by LOES Score", fontsize=13, fontweight="bold", pad=20)
ax1.grid(axis="x", alpha=0.2, linestyle="--")

for i, (bar, val) in enumerate(zip(bars, subject_scores["mean"])):
    ax1.text(
        val + 0.1,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.1f}",
        va="center",
        fontsize=9,
        color="cyan",
    )

score_bins = [0, 5, 10, 15, 20, 25, 30, 35]
score_labels = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-35"]
df["score_category"] = pd.cut(
    df["loes-score"], bins=score_bins, labels=score_labels, include_lowest=True
)
category_counts = df["score_category"].value_counts()

wedges, texts, autotexts = ax2.pie(
    category_counts.values,
    labels=category_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=sns.color_palette("coolwarm", len(category_counts)),
    explode=[0.05] * len(category_counts),
    shadow=True,
)

for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontsize(9)
    autotext.set_fontweight("bold")

ax2.set_title("LOES Score Distribution", fontsize=13, fontweight="bold", pad=20)

subject_progression = df.pivot_table(
    index="anonymized_session_id", columns="anonymized_subject_id", values="loes-score"
)

sample_subjects = df["anonymized_subject_id"].value_counts().head(8).index

for subject in sample_subjects:
    subject_data = df[df["anonymized_subject_id"] == subject].sort_values(
        "anonymized_session_id"
    )
    if len(subject_data) > 1:
        sessions = range(len(subject_data))
        scores = subject_data["loes-score"].values
        ax3.plot(
            sessions,
            scores,
            marker="o",
            linewidth=2,
            markersize=6,
            label=subject.replace("subject-", "S"),
            alpha=0.8,
        )

ax3.set_xlabel("Session Number", fontsize=11, fontweight="bold")
ax3.set_ylabel("LOES Score", fontsize=11, fontweight="bold")
ax3.set_title("Score Progression Over Time", fontsize=13, fontweight="bold", pad=20)
ax3.legend(loc="upper left", fontsize=8, framealpha=0.3)
ax3.grid(alpha=0.2, linestyle="--")

session_counts = df.groupby("anonymized_subject_id").size()
multi_session_subjects = session_counts[session_counts > 1]

score_changes = []
for subject in multi_session_subjects.index:
    subject_data = df[df["anonymized_subject_id"] == subject].sort_values(
        "anonymized_session_id"
    )
    if len(subject_data) > 1:
        initial_score = subject_data.iloc[0]["loes-score"]
        final_score = subject_data.iloc[-1]["loes-score"]
        change = final_score - initial_score
        score_changes.append(change)

if score_changes:
    n, bins, patches = ax4.hist(
        score_changes, bins=15, alpha=0.7, edgecolor="white", linewidth=1.5
    )

    for patch, bin_start, bin_end in zip(patches, bins[:-1], bins[1:]):
        if bin_end < 0:
            patch.set_facecolor("#e74c3c")
        elif bin_start > 0:
            patch.set_facecolor("#2ecc71")
        else:
            patch.set_facecolor("#f39c12")

    ax4.axvline(0, color="white", linestyle="-", linewidth=2, alpha=0.5)
    ax4.axvline(
        np.mean(score_changes),
        color="gold",
        linestyle="--",
        linewidth=2,
        label=f"Mean Change: {np.mean(score_changes):.1f}",
    )

    improved = sum(1 for x in score_changes if x < 0)
    worsened = sum(1 for x in score_changes if x > 0)
    unchanged = sum(1 for x in score_changes if x == 0)

    ax4.text(
        0.98,
        0.98,
        f"Improved: {improved}\nWorsened: {worsened}\nUnchanged: {unchanged}",
        transform=ax4.transAxes,
        fontsize=10,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.5, edgecolor="cyan"),
    )

ax4.set_xlabel("LOES Score Change (Final - Initial)", fontsize=11, fontweight="bold")
ax4.set_ylabel("Number of Subjects", fontsize=11, fontweight="bold")
ax4.set_title(
    "Score Change Distribution (Multi-Session Subjects)",
    fontsize=13,
    fontweight="bold",
    pad=20,
)
ax4.legend(loc="upper left", fontsize=9, framealpha=0.3)
ax4.grid(axis="y", alpha=0.2, linestyle="--")

stats_text = f"""
Dataset Statistics:
• Total Subjects: {df["anonymized_subject_id"].nunique()}
• Total Sessions: {len(df)}
• Score Range: {df["loes-score"].min():.1f} - {df["loes-score"].max():.1f}
• Mean Score: {df["loes-score"].mean():.2f} ± {df["loes-score"].std():.2f}
"""

fig.text(
    0.98,
    0.02,
    stats_text,
    fontsize=10,
    ha="right",
    va="bottom",
    bbox=dict(boxstyle="round", facecolor="black", alpha=0.5, edgecolor="cyan"),
)

fig.suptitle("🧠 LOES Score Analysis Dashboard", fontsize=16, fontweight="bold", y=0.98)

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_facecolor("#1a1a1a")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
        spine.set_linewidth(1)

plt.tight_layout()
plt.savefig(
    "loes_analysis_dashboard.png", dpi=150, facecolor="#0a0a0a", edgecolor="none"
)
plt.show()

print("\n✨ Visualization created successfully!")
print("📊 Dashboard saved as 'loes_analysis_dashboard.png'")
print("\nKey Insights:")
print(f"  • Analyzed {df['anonymized_subject_id'].nunique()} unique subjects")
print(f"  • Average LOES score: {df['loes-score'].mean():.2f}")
print(
    f"  • Score variance indicates {('high' if df['loes-score'].std() > 5 else 'moderate')} variability"
)
print(
    f"  • {'Most' if (df['loes-score'] < 10).sum() > len(df) / 2 else 'Many'} scores are in the lower range"
)

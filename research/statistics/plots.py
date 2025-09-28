import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from research.statistics.utils import LETTERS, build_letter_frequencies


def plot_transition_matrix(ax, df_probs, title=""):
    hm = sns.heatmap(
        df_probs.loc[list(LETTERS), list(LETTERS)],
        cmap="Reds",
        annot=False,
        cbar=False,
        ax=ax
    )
    ax.set_title(title, fontsize=12)
    return hm


def plot_letter_frequencies(males, females, sort_values=False, title=None):
    # Compute frequencies
    L_m = build_letter_frequencies(males['name']).set_index("letter")["freq"]
    L_f = build_letter_frequencies(females['name']).set_index("letter")["freq"]

    # Combine into one DataFrame
    df_plot = pd.DataFrame({"Male": L_m, "Female": L_f}).fillna(0).reset_index()
    df_plot.to_csv(f"../assets/{title}_letter_frequencies.csv", index=False)

    # Optional sorting
    if sort_values:
        df_plot = df_plot.sort_values("Male", ascending=False)

    # Plot side-by-side bars
    x = np.arange(len(df_plot))
    w = 0.4
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - w/2, df_plot["Male"], width=w, label="Male", color="steelblue", alpha=0.8)
    ax.bar(x + w/2, df_plot["Female"], width=w, label="Female", color="salmon", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["letter"])
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Letter")
    ax.set_title(f"{title} - Letter Frequencies")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

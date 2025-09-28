import seaborn as sns

from research.statistics.utils import LETTERS


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
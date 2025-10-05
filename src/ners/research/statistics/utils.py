import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
from typing import Dict, Any

LETTERS = "abcdefghijklmnopqrstuvwxyz"
START_TOKEN = "^"
END_TOKEN = "$"


def normalize_letters(s):
    """Normalize accents -> ascii, lowercase, keep only a-z."""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", errors="ignore").decode("utf-8")
    s = s.lower()
    s = re.sub(r"[^a-z]", "", s)
    return s


def build_category_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("province")["identified_category"]
        .value_counts(normalize=True)  # get proportions
        .unstack(fill_value=0)  # reshape into columns per word count
    )


def build_words_token(df: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
    # Normalize + split once (vectorized)
    s = df[source].fillna("").astype(str)
    s = s.str.lower().str.replace(r"[^\w'\-]+", " ", regex=True).str.strip().str.split()

    # Explode the token list into rows under `target`
    out = df.assign(**{target: s}).explode(target, ignore_index=True)

    # Drop NA/empty tokens and strip whitespace
    out[target] = out[target].astype(str).str.strip()
    out = out[out[target].ne("")].dropna(subset=[target]).reset_index(drop=True)

    return out


def build_letter_frequencies(series: pd.Series) -> pd.DataFrame:
    # Normalize: lowercase, remove non-letters, concatenate all into one string
    s = (
        series.astype(str)
        .str.lower()
        .str.replace(r"[^a-z]", "", regex=True)
        .str.cat(sep="")
    )

    # Convert string into Series of characters
    chars = pd.Series(list(s))

    # Count letters and ensure all letters are present
    out = (
        chars.value_counts(normalize=False)
        .reindex(list(LETTERS), fill_value=0)
        .rename_axis("letter")
        .reset_index(name="count")
    )

    # Relative frequency
    total = out["count"].sum()
    out["freq"] = out["count"] / (total if total > 0 else 1)
    return out


def build_transition_probabilities(names: pd.Series, alpha: float = 0.0) -> dict:
    # 1) Normalize
    names = names.astype(str).str.lower().str.replace(rf"[^{LETTERS}]", "", regex=True)
    names = names[names.str.len() > 0]

    # 2) Prepare sequences
    sequences = (START_TOKEN + names + END_TOKEN).tolist()

    # 3) Tokens and indices
    tokens = [START_TOKEN] + list(LETTERS) + [END_TOKEN]
    index = {t: i for i, t in enumerate(tokens)}
    V = len(tokens)

    # 4) ASCII lookup table (O(1) char -> idx); others -> -1
    lut = np.full(128, -1, dtype=np.int32)
    for ch, i in index.items():
        lut[ord(ch)] = i

    # 5) Concatenate with a separator that’s not in vocab to kill cross-boundary pairs
    concat = (" ".join(sequences)).encode("ascii", errors="ignore")

    # 6) Map bytes to indices
    arr = np.frombuffer(concat, dtype=np.uint8)
    idx = lut[arr]

    # 7) Build bigram pairs; drop invalid ones (separator & OOV)
    a = idx[:-1]
    b = idx[1:]
    mask = (a >= 0) & (b >= 0)
    a, b = a[mask], b[mask]

    # 8) Count with a single bincount
    lin = a * V + b
    counts = np.bincount(lin, minlength=V * V).reshape(V, V)

    # 9) Optional Laplace smoothing
    if alpha and alpha > 0:
        counts = counts + alpha

    # 10) Row-normalize to probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    # avoid division by zero
    probs = np.divide(counts, np.where(row_sums == 0, 1.0, row_sums), where=True)

    # 11) DataFrames
    df_counts = pd.DataFrame(counts, index=tokens, columns=tokens)
    df_probs = pd.DataFrame(probs, index=tokens, columns=tokens)

    return {
        "tokens": tokens,
        "index": index,
        "counts": counts,
        "df_counts": df_counts,
        "probs": probs,
        "df_probs": df_probs,
    }


def build_transition_comparisons(
    names_transitions: Dict[str, Any],
    surnames_transitions: Dict[str, Any],
    n_permutations: int = 1000,
) -> pd.DataFrame:
    """
    Compares letter transition probability matrices for names and surnames using
    various distance metrics and a permutation test for statistical significance.
    """

    # Helper function to flatten and smooth matrices
    def prepare_data(data):
        return {"m": data["m"]["probs"].flatten(), "f": data["f"]["probs"].flatten()}

    prepared_names = prepare_data(names_transitions)
    prepared_surnames = prepare_data(surnames_transitions)

    # Distance Metrics
    names_l2 = euclidean(prepared_names["m"], prepared_names["f"])
    surnames_l2 = euclidean(prepared_surnames["m"], prepared_surnames["f"])

    kl_names_mf = entropy(prepared_names["m"] + 1e-12, prepared_names["f"] + 1e-12)
    kl_names_fm = entropy(prepared_names["f"] + 1e-12, prepared_names["m"] + 1e-12)

    kl_surnames_mf = entropy(
        prepared_surnames["m"] + 1e-12, prepared_surnames["f"] + 1e-12
    )
    kl_surnames_fm = entropy(
        prepared_surnames["f"] + 1e-12, prepared_surnames["m"] + 1e-12
    )

    jsd_names = 0.5 * (kl_names_mf + kl_names_fm)
    jsd_surnames = 0.5 * (kl_surnames_mf + kl_surnames_fm)

    # Permutation Test
    def run_permutation_test(transitions):
        # Flattened probabilities for male and female
        P_m = transitions["m"]["probs"].flatten()
        P_f = transitions["f"]["probs"].flatten()

        # Calculate the observed JSD (our test statistic)
        observed_jsd = 0.5 * (
            entropy(P_m + 1e-12, P_f + 1e-12) + entropy(P_f + 1e-12, P_m + 1e-12)
        )

        # Concatenate male and female counts
        counts_m = transitions["m"]["counts"]
        counts_f = transitions["f"]["counts"]
        all_counts = np.concatenate((counts_m, counts_f), axis=1)
        total_counts = counts_m.shape[1] + counts_f.shape[1]

        permuted_jsds = []
        for _ in range(n_permutations):
            # Shuffle the columns (names) and split back into two groups
            shuffled_indices = np.random.permutation(total_counts)

            # Note: This is a simplified approach, assuming counts are
            # structured per name. A more robust implementation would
            # shuffle the actual names themselves.
            permuted_counts_m = all_counts[:, shuffled_indices[: counts_m.shape[1]]]
            permuted_counts_f = all_counts[:, shuffled_indices[counts_m.shape[1] :]]

            # Re-calculate probabilities and JSD for the permuted groups
            # Add a small epsilon to the denominator to prevent division by zero
            epsilon = 1e-12
            permuted_probs_m = permuted_counts_m / (
                permuted_counts_m.sum(axis=0, keepdims=True) + epsilon
            )
            permuted_probs_f = permuted_counts_f / (
                permuted_counts_f.sum(axis=0, keepdims=True) + epsilon
            )

            permuted_jsd = 0.5 * (
                entropy(
                    permuted_probs_m.mean(axis=1) + 1e-12,
                    permuted_probs_f.mean(axis=1) + 1e-12,
                )
                + entropy(
                    permuted_probs_f.mean(axis=1) + 1e-12,
                    permuted_probs_m.mean(axis=1) + 1e-12,
                )
            )
            permuted_jsds.append(permuted_jsd)

        # Calculate the p-value
        p_value = np.mean(np.array(permuted_jsds) >= observed_jsd)
        return p_value

    names_p_value = run_permutation_test(names_transitions)
    surnames_p_value = run_permutation_test(surnames_transitions)

    out = pd.DataFrame(
        {
            "l2": [names_l2, surnames_l2],
            "kl_mf": [kl_names_mf, kl_surnames_mf],
            "kl_fm": [kl_names_fm, kl_surnames_fm],
            "jsd": [jsd_names, jsd_surnames],
            "permutation_p_value": [names_p_value, surnames_p_value],
        },
        index=["names", "surnames"],
    )

    return out


import pandas as pd
from collections import Counter
from typing import Literal


def build_ngrams_count(
    df: pd.DataFrame,
    n: int,
    where: Literal["any", "prefix", "suffix"] = "any",
) -> pd.DataFrame:
    # Normalize and clean to a–z
    names = df["name"].astype(str).str.lower().str.replace(r"[^a-z]", "", regex=True)

    ngrams = []
    if where == "any":
        for s in names:
            L = len(s)
            if L >= n:
                ngrams.extend(s[i : i + n] for i in range(L - n + 1))
    elif where == "prefix":
        for s in names:
            if len(s) >= n:
                ngrams.append(s[:n])
    elif where == "suffix":
        for s in names:
            if len(s) >= n:
                ngrams.append(s[-n:])
    else:
        raise ValueError("where must be one of: 'any', 'prefix', 'suffix'")

    counter = Counter(ngrams)

    out = (
        pd.DataFrame(counter.items(), columns=[f"{n}-gram", "count"])
        .sort_values("count", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    total = out["count"].sum()
    out["freq"] = out["count"] / (total if total > 0 else 1)
    return out

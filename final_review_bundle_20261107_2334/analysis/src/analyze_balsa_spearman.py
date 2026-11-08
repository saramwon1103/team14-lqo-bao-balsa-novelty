from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "analysis"
    / "data"
    / "balsa_novelty_performance.csv"
)

OUTPUT_FILE = (
    ROOT
    / "analysis"
    / "data"
    / "balsa_spearman_summary.csv"
)


def strength_label(rho):
    value = abs(rho)

    if value < 0.20:
        return "Very weak"
    elif value < 0.40:
        return "Weak"
    elif value < 0.60:
        return "Moderate"
    elif value < 0.80:
        return "Strong"
    else:
        return "Very strong"


def analyze_group(df, split_type):

    valid = df[
        (df["balsa_status"] == "OK")
        & (df["pg_status"] == "OK")
        & df["novelty"].notna()
        & df["balsa_vs_pg_ratio"].notna()
    ].copy()

    if len(valid) < 3:
        return {
            "split_type": split_type,
            "n": len(valid),
            "rho": None,
            "p_value": None,
            "strength": "Insufficient data",
            "direction": "N/A",
        }

    rho, p_value = spearmanr(
        valid["novelty"],
        valid["balsa_vs_pg_ratio"],
    )

    if rho > 0:
        direction = "Positive"
    elif rho < 0:
        direction = "Negative"
    else:
        direction = "None"

    return {
        "split_type": split_type,
        "n": len(valid),
        "rho": rho,
        "p_value": p_value,
        "strength": strength_label(rho),
        "direction": direction,
    }


def main():

    df = pd.read_csv(INPUT_FILE)

    print("=== INPUT ===")
    print(f"Rows: {len(df)}")

    comparable = df[
        (df["balsa_status"] == "OK")
        & (df["pg_status"] == "OK")
        & df["balsa_vs_pg_ratio"].notna()
    ]

    print(
        f"Comparable rows: "
        f"{len(comparable)}"
    )

    results = []

    # Overall
    results.append(
        analyze_group(
            df,
            "ALL",
        )
    )

    # Per split
    for split_type in [
        "LOO",
        "Random",
        "BaseQuery",
    ]:

        split_df = df[
            df["split_type"]
            == split_type
        ]

        results.append(
            analyze_group(
                split_df,
                split_type,
            )
        )

    result_df = pd.DataFrame(
        results
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\n=== SPEARMAN ==="
    )

    print(
        result_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def read_compliance_rate(eval_json_path: str) -> float:
    """
    Computes compliance rate (1 - refusal_rate) from a saved *_evaluations.json file.
    """
    if not os.path.exists(eval_json_path):
        return 0.0

    with open(eval_json_path, "r") as f:
        data = json.load(f)

    # Support either summary metric keys or per-item lists
    if "substring_matching_success_rate" in data:
        return float(data["substring_matching_success_rate"])
    if "compliance_rate" in data:
        return float(data["compliance_rate"])

    completions = data.get("completions", data if isinstance(data, list) else [])
    if not completions:
        return 0.0

    complied = sum(
        1 for item in completions
        if item.get("is_jailbreak", not item.get("refused", True))
    )
    return complied / len(completions)


def plot_grouped_compliance(
    artifacts_root: str,
    langs: list,
    interventions: dict,
    dataset_name: str,
    model_title: str,
    out_path: str,
):
    """
    Plots grouped bar charts across evaluation languages for multiple interventions
    (e.g., Baseline vs. EN Ablation / Addition).
    """
    x = np.arange(len(langs))
    n_bars = len(interventions)
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(max(8, len(langs) * 0.7), 4.5))

    for idx, (label, folder_pattern) in enumerate(interventions.items()):
        rates = []
        for lang in langs:
            # Expects path like: runs/tiny-aya-global_{lang}/completions/harmful_{intervention}_evaluations.json
            eval_file = os.path.join(
                artifacts_root.format(lang=lang),
                "completions",
                f"{dataset_name}_{folder_pattern}_evaluations.json",
            )
            rates.append(read_compliance_rate(eval_file) * 100.0)

        offset = (idx - (n_bars - 1) / 2) * width
        ax.bar(x + offset, rates, width, label=label)

    ax.set_ylabel("Compliance Rate (%)", fontweight="bold")
    ax.set_title(model_title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(langs, rotation=35)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=n_bars, frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved compliance rate chart -> {out_path}")


if __name__ == "__main__":
    # Example usage matching Figure 1 / Figure 2
    langs = ["en", "de", "es", "fr", "it", "nl", "ja", "pl", "ru", "zh", "ko", "ar", "th", "yo"]
    interventions = {
        "Baseline": "baseline",
        "EN Ablation": "harm_ablation",
        # Add "EN Addition": "harm_actadd" for Figure 3
    }

    plot_grouped_compliance(
        artifacts_root="runs/tiny-aya-global_{lang}",
        langs=langs,
        interventions=interventions,
        dataset_name="harmful",
        model_title="tiny-aya-global",
        out_path="tiny_aya_compliance_rates.png",
    )

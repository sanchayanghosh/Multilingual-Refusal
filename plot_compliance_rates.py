import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def read_compliance_rate(eval_json_path: str) -> float:
    if not os.path.exists(eval_json_path):
        print(f"Warning: missing {eval_json_path}")
        return 0.0

    with open(eval_json_path, "r") as f:
        data = json.load(f)

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
    completions_dir: str,
    source_lang: str,
    eval_langs: list,
    interventions: dict,
    dataset_name: str,
    model_title: str,
    out_path: str,
):
    x = np.arange(len(eval_langs))
    n_bars = len(interventions)
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(max(6, len(eval_langs) * 1.4), 4.5))

    for idx, (label, pattern) in enumerate(interventions.items()):
        rates = []
        for lang in eval_langs:
            tag = pattern.format(src=source_lang, lang=lang)
            eval_file = os.path.join(completions_dir, f"{dataset_name}_{tag}_evaluations.json")
            rates.append(read_compliance_rate(eval_file) * 100.0)

        offset = (idx - (n_bars - 1) / 2) * width
        ax.bar(x + offset, rates, width, label=label)

    ax.set_ylabel("Compliance Rate (%)", fontweight="bold")
    ax.set_title(model_title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(eval_langs)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=n_bars, frameon=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved compliance rate chart -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--completions_dir", default="/workspace/pipeline/runs/gemma-2-9b-it-indicsafe/completions")
    parser.add_argument("--source_lang", default="en")
    parser.add_argument("--eval_langs", nargs="+", default=["en", "hi"])
    parser.add_argument("--title_prefix", default="gemma-2-9b-it (IndicSafe)")
    parser.add_argument("--out_prefix", default="/workspace/indicsafe")
    args = parser.parse_args()

    plot_grouped_compliance(
        completions_dir=args.completions_dir,
        source_lang=args.source_lang,
        eval_langs=args.eval_langs,
        interventions={
            "Baseline": "baseline_{lang}",
            "EN Ablation": "harm_ablation_{src}_on_{lang}",
        },
        dataset_name="harmful",
        model_title=f"{args.title_prefix} - Ablation",
        out_path=f"{args.out_prefix}_ablation_compliance.png",
    )

    plot_grouped_compliance(
        completions_dir=args.completions_dir,
        source_lang=args.source_lang,
        eval_langs=args.eval_langs,
        interventions={
            "Baseline": "baseline_{lang}",
            "EN Addition": "harm_actadd_{src}_on_{lang}",
        },
        dataset_name="harmful",
        model_title=f"{args.title_prefix} - Addition",
        out_path=f"{args.out_prefix}_addition_compliance.png",
    )

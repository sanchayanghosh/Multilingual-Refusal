import argparse
import copy
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import mmengine
from huggingface_hub import HfApi, login

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from pipeline.model_utils.plot_compliance_rates import plot_grouped_compliance
from pipeline.model_utils.plot_geometry_pca import plot_pca_pair
from pipeline.reports.analyze_activations_offline import (
    compute_mean_diffs,
    load_merged_split,
    plot_cross_lingual_similarity,
    plot_direction_norms,
    plot_layer_cosine_similarity,
)

DEFAULT_CONFIG = WORKSPACE / "config_aya_expanse_indicsafe.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Aya Expanse 8b on IndicSafe, build the report, and publish all artifacts to HF."
    )
    parser.add_argument("--config_path", default=str(DEFAULT_CONFIG), help="Path to the master YAML config")
    parser.add_argument("--model_path", default=None, help="Optional override for model_path")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional override for batch_size")
    parser.add_argument("--artifact_path", default=None, help="Optional override for the local artifact directory")
    parser.add_argument("--report_repo", default=None, help="HF dataset repo for the report and figures")
    parser.add_argument("--report_only", action="store_true", help="Skip model runs and rebuild the report from local artifacts")
    parser.add_argument("--skip_hf_plots", action="store_true", help="Skip PCA and activation-geometry plots that read activations from HF")
    parser.add_argument("--sanity", action="store_true", help="Use tiny sample counts and suffixed HF repos for a smoke test")
    return parser.parse_args()


def load_config(args):
    cfg = mmengine.Config.fromfile(str(args.config_path))

    if args.model_path:
        cfg.model_path = args.model_path
    if args.batch_size:
        cfg.batch_size = args.batch_size

    cfg.model_alias = Path(cfg.model_path).name
    artifact_path = args.artifact_path or str(WORKSPACE / "pipeline" / "runs" / f"{cfg.model_alias}-indicsafe")

    if args.sanity:
        artifact_path = f"{artifact_path}-sanity"
        cfg.n_train = 8
        cfg.n_val = 4
        cfg.n_test = 4
        cfg.filter_train = False
        cfg.filter_val = False
        cfg.filter_or = False
        cfg.batch_size = args.batch_size or 4
        cfg.max_new_tokens = 64
        for key in ["hf_repo_id", "hf_activations_repo_id", "hf_report_repo_id"]:
            if key in cfg:
                cfg[key] = f"{cfg[key]}-sanity"

    cfg.artifact_path = artifact_path
    os.makedirs(cfg.artifact_path, exist_ok=True)
    return cfg


def init_hub(cfg, args):
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN must be set to upload activations, prompts, and the report.")
    login(token=token)
    api = HfApi(token=token)

    report_repo = args.report_repo or getattr(
        cfg, "hf_report_repo_id", f"renessauce/{cfg.model_alias}-indicsafe-report"
    )
    for repo_id in [cfg.hf_repo_id, cfg.hf_activations_repo_id, report_repo]:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    return api, report_repo


def run_pipeline_passes(cfg):
    source_langs = getattr(cfg, "source_langs", [cfg.source_lang])

    for lang in source_langs:
        sub_cfg = copy.deepcopy(cfg)
        sub_cfg.source_lang = lang
        sub_cfg.lang = lang
        sub_cfg.artifact_path = os.path.join(cfg.artifact_path, lang)
        os.makedirs(sub_cfg.artifact_path, exist_ok=True)

        sub_cfg_path = os.path.join(sub_cfg.artifact_path, f"config_{lang}.yaml")
        sub_cfg.dump(sub_cfg_path)

        cmd = [sys.executable, "-m", "pipeline.run_pipeline", "--config_path", sub_cfg_path]
        print(f"\n[Run] source_lang={lang}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, cwd=str(WORKSPACE))


def read_rate(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)

    rate = data.get("substring_matching_success_rate", data.get("compliance_rate"))
    if rate is None:
        completions = data.get("completions", data if isinstance(data, list) else [])
        if completions:
            rate = sum(
                1 for item in completions
                if item.get("is_jailbreak", not item.get("refused", True))
            ) / len(completions)
    return float(rate) if rate is not None else None


def build_results(cfg, report_dir):
    eval_langs = getattr(cfg, "eval_langs", getattr(cfg, "source_langs", [cfg.source_lang]))
    source_langs = getattr(cfg, "source_langs", [cfg.source_lang])
    rows = []

    for source_lang in source_langs:
        completions_dir = Path(cfg.artifact_path) / source_lang / "completions"
        for eval_lang in eval_langs:
            baseline = read_rate(completions_dir / f"harmful_baseline_{eval_lang}_evaluations.json")
            patterns = {
                "baseline": f"baseline_{eval_lang}",
                "ablation": f"harm_ablation_{source_lang}_on_{eval_lang}",
                "addition": f"harm_actadd_{source_lang}_on_{eval_lang}",
            }
            for intervention, pattern in patterns.items():
                rate = read_rate(completions_dir / f"harmful_{pattern}_evaluations.json")
                rows.append(
                    {
                        "source_lang": source_lang,
                        "eval_lang": eval_lang,
                        "intervention": intervention,
                        "compliance_rate": rate,
                        "delta_vs_baseline": (rate - baseline) if (rate is not None and baseline is not None) else None,
                    }
                )

    with (report_dir / "results.json").open("w") as f:
        json.dump(rows, f, indent=4)

    with (report_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [
            "source_lang", "eval_lang", "intervention", "compliance_rate", "delta_vs_baseline"
        ])
        writer.writeheader()
        writer.writerows(rows)

    return rows


def read_direction_metadata(cfg):
    metadata = {}
    for lang in getattr(cfg, "source_langs", [cfg.source_lang]):
        path = Path(cfg.artifact_path) / lang / "direction_metadata_ablation.json"
        if path.exists():
            with path.open() as f:
                data = json.load(f)
            metadata[lang] = {
                "pos": data["pos"][0] if isinstance(data.get("pos"), list) else data.get("pos"),
                "layer": data["layer"][0] if isinstance(data.get("layer"), list) else data.get("layer"),
            }
    return metadata


def generate_figures(cfg, report_dir, direction_metadata, skip_hf_plots):
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    source_langs = getattr(cfg, "source_langs", [cfg.source_lang])
    eval_langs = getattr(cfg, "eval_langs", source_langs)
    ref_lang = source_langs[0]
    completions_dir = Path(cfg.artifact_path) / ref_lang / "completions"
    title = f"{cfg.model_alias} (IndicSafe)"

    ablation_path = figures_dir / "ablation_compliance.png"
    addition_path = figures_dir / "addition_compliance.png"
    plot_grouped_compliance(
        completions_dir=str(completions_dir),
        source_lang=ref_lang,
        eval_langs=eval_langs,
        interventions={
            "Baseline": "baseline_{lang}",
            f"{ref_lang.upper()} Ablation": "harm_ablation_{src}_on_{lang}",
        },
        dataset_name="harmful",
        model_title=f"{title} - Ablation",
        out_path=str(ablation_path),
    )
    plot_grouped_compliance(
        completions_dir=str(completions_dir),
        source_lang=ref_lang,
        eval_langs=eval_langs,
        interventions={
            "Baseline": "baseline_{lang}",
            f"{ref_lang.upper()} Addition": "harm_actadd_{src}_on_{lang}",
        },
        dataset_name="harmful",
        model_title=f"{title} - Addition",
        out_path=str(addition_path),
    )
    figures = [ablation_path, addition_path]

    if not direction_metadata or skip_hf_plots:
        return figures

    pos_idx = direction_metadata[ref_lang]["pos"] if ref_lang in direction_metadata else -1
    layer_idx = direction_metadata[ref_lang]["layer"] if ref_lang in direction_metadata else 0

    for target_lang in source_langs:
        if target_lang == ref_lang:
            continue
        out_path = figures_dir / f"pca_{ref_lang}_{target_lang}.png"
        plot_pca_pair(
            repo_id=cfg.hf_activations_repo_id,
            ref_lang=ref_lang,
            target_lang=target_lang,
            pos_idx=pos_idx,
            layer_idx=layer_idx,
            target_eval_json=str(completions_dir / f"harmful_baseline_{target_lang}_evaluations.json"),
            out_path=str(out_path),
            title_prefix=title,
        )
        figures.append(out_path)

    cache_dir = report_dir / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mean_diffs_by_lang = {}
    for lang in source_langs:
        harmful_acts, _ = load_merged_split(cfg.hf_activations_repo_id, lang, "harmful_train", str(cache_dir))
        harmless_acts, _ = load_merged_split(cfg.hf_activations_repo_id, lang, "harmless_train", str(cache_dir))
        mean_diffs = compute_mean_diffs(harmful_acts, harmless_acts)
        mean_diffs_by_lang[lang] = mean_diffs

        heatmap_path = figures_dir / f"{lang}_layer_cosine_sim_pos{pos_idx}.png"
        norms_path = figures_dir / f"{lang}_direction_norms.png"
        plot_layer_cosine_similarity(mean_diffs, lang, pos_idx, str(figures_dir))
        plot_direction_norms(mean_diffs, lang, str(figures_dir))
        figures.extend([heatmap_path, norms_path])

    cross_path = figures_dir / f"cross_lingual_cosine_sim_pos{pos_idx}.png"
    plot_cross_lingual_similarity(mean_diffs_by_lang, pos_idx, str(figures_dir))
    if cross_path.exists():
        figures.append(cross_path)

    return [path for path in figures if path.exists()]


def format_rate(value):
    return "n/a" if value is None else f"{100 * value:.1f}%"


def write_report(cfg, report_dir, rows, direction_metadata, figures, report_repo):
    source_langs = getattr(cfg, "source_langs", [cfg.source_lang])
    lines = [
        f"# {cfg.model_alias} refusal-direction transfer on IndicSafe",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Setup",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Model | `{cfg.model_path}` |",
        "| Dataset | IndicSafe (harmful) + indic-align (harmless) |",
        f"| Source languages | {', '.join(source_langs)} |",
        f"| Eval languages | {', '.join(getattr(cfg, 'eval_langs', source_langs))} |",
        f"| Train / val / test prompts per source language | {cfg.n_train} / {cfg.n_val} / {cfg.n_test} |",
        f"| Random seed | {cfg.random_seed} |",
        f"| Ablation coefficient | {cfg.ablation_coeff} |",
        f"| Activation addition coefficient | {cfg.addact_coeff} |",
        f"| KL threshold | {cfg.ablate_kl_threshold} |",
        f"| Intervention top-n | {cfg.top_n} |",
        f"| Prompts | https://huggingface.co/datasets/{cfg.hf_repo_id} |",
        f"| Activations, directions, completions | https://huggingface.co/datasets/{cfg.hf_activations_repo_id} |",
        f"| This report | https://huggingface.co/datasets/{report_repo} |",
        "",
        "## Selected directions",
        "",
        "| Source language | EOI position | Layer |",
        "| --- | --- | --- |",
    ]

    for lang, meta in direction_metadata.items():
        lines.append(f"| {lang} | {meta['pos']} | {meta['layer']} |")
    if not direction_metadata:
        lines.append("| n/a | n/a | n/a |")

    lines.extend([
        "",
        "## Compliance rates",
        "",
        "Higher compliance means more jailbreaks (lower refusal). Deltas are relative to the baseline for the same language pair.",
        "",
        "| Source | Eval | Intervention | Compliance | Delta |",
        "| --- | --- | --- | --- | --- |",
    ])
    for row in rows:
        delta = row["delta_vs_baseline"]
        delta_str = "n/a" if delta is None else f"{100 * delta:+.1f} pp"
        lines.append(
            f"| {row['source_lang']} | {row['eval_lang']} | {row['intervention']} | "
            f"{format_rate(row['compliance_rate'])} | {delta_str} |"
        )

    lines.extend(["", "## Figures", ""])
    for figure in figures:
        lines.append(f"### {figure.stem.replace('_', ' ')}")
        lines.append("")
        lines.append(f"![{figure.stem}](figures/{figure.name})")
        lines.append("")

    (report_dir / "report.md").write_text("\n".join(lines) + "\n")
    (report_dir / "README.md").write_text(
        "# Aya Expanse 8b IndicSafe report\n\n"
        f"- [Markdown report](report.md)\n"
        f"- [Results (JSON)](results.json)\n"
        f"- [Results (CSV)](results.csv)\n"
        f"- Prompts: https://huggingface.co/datasets/{cfg.hf_repo_id}\n"
        f"- Activations: https://huggingface.co/datasets/{cfg.hf_activations_repo_id}\n"
    )


def upload_report(api, report_repo, report_dir):
    api.upload_folder(
        folder_path=str(report_dir),
        repo_id=report_repo,
        repo_type="dataset",
        commit_message="Upload Aya Expanse 8b IndicSafe report, figures, and results",
    )
    print(f"[HF] Report uploaded to https://huggingface.co/datasets/{report_repo}", flush=True)


def main():
    args = parse_args()
    cfg = load_config(args)
    api, report_repo = init_hub(cfg, args)

    if not args.report_only:
        run_pipeline_passes(cfg)

    report_dir = Path(cfg.artifact_path) / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    source_langs = getattr(cfg, "source_langs", [cfg.source_lang])
    ref_config = Path(cfg.artifact_path) / source_langs[0] / "config_run.yaml"
    if ref_config.exists():
        shutil.copy2(ref_config, report_dir / "config_run.yaml")

    rows = build_results(cfg, report_dir)
    direction_metadata = read_direction_metadata(cfg)
    figures = generate_figures(cfg, report_dir, direction_metadata, args.skip_hf_plots)
    write_report(cfg, report_dir, rows, direction_metadata, figures, report_repo)
    upload_report(api, report_repo, report_dir)

    print(f"[Done] Local report: {report_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()

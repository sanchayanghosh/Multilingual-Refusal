import os
import sys
import json
import copy
import argparse
import subprocess
import mmengine

WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)

from pipeline.model_utils.plot_compliance_rates import plot_grouped_compliance
from pipeline.model_utils.plot_geometry_pca import plot_pca_pair


def main():
    parser = argparse.ArgumentParser(description="Run full multi-language pipeline and generate all plots.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to master YAML config")
    parser.add_argument("--model_path", type=str, default=None, help="Optional override for model_path")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional override for batch_size")
    args = parser.parse_args()

    base_cfg = mmengine.Config.fromfile(args.config_path)
    base_artifact_dir = base_cfg.artifact_path
    os.makedirs(base_artifact_dir, exist_ok=True)

    source_langs = getattr(base_cfg, "source_langs", [getattr(base_cfg, "source_lang", "en")])
    eval_langs = getattr(base_cfg, "eval_langs", source_langs)
    ref_lang = source_langs[0]

    # 1. Run each source language pass in a clean subprocess so GPU VRAM is 100% freed between passes
    lang_artifact_dirs = {}
    pipeline_script = os.path.join(os.path.dirname(__file__), "run_pipeline.py")

    for s_lang in source_langs:
        print(f"\n{'='*60}\n[Orchestrator] Running pipeline pass for source_lang='{s_lang}'\n{'='*60}")
        sub_cfg = copy.deepcopy(base_cfg)
        sub_cfg.source_lang = s_lang
        sub_cfg.lang = s_lang
        sub_cfg.artifact_path = os.path.join(base_artifact_dir, s_lang)
        os.makedirs(sub_cfg.artifact_path, exist_ok=True)

        sub_cfg_path = os.path.join(sub_cfg.artifact_path, f"config_{s_lang}.yaml")
        sub_cfg.dump(sub_cfg_path)

        cmd = [sys.executable, pipeline_script, "--config_path", sub_cfg_path]
        if args.model_path is not None:
            cmd.extend(["--model_path", args.model_path])
        if args.batch_size is not None:
            cmd.extend(["--batch_size", str(args.batch_size)])

        subprocess.run(cmd, check=True)
        lang_artifact_dirs[s_lang] = sub_cfg.artifact_path

    # 2. Generate Compliance Rate Plots (Figures 1-3)
    ref_completions_dir = os.path.join(lang_artifact_dirs[ref_lang], "completions")
    dataset_tag = getattr(base_cfg, "dataset_source", "indicsafe").upper()
    model_alias = os.path.basename(args.model_path or base_cfg.model_path)

    ablation_plot_path = os.path.join(WORKSPACE_DIR, f"{dataset_tag.lower()}_ablation_compliance.png")
    addition_plot_path = os.path.join(WORKSPACE_DIR, f"{dataset_tag.lower()}_addition_compliance.png")

    plot_grouped_compliance(
        completions_dir=ref_completions_dir,
        source_lang=ref_lang,
        eval_langs=eval_langs,
        interventions={
            "Baseline": "baseline_{lang}",
            f"{ref_lang.upper()} Ablation": "harm_ablation_{src}_on_{lang}",
        },
        dataset_name="harmful",
        model_title=f"{model_alias} ({dataset_tag} - Ablation)",
        out_path=ablation_plot_path,
    )

    plot_grouped_compliance(
        completions_dir=ref_completions_dir,
        source_lang=ref_lang,
        eval_langs=eval_langs,
        interventions={
            "Baseline": "baseline_{lang}",
            f"{ref_lang.upper()} Addition": "harm_actadd_{src}_on_{lang}",
        },
        dataset_name="harmful",
        model_title=f"{model_alias} ({dataset_tag} - Addition)",
        out_path=addition_plot_path,
    )

    # 3. Generate PCA Geometry Plots (Figure 4)
    meta_path = os.path.join(lang_artifact_dirs[ref_lang], "direction_metadata_ablation.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    pos_idx = meta["pos"][0] if isinstance(meta["pos"], list) else meta["pos"]
    layer_idx = meta["layer"][0] if isinstance(meta["layer"], list) else meta["layer"]

    for target_lang in source_langs:
        if target_lang == ref_lang:
            continue
        target_eval_json = os.path.join(ref_completions_dir, f"harmful_baseline_{target_lang}_evaluations.json")
        pca_out_path = os.path.join(WORKSPACE_DIR, f"pca_{dataset_tag.lower()}_{ref_lang}_{target_lang}.png")

        plot_pca_pair(
            repo_id=base_cfg.hf_activations_repo_id,
            ref_lang=ref_lang,
            target_lang=target_lang,
            pos_idx=pos_idx,
            layer_idx=layer_idx,
            target_eval_json=target_eval_json,
            out_path=pca_out_path,
            title_prefix=f"{model_alias} ({dataset_tag})",
        )

    print("\n[Orchestrator] All pipeline passes and plots complete!")


if __name__ == "__main__":
    main()

import os
import glob
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import snapshot_download, login


def load_merged_split(repo_id: str, lang: str, split_name: str, local_cache_dir: str = "./hf_cache"):
    """
    Downloads all shard_*.pt files for a given language and split from Hugging Face
    and concatenates them along the prompt dimension.
    
    Returns:
        activations: Tensor of shape [total_prompts, eoi_len, num_layers, d_model]
        prompts: List of prompt strings
    """
    pattern = f"{lang}/activations/{split_name}/shard_*.pt"
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=pattern,
        local_dir=local_cache_dir,
    )

    shard_paths = sorted(glob.glob(os.path.join(local_dir, lang, "activations", split_name, "shard_*.pt")))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found matching {pattern} in {repo_id}")

    act_chunks = []
    prompts = []
    for path in shard_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        act_chunks.append(payload["activations"].float())
        prompts.extend(payload["prompts"])

    merged_acts = torch.cat(act_chunks, dim=0)
    print(f"[{lang} | {split_name}] Loaded {len(shard_paths)} shards -> shape {tuple(merged_acts.shape)}")
    return merged_acts, prompts


def compute_mean_diffs(harmful_acts: torch.Tensor, harmless_acts: torch.Tensor) -> torch.Tensor:
    """
    Computes candidate refusal directions: mean(harmful) - mean(harmless).
    
    Args:
        harmful_acts: [N_harm, eoi_len, num_layers, d_model]
        harmless_acts: [N_harmless, eoi_len, num_layers, d_model]
    Returns:
        mean_diffs: [eoi_len, num_layers, d_model]
    """
    harmful_mean = harmful_acts.mean(dim=0)
    harmless_mean = harmless_acts.mean(dim=0)
    return harmful_mean - harmless_mean


def plot_layer_cosine_similarity(mean_diffs: torch.Tensor, lang: str, pos_idx: int, out_dir: str):
    """
    Plots a [num_layers x num_layers] cosine similarity matrix of the refusal direction
    at a chosen EOI token position (e.g., -1 for the final assistant header token).
    """
    # directions: [num_layers, d_model]
    directions = mean_diffs[pos_idx]
    normed = F.normalize(directions, p=2, dim=-1)
    cos_sim = (normed @ normed.T).numpy()

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cos_sim,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        cbar_kws={"label": "Cosine Similarity"},
    )
    plt.title(f"Layer-to-Layer Refusal Direction Cosine Similarity ({lang}, pos={pos_idx})")
    plt.xlabel("Layer")
    plt.ylabel("Layer")
    plt.tight_layout()

    save_path = os.path.join(out_dir, f"{lang}_layer_cosine_sim_pos{pos_idx}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved heatmap -> {save_path}")


def plot_direction_norms(mean_diffs: torch.Tensor, lang: str, out_dir: str):
    """
    Plots the L2 norm of mean_diffs across layers for every EOI token position.
    """
    # norms: [eoi_len, num_layers]
    norms = torch.norm(mean_diffs, p=2, dim=-1).numpy()
    eoi_len, num_layers = norms.shape

    plt.figure(figsize=(10, 5))
    for pos in range(eoi_len):
        rel_pos = pos - eoi_len  # e.g., -5, -4, -3, -2, -1
        plt.plot(range(num_layers), norms[pos], marker="o", markersize=3, label=f"EOI pos {rel_pos}")

    plt.title(f"Candidate Refusal Direction L2 Norm Across Layers ({lang})")
    plt.xlabel("Layer")
    plt.ylabel("L2 Norm ||mean_harmful - mean_harmless||")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(out_dir, f"{lang}_direction_norms.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved norm plot -> {save_path}")


def plot_cross_lingual_similarity(mean_diffs_by_lang: dict, pos_idx: int, out_dir: str):
    """
    If multiple languages are loaded, plots layer-by-layer cosine similarity
    between the first (reference) language and all other languages.
    """
    langs = list(mean_diffs_by_lang.keys())
    if len(langs) < 2:
        return

    ref_lang = langs[0]
    ref_dirs = F.normalize(mean_diffs_by_lang[ref_lang][pos_idx], p=2, dim=-1)  # [num_layers, d_model]
    num_layers = ref_dirs.shape[0]

    plt.figure(figsize=(10, 5))
    for other_lang in langs[1:]:
        other_dirs = F.normalize(mean_diffs_by_lang[other_lang][pos_idx], p=2, dim=-1)
        layer_cos = (ref_dirs * other_dirs).sum(dim=-1).numpy()
        plt.plot(range(num_layers), layer_cos, marker="o", markersize=4, label=f"{ref_lang} vs {other_lang}")

    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.title(f"Cross-Lingual Refusal Direction Cosine Similarity Across Layers (pos={pos_idx})")
    plt.xlabel("Layer")
    plt.ylabel("Cosine Similarity")
    plt.ylim(-0.5, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(out_dir, f"cross_lingual_cosine_sim_pos{pos_idx}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved cross-lingual plot -> {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Offline refusal direction & cosine similarity analysis.")
    parser.add_argument("--repo_id", type=str, default="renessauce/tiny-aya-global-activations")
    parser.add_argument("--langs", nargs="+", default=["yo"], help="One or more language codes (e.g., en yo)")
    parser.add_argument("--pos_idx", type=int, default=-1, help="EOI token position index (-1 = last token)")
    parser.add_argument("--out_dir", type=str, default="./offline_analysis_outputs")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    os.makedirs(args.out_dir, exist_ok=True)
    mean_diffs_by_lang = {}

    for lang in args.langs:
        harmful_acts, _ = load_merged_split(args.repo_id, lang, "harmful_train")
        harmless_acts, _ = load_merged_split(args.repo_id, lang, "harmless_train")

        mean_diffs = compute_mean_diffs(harmful_acts, harmless_acts)
        mean_diffs_by_lang[lang] = mean_diffs

        # Save merged mean_diffs locally for reuse
        torch.save(mean_diffs, os.path.join(args.out_dir, f"{lang}_mean_diffs.pt"))

        # Generate per-language plots
        plot_layer_cosine_similarity(mean_diffs, lang, args.pos_idx, args.out_dir)
        plot_direction_norms(mean_diffs, lang, args.out_dir)

    # Generate cross-lingual comparison if >= 2 languages were passed
    plot_cross_lingual_similarity(mean_diffs_by_lang, args.pos_idx, args.out_dir)


if __name__ == "__main__":
    main()

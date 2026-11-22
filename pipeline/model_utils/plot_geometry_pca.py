import os
import glob
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from huggingface_hub import snapshot_download, login


def load_merged_split(repo_id: str, lang: str, split_name: str, cache_dir: str = "./hf_cache"):
    pattern = f"{lang}/activations/{split_name}/shard_*.pt"
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=pattern,
        local_dir=cache_dir,
    )
    shard_paths = sorted(glob.glob(os.path.join(local_dir, lang, "activations", split_name, "shard_*.pt")))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found for {pattern}")

    acts, prompts = [], []
    for p in shard_paths:
        data = torch.load(p, map_location="cpu", weights_only=False)
        acts.append(data["activations"].float())
        prompts.extend(data["prompts"])
    return torch.cat(acts, dim=0), prompts  # [N, eoi_len, num_layers, d_model]


def load_jailbreak_flags(eval_json_path: str, num_prompts: int):
    """
    Loads boolean flags (True = jailbroken/complied, False = refused) from your
    saved baseline evaluation JSON. If unavailable, defaults all to refused (False).
    """
    if not eval_json_path or not os.path.exists(eval_json_path):
        return np.zeros(num_prompts, dtype=bool)

    with open(eval_json_path, "r") as f:
        eval_data = json.load(f)

    # Adapts to standard pipeline evaluation dictionary formats
    completions = eval_data.get("completions", eval_data if isinstance(eval_data, list) else [])
    flags = []
    for item in completions[:num_prompts]:
        # True if jailbreak succeeded (no refusal), False if refused
        is_jb = item.get("is_jailbreak", not item.get("refused", True))
        flags.append(bool(is_jb))

    if len(flags) < num_prompts:
        flags.extend([False] * (num_prompts - len(flags)))
    return np.array(flags, dtype=bool)


def plot_pca_pair(
    repo_id: str,
    ref_lang: str,
    target_lang: str,
    pos_idx: int,
    layer_idx: int,
    target_eval_json: str,
    out_path: str,
):
    # Load activations at [N, d_model] for the chosen (pos_idx, layer_idx)
    en_harmless, _ = load_merged_split(repo_id, ref_lang, "harmless_train")
    en_harmful, _ = load_merged_split(repo_id, ref_lang, "harmful_train")
    tg_harmless, _ = load_merged_split(repo_id, target_lang, "harmless_train")
    tg_harmful, _ = load_merged_split(repo_id, target_lang, "harmful_train")

    X_en_hl = en_harmless[:, pos_idx, layer_idx, :].numpy()
    X_en_hm = en_harmful[:, pos_idx, layer_idx, :].numpy()
    X_tg_hl = tg_harmless[:, pos_idx, layer_idx, :].numpy()
    X_tg_hm = tg_harmful[:, pos_idx, layer_idx, :].numpy()

    # Split target harmful prompts into refused vs. jailbroken (red vs. blue in Fig. 4)
    jb_mask = load_jailbreak_flags(target_eval_json, len(X_tg_hm))
    X_tg_hm_refuse = X_tg_hm[~jb_mask]
    X_tg_hm_jb = X_tg_hm[jb_mask]

    # Fit 2D PCA across combined representations
    all_X = np.vstack([X_en_hl, X_en_hm, X_tg_hl, X_tg_hm])
    pca = PCA(n_components=2, random_state=42)
    pca.fit(all_X)

    Z_en_hl = pca.transform(X_en_hl)
    Z_en_hm = pca.transform(X_en_hm)
    Z_tg_hl = pca.transform(X_tg_hl)
    Z_tg_hm_ref = pca.transform(X_tg_hm_refuse) if len(X_tg_hm_refuse) > 0 else np.empty((0, 2))
    Z_tg_hm_jb = pca.transform(X_tg_hm_jb) if len(X_tg_hm_jb) > 0 else np.empty((0, 2))

    # Plot scatter clusters matching Figure 4 color scheme
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(Z_en_hl[:, 0], Z_en_hl[:, 1], c="gray", alpha=0.65, s=18, label=f"harmless_{ref_lang}")
    ax.scatter(Z_en_hm[:, 0], Z_en_hm[:, 1], c="darkorange", alpha=0.65, s=18, label=f"harmful_refuse_{ref_lang}")
    ax.scatter(Z_tg_hl[:, 0], Z_tg_hl[:, 1], c="tab:green", alpha=0.65, s=18, label=f"harmless_{target_lang}")
    if len(Z_tg_hm_ref) > 0:
        ax.scatter(Z_tg_hm_ref[:, 0], Z_tg_hm_ref[:, 1], c="tab:blue", alpha=0.65, s=18, label=f"harmful_refuse_{target_lang}")
    if len(Z_tg_hm_jb) > 0:
        ax.scatter(Z_tg_hm_jb[:, 0], Z_tg_hm_jb[:, 1], c="tab:red", alpha=0.75, s=18, label=f"harmful_jailbreak_{target_lang}")

    # Overlay refusal direction arrows (mean_harmless -> mean_harmful)
    en_start, en_end = Z_en_hl.mean(axis=0), Z_en_hm.mean(axis=0)
    tg_start, tg_end = Z_tg_hl.mean(axis=0), pca.transform(X_tg_hm).mean(axis=0)

    ax.annotate(
        "", xy=en_end, xytext=en_start,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.8, ls="-"),
    )
    ax.annotate(
        "", xy=tg_end, xytext=tg_start,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.8, ls="--"),
    )

    ax.set_title(f"Tiny-Aya-Global: {ref_lang}-{target_lang} (Layer {layer_idx}, Pos {pos_idx})")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved PCA geometry plot -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="renessauce/tiny-aya-global-activations")
    parser.add_argument("--ref_lang", default="en")
    parser.add_argument("--target_lang", default="yo")
    parser.add_argument("--pos_idx", type=int, default=-1)
    parser.add_argument("--layer_idx", type=int, default=16)
    parser.add_argument("--target_eval_json", type=str, default="")
    parser.add_argument("--out_path", default="pca_en_yo.png")
    args = parser.parse_args()

    if os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"])

    plot_pca_pair(
        args.repo_id,
        args.ref_lang,
        args.target_lang,
        args.pos_idx,
        args.layer_idx,
        args.target_eval_json,
        args.out_path,
    )

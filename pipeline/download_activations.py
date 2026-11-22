import glob
import torch
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="renessauce/tiny-aya-global-activations",
    repo_type="dataset",
    allow_patterns="yo/activations/harmful_train/*.pt",
)

shard_files = sorted(glob.glob(f"{local_dir}/yo/activations/harmful_train/shard_*.pt"))
harmful_acts = torch.cat([torch.load(f)["activations"] for f in shard_files], dim=0)
# Shape: [total_prompts, eoi_len, num_layers, d_model]

import torch
import random
import json
import os
import argparse
import sys
import mmengine
import jsonpickle
import time

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import (
    get_activation_addition_input_pre_hook,
    get_all_direction_ablation_hooks,
    add_hooks,
)
from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import (
    select_direction,
    get_refusal_scores,
    randomly_select_direction,
)
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss
from pipeline.evaluator.evalharness import LMEvalHarness


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--config_path', type=str, required=True, help='Path to the config')
    parser.add_argument('--model_path', type=str, required=False, default=None, help='Path to the model')
    parser.add_argument('--batch_size', type=int, required=False, default=None, help='Batch size')
    return parser.parse_args()


def get_dataset_loaders(cfg):
    """
    Dynamically selects IndicSafe or PolyRefuse dataset loaders based on
    `cfg.dataset_source` or whether 'indicsafe' appears in the HF repo IDs.
    """
    source = getattr(cfg, "dataset_source", None)
    if source is None:
        repo_str = f"{getattr(cfg, 'hf_repo_id', '')} {getattr(cfg, 'hf_activations_repo_id', '')}".lower()
        source = "indicsafe" if "indicsafe" in repo_str else "polyrefuse"

    if source.lower() == "indicsafe":
        print("[Dataset Loader] Using IndicSafe (dataset.load_dataset_indicsafe)")
        from dataset.load_dataset_indicsafe import load_dataset_split, load_dataset
    else:
        print("[Dataset Loader] Using PolyRefuse (dataset.load_dataset)")
        from dataset.load_dataset import load_dataset_split, load_dataset

    return load_dataset_split, load_dataset


def init_hf():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is not set.")
    login(token=hf_token)
    return HfApi(token=hf_token)


@torch.no_grad()
def extract_and_checkpoint_split(
    cfg,
    model_base,
    prompts: list,
    split_name: str,
    accumulated_prompt_splits: dict,
    checkpoint_every: int = 50,
    upload_interval_sec: int = 360,  # Max ~5 uploads per 30 minutes (1 every 6 min)
):
    """
    Processes `prompts` in chunks of `checkpoint_every`:
      1. Saves activation shards locally every `checkpoint_every` prompts
      2. Uploads accumulated shards to HF Hub in a single batched commit at most once
         every `upload_interval_sec` seconds (and at the end of the split)
      3. Pushes the prompt DatasetDict once at the end of the split
    """
    api = init_hf()

    prompts_repo_id = getattr(cfg, "hf_repo_id", f"renessauce/tiny-aya-prompts-{cfg.source_lang}")
    acts_repo_id = getattr(cfg, "hf_activations_repo_id", f"renessauce/tiny-aya-activations-{cfg.source_lang}")

    api.create_repo(repo_id=acts_repo_id, repo_type="dataset", private=True, exist_ok=True)

    shard_dir = os.path.join(cfg.artifact_path, "activations", split_name)
    os.makedirs(shard_dir, exist_ok=True)

    num_layers = len(model_base.model_block_modules)
    eoi_len = len(model_base.eoi_toks)

    if split_name not in accumulated_prompt_splits:
        accumulated_prompt_splits[split_name] = []

    last_upload_time = time.time()

    for start_idx in range(0, len(prompts), checkpoint_every):
        end_idx = min(start_idx + checkpoint_every, len(prompts))
        chunk_prompts = prompts[start_idx:end_idx]

        chunk_prompt_strs = [
            p["instruction"] if isinstance(p, dict) else p
            for p in chunk_prompts
        ]

        chunk_layer_acts = {layer_idx: [] for layer_idx in range(num_layers)}

        def get_capture_hook(layer_idx):
            def hook_fn(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                eoi_act = hidden[:, -eoi_len:, :].detach().cpu()
                chunk_layer_acts[layer_idx].append(eoi_act)
            return hook_fn

        handles = [
            block.register_forward_hook(get_capture_hook(i))
            for i, block in enumerate(model_base.model_block_modules)
        ]

        try:
            sub_bs = min(cfg.batch_size, len(chunk_prompt_strs))
            for b_start in range(0, len(chunk_prompt_strs), sub_bs):
                batch_prompts = chunk_prompt_strs[b_start : b_start + sub_bs]
                inputs = model_base.tokenize_instructions_fn(instructions=batch_prompts)
                inputs = {k: v.to(model_base.model.device) for k, v in inputs.items()}
                model_base.model(**inputs)
        finally:
            for h in handles:
                h.remove()

        chunk_tensor = torch.stack(
            [torch.cat(chunk_layer_acts[i], dim=0) for i in range(num_layers)],
            dim=2
        )

        # 1. Save shard locally
        shard_filename = f"shard_{start_idx:05d}_{end_idx:05d}.pt"
        local_shard_path = os.path.join(shard_dir, shard_filename)
        torch.save(
            {
                "activations": chunk_tensor,
                "prompts": chunk_prompt_strs,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "lang": cfg.source_lang,
                "split": split_name,
            },
            local_shard_path,
        )

        for offset, (raw_item, p_str) in enumerate(zip(chunk_prompts, chunk_prompt_strs)):
            accumulated_prompt_splits[split_name].append({
                "index": start_idx + offset,
                "prompt": p_str,
                "category": raw_item.get("category", "") if isinstance(raw_item, dict) else "",
                "shard_file": shard_filename,
                "lang": cfg.source_lang,
                "split": split_name,
            })

        # 2. Upload to HF only if >= 6 minutes (360s) have elapsed or this is the final chunk
        is_last_chunk = (end_idx == len(prompts))
        elapsed = time.time() - last_upload_time

        if elapsed >= upload_interval_sec or is_last_chunk:
            api.upload_folder(
                folder_path=shard_dir,
                path_in_repo=f"{cfg.source_lang}/activations/{split_name}",
                repo_id=acts_repo_id,
                repo_type="dataset",
                commit_message=f"[{split_name}] Activation shards up to {end_idx}/{len(prompts)}",
            )
            last_upload_time = time.time()
            print(f"[HF Upload {end_idx}/{len(prompts)}] Synced {split_name} shards to Hub")
        else:
            print(f"[Local Checkpoint {end_idx}/{len(prompts)}] Saved {shard_filename} locally")

    # 3. Push prompt metadata once per split (1 commit per split instead of every 5 prompts)
    ds_dict = DatasetDict({
        s_name: Dataset.from_list(rows)
        for s_name, rows in accumulated_prompt_splits.items()
        if len(rows) > 0
    })
    ds_dict.push_to_hub(
        prompts_repo_id,
        private=True,
        commit_message=f"[{split_name}] Prompts complete ({len(prompts)} rows)",
    )
    print(f"[HF Prompts Synced] {split_name} ({len(prompts)} rows)")

def push_directions_and_evals_to_hf(cfg):
    """
    Uploads candidate directions (mean_diffs.pt), selected direction tensors,
    and evaluation JSONs to HF Hub so all plots (Figs 1-4) can be rendered offline.
    """
    api = init_hf()
    act_repo_id = getattr(cfg, "hf_activations_repo_id", f"renessauce/tiny-aya-activations-{cfg.source_lang}")
    api.create_repo(repo_id=act_repo_id, repo_type="dataset", private=True, exist_ok=True)

    for folder_name in ["generate_directions", "completions"]:
        local_dir = os.path.join(cfg.artifact_path, folder_name)
        if os.path.exists(local_dir):
            api.upload_folder(
                folder_path=local_dir,
                path_in_repo=f"{cfg.source_lang}/{folder_name}",
                repo_id=act_repo_id,
                repo_type="dataset",
                commit_message=f"Upload {folder_name} ({cfg.source_lang})",
            )

    for fname in ["direction_ablation.pt", "direction_metadata_ablation.json"]:
        fpath = os.path.join(cfg.artifact_path, fname)
        if os.path.exists(fpath):
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=f"{cfg.source_lang}/{fname}",
                repo_id=act_repo_id,
                repo_type="dataset",
                commit_message=f"Upload {fname} ({cfg.source_lang})",
            )

    print(f"Uploaded directions & evaluation JSONs to https://huggingface.co/datasets/{act_repo_id}")


class Logger:
    def __init__(self, log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(log_file, "a", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def load_and_sample_datasets(cfg, load_dataset_split_fn):
    """Load datasets and sample them based on the configuration (clamped to dataset size)."""
    random.seed(cfg.random_seed)
    d1_tr = load_dataset_split_fn(harmtype=cfg.harmtype_1, split='train', lang=cfg.source_lang, instructions_only=True)
    d2_tr = load_dataset_split_fn(harmtype=cfg.harmtype_2, split='train', lang=cfg.source_lang, instructions_only=True)
    d1_va = load_dataset_split_fn(harmtype=cfg.harmtype_1, split='val', lang=cfg.source_lang, instructions_only=True)
    d2_va = load_dataset_split_fn(harmtype=cfg.harmtype_2, split='val', lang=cfg.source_lang, instructions_only=True)

    data1_train = random.sample(d1_tr, min(len(d1_tr), cfg.n_train))
    data2_train = random.sample(d2_tr, min(len(d2_tr), cfg.n_train))
    data1_val = random.sample(d1_va, min(len(d1_va), cfg.n_val))
    data2_val = random.sample(d2_va, min(len(d2_va), cfg.n_val))
    return data1_train, data2_train, data1_val, data2_val

def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val, system_harm, system_or, system_harmless):
    """Filter datasets based on refusal scores."""
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    if cfg.filter_train:
        harmful_train_scores = get_refusal_scores(model_base.model, harmful_train, model_base.tokenize_instructions_fn, model_base.refusal_toks, system=system_harm, batch_size=cfg.batch_size)
        harmless_train_scores = get_refusal_scores(model_base.model, harmless_train, model_base.tokenize_instructions_fn, model_base.refusal_toks, system=system_harmless, batch_size=cfg.batch_size)
        harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
        harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)

    if cfg.filter_val:
        harmful_val_scores = get_refusal_scores(model_base.model, harmful_val, model_base.tokenize_instructions_fn, model_base.refusal_toks, system=system_harm, batch_size=cfg.batch_size)
        harmless_val_scores = get_refusal_scores(model_base.model, harmless_val, model_base.tokenize_instructions_fn, model_base.refusal_toks, system=system_harmless, batch_size=cfg.batch_size)
        harmful_val = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
        harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)

    return harmful_train, harmless_train, harmful_val, harmless_val


def generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train, system):
    """Generate and save candidate directions."""
    if not os.path.exists(os.path.join(cfg.artifact_path, 'generate_directions')):
        os.makedirs(os.path.join(cfg.artifact_path, 'generate_directions'))

    mean_diffs = generate_directions(
        system,
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=os.path.join(cfg.artifact_path, "generate_directions"),
        batch_size=cfg.batch_size,
    )

    torch.save(mean_diffs, os.path.join(cfg.artifact_path, 'generate_directions/mean_diffs.pt'))
    return mean_diffs


def select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions, pair_name, mode, kl_threshold, top_n):
    """Select and save the direction."""
    if not os.path.exists(os.path.join(cfg.artifact_path, 'select_direction')):
        os.makedirs(os.path.join(cfg.artifact_path, 'select_direction'))

    pos, layer, direction = select_direction(
        cfg,
        model_base,
        harmful_val,
        harmless_val,
        candidate_directions,
        pair_name,
        kl_threshold=kl_threshold,
        artifact_dir=os.path.join(cfg.artifact_path, "select_direction"),
        mode=mode,
        top_n=top_n,
        batch_size=cfg.batch_size,
    )

    with open(f'{cfg.artifact_path}/direction_metadata_{mode}.json', "w") as f:
        json.dump({"pos": pos, "layer": layer}, f, indent=4)

    torch.save(direction, f'{cfg.artifact_path}/direction_{mode}.pt')
    return pos, layer, direction


def generate_and_save_completions_for_dataset(
    cfg,
    model_base,
    fwd_pre_hooks,
    fwd_hooks,
    intervention_label,
    dataset_name,
    dataset=None,
    system=None,
    load_dataset_fn=None,
):
    """Generate and save completions for a dataset."""
    if not os.path.exists(os.path.join(cfg.artifact_path, 'completions')):
        os.makedirs(os.path.join(cfg.artifact_path, 'completions'))

    if dataset is None:
        if load_dataset_fn is None:
            _, load_dataset_fn = get_dataset_loaders(cfg)
        dataset = load_dataset_fn(dataset_name)

    # Ensure every item is a dict with 'instruction' for model_base.generate_completions
    dataset = [
        {"instruction": x, "category": "harmful"} if isinstance(x, str) else x
        for x in dataset
    ]

    completions = model_base.generate_completions(
        dataset,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
        max_new_tokens=cfg.max_new_tokens,
        batch_size=cfg.batch_size,
        system=system,
    )

    with open(f'{cfg.artifact_path}/completions/{dataset_name}_{intervention_label}_completions.json', "w") as f:
        json.dump(completions, f, indent=4)


def evaluate_completions_and_save_results_for_dataset(cfg, intervention_label, dataset_name, eval_methodologies):
    """Evaluate completions and save results for a dataset."""
    with open(os.path.join(cfg.artifact_path, f'completions/{dataset_name}_{intervention_label}_completions.json'), 'r') as f:
        completions = json.load(f)

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(cfg.artifact_path, "completions", f"{dataset_name}_{intervention_label}_evaluations.json"),
        cfg=cfg,
        logger=None,
    )

    with open(f'{cfg.artifact_path}/completions/{dataset_name}_{intervention_label}_evaluations.json', "w") as f:
        json.dump(evaluation, f, indent=4)


def evaluate_loss_for_datasets(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label):
    """Evaluate loss on datasets."""
    if not os.path.exists(os.path.join(cfg.artifact_path, 'loss_evals')):
        os.makedirs(os.path.join(cfg.artifact_path, 'loss_evals'))

    on_distribution_completions_file_path = os.path.join(cfg.artifact_path, 'completions/harmless_baseline_completions.json')
    loss_evals = evaluate_loss(model_base, fwd_pre_hooks, fwd_hooks, batch_size=cfg.ce_loss_batch_size, n_batches=cfg.ce_loss_n_batches, completions_file_path=on_distribution_completions_file_path)

    with open(f'{cfg.artifact_path}/loss_evals/{intervention_label}_loss_eval.json', "w") as f:
        json.dump(loss_evals, f, indent=4)


def eval_harness(cfg, model_base, identifier):
    eval_harness_evaluator_mmlu = LMEvalHarness(cfg.eval_harness_mmlu)
    lm_eval_results_mmlu = eval_harness_evaluator_mmlu.evaluate(model=model_base)
    if not os.path.exists(f'{cfg.artifact_path}/lm_eval_results'):
        os.makedirs(f'{cfg.artifact_path}/lm_eval_results')
    with open(f'{cfg.artifact_path}/lm_eval_results/{identifier}_mmlu.json', "w") as f:
        f.write(jsonpickle.encode(lm_eval_results_mmlu, indent=4))

    if 'eval_harness' in cfg:
        eval_harness_evaluator = LMEvalHarness(cfg.eval_harness)
        lm_eval_results = eval_harness_evaluator.evaluate(model=model_base)
        if not os.path.exists(f'{cfg.artifact_path}/lm_eval_results'):
            os.makedirs(f'{cfg.artifact_path}/lm_eval_results')
        with open(f'{cfg.artifact_path}/lm_eval_results/{identifier}.json', "w") as f:
            f.write(jsonpickle.encode(lm_eval_results, indent=4))

    return lm_eval_results_mmlu


def ortho_refusal_directions(cfg, candidate_directions_harm_contrast, candidate_directions_or_contrast):
    epsilon = 1e-8
    dp = torch.sum(candidate_directions_harm_contrast * candidate_directions_or_contrast, dim=-1)
    FUD_norm_sq = torch.sum(candidate_directions_or_contrast * candidate_directions_or_contrast, dim=-1)
    s = dp / (FUD_norm_sq + epsilon)
    s_expanded = s.unsqueeze(-1)
    proj = s_expanded * candidate_directions_or_contrast

    lambda_value = cfg.ortho_lambda
    candidate_directions_harm_contrast_orth = candidate_directions_harm_contrast - lambda_value * proj
    harm_contrast_orth_norm = torch.sqrt(torch.sum(candidate_directions_harm_contrast_orth * candidate_directions_harm_contrast_orth, dim=-1))
    candidate_directions_harm_contrast_orth_normalized = candidate_directions_harm_contrast_orth / (harm_contrast_orth_norm.unsqueeze(-1) + epsilon)

    return candidate_directions_harm_contrast_orth_normalized


def run_pipeline(config_path, model_path, batch_size):
    """Run the full pipeline."""
    cfg = mmengine.Config.fromfile(config_path)
    if model_path is not None:
        cfg.model_path = model_path
    model_alias = os.path.basename(cfg.model_path)
    cfg.model_alias = model_alias

    if batch_size is not None:
        cfg.batch_size = batch_size

    if 'artifact_path' not in cfg:
        cfg.artifact_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", cfg.model_alias)

    sys.stdout = Logger(f"{cfg.artifact_path}/output.log")
    sys.stderr = Logger(f"{cfg.artifact_path}/error.log")
    cfg.dump(f'{cfg.artifact_path}/config_run.yaml')

    # Dynamically select IndicSafe or PolyRefuse dataset loaders
    load_dataset_split, load_dataset = get_dataset_loaders(cfg)

    model_base = construct_model_base(cfg.model_path, cfg.source_lang)

    # Load, sample, and filter datasets
    harmless_train, harmful_train, harmless_val, harmful_val = load_and_sample_datasets(cfg, load_dataset_split)
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(
        cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val,
        system_harm=None, system_or=None, system_harmless=None
    )


    test_pool = load_dataset_split(harmtype='harmful', split='test', lang=cfg.source_lang)
    harmful_data = random.sample(test_pool, min(len(test_pool), cfg.n_test))

    # 1. Checkpointed extraction + HF upload every 5 prompts for both prompts and activations
    accumulated_prompt_splits = {}
    splits_to_process = {
        "harmful_train": harmful_train,
        "harmless_train": harmless_train,
        "harmful_val": harmful_val,
        "harmless_val": harmless_val,
        "harmful_test": harmful_data,
    }

    for split_name, split_prompts in splits_to_process.items():
        extract_and_checkpoint_split(
            cfg=cfg,
            model_base=model_base,
            prompts=split_prompts,
            split_name=split_name,
            accumulated_prompt_splits=accumulated_prompt_splits,
            checkpoint_every=50,
            upload_interval_sec=360
        )

    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []

    if not cfg.baseline:
        candidate_directions_harm_contrast = generate_and_save_candidate_directions(
            cfg, model_base, harmful_train, harmless_train, system=None
        )[:, cfg.start_layer:,]
        pair_name = ['harmful', 'harmless']

        positions_harm, layers_harm, directions_harm = select_and_save_direction(
            cfg, model_base, harmful_val, harmless_val, candidate_directions_harm_contrast,
            pair_name, 'ablation', cfg.ablate_kl_threshold, cfg.top_n
        )

        harm_actadd_fwd_pre_hooks, harm_actadd_fwd_hooks = [], []
        harm_ablation_fwd_pre_hooks, harm_ablation_fwd_hooks = [], []

        for pos_harm, layer_harm, direction_harm in zip(positions_harm, layers_harm, directions_harm):
            harm_actadd_fwd_pre_hooks.append(
                (model_base.model_block_modules[layer_harm], get_activation_addition_input_pre_hook(vector=direction_harm, coeff=+cfg.addact_coeff))
            )
            pre_hooks, fwd_hooks = get_all_direction_ablation_hooks(model_base, direction_harm, cfg.start_layer, cfg.ablation_coeff)
            harm_ablation_fwd_pre_hooks.extend(pre_hooks)
            harm_ablation_fwd_hooks.extend(fwd_hooks)

        # Standard single-language evaluations
        generate_and_save_completions_for_dataset(cfg, model_base, harm_ablation_fwd_pre_hooks, harm_ablation_fwd_hooks, 'harm_ablation', 'harmful', dataset=harmful_data, system=cfg.system, load_dataset_fn=load_dataset)
        generate_and_save_completions_for_dataset(cfg, model_base, harm_ablation_fwd_pre_hooks, harm_ablation_fwd_hooks, 'harm_ablation', 'xstest', system=cfg.system, load_dataset_fn=load_dataset)
        generate_and_save_completions_for_dataset(cfg, model_base, harm_ablation_fwd_pre_hooks, harm_ablation_fwd_hooks, 'harm_ablation', 'jailbreakbench', system=cfg.system, load_dataset_fn=load_dataset)

        for dataset_name in cfg.jailbreak_evaluation_datasets:
            evaluate_completions_and_save_results_for_dataset(cfg, 'harm_ablation', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)

        # Cross-lingual evaluations across eval_langs (Baseline, Ablation, and Addition for Figs 1-3)
        eval_langs = getattr(cfg, "eval_langs", ["en", "yo"])
        for eval_lang in eval_langs:

            lang_pool = load_dataset_split(harmtype="harmful", split="test", lang=eval_lang)
            lang_test_data = random.sample(lang_pool, min(len(lang_pool), cfg.n_test))

            # Baseline
            generate_and_save_completions_for_dataset(
                cfg, model_base, [], [], f"baseline_{eval_lang}", "harmful",
                dataset=lang_test_data, system=cfg.system, load_dataset_fn=load_dataset
            )
            evaluate_completions_and_save_results_for_dataset(
                cfg, f"baseline_{eval_lang}", "harmful",
                eval_methodologies=cfg.jailbreak_eval_methodologies
            )

            # Direction Ablation
            generate_and_save_completions_for_dataset(
                cfg, model_base, harm_ablation_fwd_pre_hooks, harm_ablation_fwd_hooks,
                f"harm_ablation_{cfg.source_lang}_on_{eval_lang}", "harmful",
                dataset=lang_test_data, system=cfg.system, load_dataset_fn=load_dataset
            )
            evaluate_completions_and_save_results_for_dataset(
                cfg, f"harm_ablation_{cfg.source_lang}_on_{eval_lang}", "harmful",
                eval_methodologies=cfg.jailbreak_eval_methodologies
            )

            # Activation Addition
            generate_and_save_completions_for_dataset(
                cfg, model_base, harm_actadd_fwd_pre_hooks, harm_actadd_fwd_hooks,
                f"harm_actadd_{cfg.source_lang}_on_{eval_lang}", "harmful",
                dataset=lang_test_data, system=cfg.system, load_dataset_fn=load_dataset
            )
            evaluate_completions_and_save_results_for_dataset(
                cfg, f"harm_actadd_{cfg.source_lang}_on_{eval_lang}", "harmful",
                eval_methodologies=cfg.jailbreak_eval_methodologies
            )

        if getattr(cfg, "run_eval_harness", False):
            with add_hooks(module_forward_pre_hooks=harm_ablation_fwd_pre_hooks, module_forward_hooks=harm_ablation_fwd_hooks):
                eval_harness(cfg, model_base, 'harm_ablation')

    else:
        generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', 'harmful', dataset=harmful_data, system=cfg.system, load_dataset_fn=load_dataset)
        generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', 'xstest', system=cfg.system, load_dataset_fn=load_dataset)
        generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', 'jailbreakbench', system=cfg.system, load_dataset_fn=load_dataset)
        for dataset_name in cfg.jailbreak_evaluation_datasets:
            evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)

    # Push final directions and all evaluation JSONs to HF Hub
    push_directions_and_evals_to_hf(cfg)


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(config_path=args.config_path, model_path=args.model_path, batch_size=args.batch_size)

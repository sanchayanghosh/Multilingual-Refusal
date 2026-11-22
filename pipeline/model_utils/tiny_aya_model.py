import torch
import functools
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch import Tensor
from jaxtyping import Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

# Starter refusal words per language (dynamically encoded using the Aya tokenizer)
REFUSAL_PREFIX_LANG = {
    'en': 'I',
    'zh': '我',
    'de': 'Ich',
    'th': 'ฉ',
    'yi': 'איך',
    'yo': 'Mo',
    'ru': 'Я',
    'ko': '죄송',
    'ja': '申し訳',
}

def format_instruction_tiny_aya_chat(
    tokenizer: AutoTokenizer,
    instruction: str,
    output: str = None,
    system: str = None,
    include_trailing_whitespace: bool = True
):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": instruction})

    formatted_instruction = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction


def tokenize_instructions_tiny_aya_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    system: str = None,
    include_trailing_whitespace: bool = True
):
    if outputs is not None:
        prompts = [
            format_instruction_tiny_aya_chat(
                tokenizer=tokenizer,
                instruction=instruction,
                output=output,
                system=system,
                include_trailing_whitespace=include_trailing_whitespace
            )
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_tiny_aya_chat(
                tokenizer=tokenizer,
                instruction=instruction,
                system=system,
                include_trailing_whitespace=include_trailing_whitespace
            )
            for instruction in instructions
        ]

    return tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )


def orthogonalize_tiny_aya_weights(model, direction: Float[Tensor, "d_model"]):
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(
        model.model.embed_tokens.weight.data, direction
    )
    for block in model.model.layers:
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(
            block.self_attn.o_proj.weight.data.T, direction
        ).T
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(
            block.mlp.down_proj.weight.data.T, direction
        ).T


def act_add_tiny_aya_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    dtype = model.model.layers[layer - 1].mlp.down_proj.weight.dtype
    device = model.model.layers[layer - 1].mlp.down_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)
    model.model.layers[layer - 1].mlp.down_proj.bias = torch.nn.Parameter(bias)


class TinyAyaModel(ModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_tiny_aya_chat,
            tokenizer=self.tokenizer,
            system=None,
            include_trailing_whitespace=True
        )

    def _get_eoi_toks(self):
        # Extract the exact end-of-instruction / assistant header tokens from the template
        sample = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "{instruction}"}],
            tokenize=False,
            add_generation_prompt=True
        )
        eoi_str = sample.split("{instruction}")[-1]
        return self.tokenizer.encode(eoi_str, add_special_tokens=False)

    def _get_refusal_toks(self, lang):
        prefix = REFUSAL_PREFIX_LANG.get(lang, 'I')
        toks = self.tokenizer.encode(prefix, add_special_tokens=False)
        return [toks[0]]

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block.self_attn for block in self.model_block_modules])

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block.mlp for block in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_tiny_aya_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_tiny_aya_weights, direction=direction, coeff=coeff, layer=layer)

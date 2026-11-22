import random
from datasets import load_dataset as hf_load_dataset
from dataset.load_dataset import load_dataset  # re-export for xstest / jailbreakbench

LANG_NAME_MAP = {
    "en": "english",
    "hi": "hindi",
    "bn": "bengali",
    "ta": "tamil",
    "te": "telugu",
    "mr": "marathi",
    "gu": "gujarati",
    "kn": "kannada",
    "ml": "malayalam",
    "pa": "punjabi",
    "or": "odia",
    "ur": "urdu",
    "ne": "nepali",
}

INDIC_ALIGN_LANG_COL = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "pa": "pan_Guru",
    "or": "ory_Orya",
    "ur": "urd_Arab",
    "ne": "npi_Deva",
}

def load_dataset_split(harmtype: str, split: str, lang: str = "en", instructions_only: bool = True):
    """
    Loads parallel harmful prompts from `ppattnay/IndicSafe` and parallel
    harmless prompts from `ai4bharat/indic-align` ('Dolly_T' config).
    """
    if harmtype == "harmless":
        lang_col = INDIC_ALIGN_LANG_COL.get(lang.lower(), "eng_Latn")
        benign_ds = hf_load_dataset("ai4bharat/indic-align", "Dolly_T", split="train[:1500]")

        prompts = []
        for row in benign_ds:
            turns = row.get(lang_col, [])
            text = turns[0][0] if turns and isinstance(turns[0], list) and len(turns[0]) > 0 else ""
            if text:
                prompts.append(text if instructions_only else {"instruction": text, "category": "harmless"})
    else:
        ds = hf_load_dataset("ppattnay/IndicSafe", split="benchmark")
        target_lang = LANG_NAME_MAP.get(lang.lower(), lang.lower())

        prompts = []
        seen_en = set()

        for row in ds:
            category = str(row.get("Category") or row.get("category") or "Harmful Instructions")

            if target_lang == "english":
                text = row.get("Prompt")
                if not text or text in seen_en:
                    continue
                seen_en.add(text)
            else:
                row_lang = str(row.get("Language") or "").strip().lower()
                if row_lang != target_lang:
                    continue
                text = row.get("Translated_Prompt")
                if not text:
                    continue

            prompts.append(text if instructions_only else {"instruction": text, "category": category})

    rng = random.Random(42)
    prompts = list(prompts)
    rng.shuffle(prompts)

    n = len(prompts)
    n_train = int(0.7 * n)
    n_val = int(0.1 * n)

    if split == "train":
        return prompts[:n_train]
    elif split == "val":
        return prompts[n_train : n_train + n_val]
    elif split == "test":
        return prompts[n_train + n_val :]
    else:
        raise ValueError(f"Unknown split: {split}")

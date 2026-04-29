"""
titanbit.alignment.data
~~~~~~~~~~~~~~~~~~~~~~~
Preference dataset handling for DPO/KTO alignment.

Supports standard preference formats:
    - Anthropic HH-RLHF (chosen/rejected pairs)
    - UltraFeedback (multi-aspect scores)
    - Custom JSONL with {prompt, chosen, rejected} fields

The key insight: for ternary models, we need to be careful about
how we compute log-probabilities.  The quantised forward pass
introduces small numerical differences that can destabilise the
DPO loss if not handled correctly.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


@dataclass
class PreferenceExample:
    """A single preference pair."""
    prompt: str
    chosen: str        # preferred response
    rejected: str      # dispreferred response
    metadata: dict[str, Any] | None = None


class PreferenceDataset(Dataset):
    """
    Dataset of preference pairs for DPO training.

    Each item returns tokenised tensors for:
        - prompt + chosen  (the "winning" completion)
        - prompt + rejected  (the "losing" completion)

    We tokenise on-the-fly to support different tokenizers
    and sequence lengths without pre-processing.
    """

    def __init__(
        self,
        examples: list[PreferenceExample],
        tokenizer: Any,
        max_length: int = 1024,
        max_prompt_length: int = 512,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

        logger.info(
            "PreferenceDataset: %d pairs, max_length=%d",
            len(examples), max_length,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]

        # Tokenise prompt
        prompt_ids = self._tokenise(ex.prompt)
        if len(prompt_ids) > self.max_prompt_length:
            prompt_ids = prompt_ids[:self.max_prompt_length]

        # Tokenise chosen and rejected completions
        chosen_ids = self._tokenise(ex.chosen)
        rejected_ids = self._tokenise(ex.rejected)

        # Build full sequences: prompt + completion
        chosen_input = self._build_sequence(prompt_ids, chosen_ids)
        rejected_input = self._build_sequence(prompt_ids, rejected_ids)

        prompt_len = len(prompt_ids)

        return {
            "chosen_input_ids": chosen_input["input_ids"],
            "chosen_attention_mask": chosen_input["attention_mask"],
            "chosen_labels": chosen_input["labels"],
            "rejected_input_ids": rejected_input["input_ids"],
            "rejected_attention_mask": rejected_input["attention_mask"],
            "rejected_labels": rejected_input["labels"],
            "prompt_length": torch.tensor(prompt_len, dtype=torch.long),
        }

    def _tokenise(self, text: str) -> list[int]:
        """Tokenise text using the provided tokenizer."""
        if hasattr(self.tokenizer, "encode_ordinary"):
            # tiktoken
            return self.tokenizer.encode_ordinary(text)
        elif hasattr(self.tokenizer, "encode"):
            # HuggingFace tokenizer
            return self.tokenizer.encode(text, add_special_tokens=False)
        else:
            raise ValueError(f"Unsupported tokenizer type: {type(self.tokenizer)}")

    def _build_sequence(
        self,
        prompt_ids: list[int],
        completion_ids: list[int],
    ) -> dict[str, torch.Tensor]:
        """Build a padded sequence with labels masked for the prompt."""
        full_ids = prompt_ids + completion_ids
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]

        seq_len = len(full_ids)
        pad_len = self.max_length - seq_len

        # Input IDs (padded with 0)
        input_ids = torch.tensor(full_ids + [0] * pad_len, dtype=torch.long)

        # Attention mask
        attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        attention_mask[:seq_len] = 1

        # Labels: mask the prompt portion with -100
        labels = torch.full((self.max_length,), -100, dtype=torch.long)
        prompt_len = len(prompt_ids)
        # Only compute loss on the completion tokens
        labels[prompt_len:seq_len] = input_ids[prompt_len:seq_len]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_preference_data(
    path: str,
    format: str = "auto",
    max_examples: int | None = None,
) -> list[PreferenceExample]:
    """
    Load preference data from various formats.

    Supported formats:
        - "jsonl": One JSON object per line with {prompt, chosen, rejected}
        - "hh": Anthropic HH-RLHF format
        - "auto": Auto-detect from file extension / content

    Parameters
    ----------
    path        : path to data file
    format      : data format
    max_examples: maximum number of examples to load

    Returns
    -------
    List of PreferenceExample objects
    """
    if format == "auto":
        if path.endswith(".jsonl"):
            format = "jsonl"
        elif path.endswith(".json"):
            format = "json"
        else:
            format = "jsonl"

    examples: list[PreferenceExample] = []

    if format == "jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_examples and i >= max_examples:
                    break
                obj = json.loads(line.strip())
                examples.append(PreferenceExample(
                    prompt=obj["prompt"],
                    chosen=obj["chosen"],
                    rejected=obj["rejected"],
                    metadata=obj.get("metadata"),
                ))

    elif format == "json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for i, obj in enumerate(data):
            if max_examples and i >= max_examples:
                break
            examples.append(PreferenceExample(
                prompt=obj["prompt"],
                chosen=obj["chosen"],
                rejected=obj["rejected"],
                metadata=obj.get("metadata"),
            ))

    elif format == "hh":
        # Anthropic HH-RLHF format: {chosen: "Human: ...\n\nAssistant: ...", rejected: ...}
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_examples and i >= max_examples:
                    break
                obj = json.loads(line.strip())
                chosen_text = obj["chosen"]
                rejected_text = obj["rejected"]

                # Extract prompt (everything up to last "Assistant:" in chosen)
                parts = chosen_text.rsplit("\n\nAssistant:", 1)
                if len(parts) == 2:
                    prompt = parts[0] + "\n\nAssistant:"
                    chosen_resp = parts[1].strip()
                else:
                    prompt = ""
                    chosen_resp = chosen_text

                # Extract rejected response
                rej_parts = rejected_text.rsplit("\n\nAssistant:", 1)
                rejected_resp = rej_parts[1].strip() if len(rej_parts) == 2 else rejected_text

                examples.append(PreferenceExample(
                    prompt=prompt,
                    chosen=chosen_resp,
                    rejected=rejected_resp,
                ))

    logger.info("Loaded %d preference examples from %s", len(examples), path)
    return examples


def create_preference_dataloader(
    examples: list[PreferenceExample],
    tokenizer: Any,
    batch_size: int = 4,
    max_length: int = 1024,
    max_prompt_length: int = 512,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for preference training."""
    dataset = PreferenceDataset(
        examples=examples,
        tokenizer=tokenizer,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

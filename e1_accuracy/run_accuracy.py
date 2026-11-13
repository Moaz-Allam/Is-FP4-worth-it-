"""Perplexity (WikiText-2 and C4), drift from the unquantized model, and weight error under each weight format.

Usage: python run_accuracy.py <experiment: a key of quant.EXPERIMENTS> <model_name> <window_tokens>
Appends one JSON line per format to ../results/<experiment>.jsonl.
"""

import gc
import json
import math
import pathlib
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D

import quant

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
OUTPUT_HEAD_NAME = "lm_head"
C4_VALIDATION_SHARD = "en/c4-validation.00000-of-00008.json.gz"
C4_DOCUMENTS = 1100
C4_WINDOWS = 256
REFERENCE_WINDOWS = 32
TOKENS_PER_DIVERGENCE_CHUNK = 256
BYTES_PER_HALF_PRECISION_WEIGHT = 2
ACTIVATION_HEADROOM_BYTES = 4 * 1024 ** 3
NO_DEVICE = ""


def main(experiment: str, model_name: str, window_tokens: int) -> None:
    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    wikitext2 = wikitext2_test_token_ids(tokenizer)
    c4 = c4_validation_token_ids(tokenizer, window_tokens)
    reference = load_reference(model_name, device)
    for weight_format in quant.EXPERIMENTS[experiment]:
        started = time.time()
        model, weight_error = load_quantized_model(model_name, weight_format, device)
        record = {
            **format_description(model_name, weight_format),
            **perplexity_summary(window_losses(model, wikitext2, window_tokens, device), prefix=""),
            **perplexity_summary(window_losses(model, c4, window_tokens, device), prefix="c4_"),
            **reference.drift_of(model, wikitext2[:REFERENCE_WINDOWS * window_tokens], window_tokens, device),
            **weight_error,
            "window_tokens": window_tokens,
            "device": device,
            "evaluation_dtype": str(model.dtype),
            "seconds": round(time.time() - started, 1),
        }
        append_record(record, RESULTS_DIR / f"{experiment}.jsonl")
        print(json.dumps({key: value for key, value in record.items() if not isinstance(value, list)}), flush=True)
        del model
        free_device_memory()


def free_device_memory() -> None:
    """Call after dropping the last reference to a model, so the next model fits on the GPU."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def wikitext2_test_token_ids(tokenizer) -> torch.Tensor:
    test_split = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return tokenizer("\n\n".join(test_split["text"]), return_tensors="pt").input_ids[0]


def c4_validation_token_ids(tokenizer, window_tokens: int) -> torch.Tensor:
    """The C4 validation subset used by GPTQ: the first 1100 documents, cut to 256 windows."""
    validation = load_dataset("allenai/c4", data_files={"validation": C4_VALIDATION_SHARD}, split="validation")
    text = " ".join(validation[:C4_DOCUMENTS]["text"])
    return tokenizer(text, return_tensors="pt").input_ids[0][:C4_WINDOWS * window_tokens]


def load_quantized_model(model_name: str, weight_format, device: str):
    """Quantize in 32-bit floats; on a GPU, run the quantized model in 16-bit floats."""
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    weight_error = quantize_linear_weights(model, weight_format)
    return on_device(model, device), weight_error


def on_device(model, device: str):
    """On a GPU the model runs in 16-bit floats, converted before the move so it never needs the 32-bit size there."""
    if device.startswith("cuda"):
        model.half()
    return model.to(device).eval()


def format_description(model_name: str, weight_format) -> dict:
    return {
        "model": model_name,
        "format": weight_format.name,
        "block_size": weight_format.block_size,
        "bits_per_weight": weight_format.bits_per_weight,
    }


def load_reference(model_name: str, device: str):
    """The unquantized model, kept in memory next to each quantized model when a device has room for it."""
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    reference_device = device_with_room_for_reference(model, device)
    if reference_device == NO_DEVICE:
        return NoReference()
    return Reference(on_device(model, reference_device))


def device_with_room_for_reference(model, device: str) -> str:
    if not device.startswith("cuda"):
        return device
    if torch.cuda.device_count() > 1:
        return "cuda:1"
    model_bytes = sum(parameter.numel() for parameter in model.parameters()) * BYTES_PER_HALF_PRECISION_WEIGHT
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    return device if 2 * model_bytes + ACTIVATION_HEADROOM_BYTES < total_bytes else NO_DEVICE


class Reference:
    """Measures how far a quantized model's next-token predictions drift from the unquantized model's."""

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def drift_of(self, model, token_ids, window_tokens: int, device: str) -> dict:
        divergences, agreements = [], []
        for window in windows_of(token_ids, window_tokens):
            logits = model(window.to(device)).logits[0]
            reference_logits = self.model(window.to(self.model.device)).logits[0].to(device)
            divergence, agreement = divergence_and_agreement(reference_logits, logits)
            divergences.append(divergence)
            agreements.append(agreement)
        return {
            "kl_divergence": sum(divergences) / len(divergences),
            "top1_agreement": sum(agreements) / len(agreements),
            "window_kl_divergences": divergences,
            "window_top1_agreements": agreements,
        }


class NoReference:
    """Stands in when no device can hold a second copy of the model: no drift is reported."""

    def drift_of(self, model, token_ids, window_tokens: int, device: str) -> dict:
        return {}


def divergence_and_agreement(reference_logits: torch.Tensor, logits: torch.Tensor):
    """Mean Kullback-Leibler divergence (reference to quantized) and top-1 agreement over one window."""
    divergence_sum, agreeing_tokens = 0.0, 0
    chunks = zip(reference_logits.split(TOKENS_PER_DIVERGENCE_CHUNK), logits.split(TOKENS_PER_DIVERGENCE_CHUNK))
    for reference_chunk, chunk in chunks:
        reference_log_probabilities = torch.log_softmax(reference_chunk.float(), dim=-1)
        log_probabilities = torch.log_softmax(chunk.float(), dim=-1)
        divergence_sum += torch.sum(reference_log_probabilities.exp()
                                    * (reference_log_probabilities - log_probabilities)).item()
        agreeing_tokens += int((reference_chunk.argmax(dim=-1) == chunk.argmax(dim=-1)).sum())
    return divergence_sum / len(logits), agreeing_tokens / len(logits)


@torch.no_grad()
def quantize_linear_weights(model, weight_format) -> dict:
    """Replace every linear weight except the output head; return the error over all of them."""
    error = WeightErrorTotals()
    for name, module in model.named_modules():
        if not is_quantized_layer(name, module):
            continue
        rows_are_outputs = as_outputs_by_inputs(module)
        dequantized, clamped = quant.quantize_matrix(rows_are_outputs, weight_format)
        error.add(rows_are_outputs, dequantized, clamped)
        module.weight.copy_(dequantized.t() if isinstance(module, Conv1D) else dequantized)
    return error.summary()


def is_quantized_layer(name: str, module) -> bool:
    return isinstance(module, (torch.nn.Linear, Conv1D)) and OUTPUT_HEAD_NAME not in name


def as_outputs_by_inputs(module) -> torch.Tensor:
    """GPT-2's Conv1D stores (inputs, outputs); nn.Linear stores (outputs, inputs)."""
    return module.weight.t() if isinstance(module, Conv1D) else module.weight


class WeightErrorTotals:
    def __init__(self):
        self.squared_error = 0.0
        self.squared_signal = 0.0
        self.clamped = 0
        self.count = 0

    def add(self, original, dequantized, clamped) -> None:
        self.squared_error += torch.sum((dequantized.double() - original.double()) ** 2).item()
        self.squared_signal += torch.sum(original.double() ** 2).item()
        self.clamped += int(clamped.sum())
        self.count += original.numel()

    def summary(self) -> dict:
        return {
            "weight_mse": self.squared_error / self.count,
            "sqnr_db": 10 * math.log10(self.squared_signal / self.squared_error) if self.squared_error else math.inf,
            "clamped_fraction": self.clamped / self.count,
            "quantized_weights": self.count,
        }


def windows_of(token_ids: torch.Tensor, window_tokens: int):
    """Non-overlapping windows as batches of one; a final window too short to predict anything is dropped."""
    for start in range(0, len(token_ids), window_tokens):
        window = token_ids[start:start + window_tokens].unsqueeze(0)
        if window.shape[1] >= 2:
            yield window


@torch.no_grad()
def window_losses(model, token_ids, window_tokens, device) -> list:
    """Mean negative log-likelihood and predicted-token count of each window."""
    return [(model(window.to(device), labels=window.to(device)).loss.float().item(), window.shape[1] - 1)
            for window in windows_of(token_ids, window_tokens)]


def perplexity_summary(losses: list, prefix: str) -> dict:
    """Perplexity over every predicted token; its standard error comes from the spread across windows."""
    predicted_tokens = sum(tokens for _, tokens in losses)
    mean_loss = sum(loss * tokens for loss, tokens in losses) / predicted_tokens
    variance = sum((loss - mean_loss) ** 2 for loss, _ in losses) / (len(losses) - 1)
    perplexity = math.exp(mean_loss)
    return {
        f"{prefix}perplexity": perplexity,
        f"{prefix}perplexity_standard_error": perplexity * math.sqrt(variance / len(losses)),
        f"{prefix}window_losses": [loss for loss, _ in losses],
    }


def append_record(record: dict, results_path: pathlib.Path) -> None:
    results_path.parent.mkdir(exist_ok=True)
    with results_path.open("a") as results_file:
        results_file.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))

"""
generate.py — Code generation wrapper for next-line prediction.

Wraps any HuggingFace causal LM (Qwen2.5-Coder, DeepSeek-Coder, etc.) for
the RepoBench next-line prediction task.

Two entry points:
- ``generate_next_line``       — single prompt, used by EFL
- ``generate_next_line_batch`` — multiple prompts in one GPU call, ~6× faster
                                 for bulk evaluation loops

Usage in notebooks::

    from haluguard.generate import build_completion_prompt, generate_next_line_batch

    prompts = [build_completion_prompt(...) for ex in batch]
    predictions = generate_next_line_batch(prompts, tokenizer, model, device="cuda")
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch


def _set_padding_side(tokenizer: Any, side: str) -> Optional[str]:
    """Set tokenizer padding side and return the previous value."""
    prev = getattr(tokenizer, "padding_side", None)
    if prev is not None:
        tokenizer.padding_side = side
    return prev


def _restore_padding_side(tokenizer: Any, prev: Optional[str]) -> None:
    if prev is not None:
        tokenizer.padding_side = prev


def _tokenize_prompt(
    prompt: str,
    tokenizer: Any,
    max_prompt_tokens: int,
) -> Any:
    """Tokenize a prompt while preserving the most recent code at the end."""
    original_side = getattr(tokenizer, "truncation_side", None)
    if original_side is not None:
        tokenizer.truncation_side = "left"

    try:
        return tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_tokens,
        )
    finally:
        if original_side is not None:
            tokenizer.truncation_side = original_side


def build_completion_prompt(
    cropped_code: str,
    import_statement: str,
    selected_snippets: List[str],
    previous_error: Optional[str] = None,
) -> str:
    """Assemble a prompt for next-line code completion.

    Prepends cross-file context snippets to the current file's code so the
    LLM can reference definitions, signatures, and imports from other files.
    When ``previous_error`` is provided (EFL retry path) the error is injected
    as a comment so the model can self-correct.

    Args:
        cropped_code:      Code written so far in the current file (the query).
        import_statement:  Import statements from the current file.
        selected_snippets: Code snippets from other files, selected by HCCS
                           or a baseline method.
        previous_error:    Error string from a prior EFL attempt, e.g.
                           ``"ImportError: No module named 'foo'"``.

    Returns:
        Formatted prompt string ready for a causal LM.
    """
    parts: List[str] = []

    for snippet in selected_snippets:
        parts.append(f"# Cross-file context:\n{snippet}")

    if previous_error:
        parts.append(f"# Previous attempt failed with:\n# {previous_error}")

    if import_statement.strip():
        parts.append(import_statement)

    parts.append(cropped_code)

    return "\n\n".join(parts)


@torch.no_grad()
def generate_next_line(
    prompt: str,
    tokenizer: Any,
    model: Any,
    device: str = "cuda",
    max_new_tokens: int = 64,
    temperature: float = 0.2,
    max_prompt_tokens: int = 2048,
) -> str:
    """Generate the next line of code using a causal language model.

    Tokenises the prompt, generates up to ``max_new_tokens`` tokens, and
    returns only the first line of the generated text (stripping the prompt).

    Args:
        prompt:           Full prompt string from ``build_completion_prompt``.
        tokenizer:        HuggingFace tokenizer for the generator model.
        model:            HuggingFace causal LM (e.g. DeepSeek-Coder-1.3B).
        device:           Torch device string.  Default ``"cuda"``.
        max_new_tokens:   Maximum tokens to generate.  Default 64.
        temperature:      Sampling temperature.  Default 0.2.
        max_prompt_tokens: Truncate prompt to this many tokens.  Default 2048.

    Returns:
        The predicted next line of code (single line, stripped).
    """
    inputs = _tokenize_prompt(prompt, tokenizer, max_prompt_tokens).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode only the generated tokens (strip the prompt)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Return only the first line
    first_line = generated_text.split("\n")[0]
    return first_line.rstrip()


@torch.no_grad()
def generate_next_line_batch(
    prompts: List[str],
    tokenizer: Any,
    model: Any,
    device: str = "cuda",
    max_new_tokens: int = 64,
    temperature: float = 0.2,
    max_prompt_tokens: int = 2048,
) -> List[str]:
    """Generate the next line for a batch of prompts in a single GPU call.

    Uses left-padding so every prompt ends at the same position, which is
    required for correct causal-LM generation with padding.  ~6× faster than
    calling ``generate_next_line`` in a loop.

    Args:
        prompts:          List of prompt strings from ``build_completion_prompt``.
        tokenizer:        HuggingFace tokenizer (must support padding).
        model:            HuggingFace causal LM.
        device:           Torch device string.  Default ``"cuda"``.
        max_new_tokens:   Maximum new tokens per prompt.  Default 64.
        temperature:      Sampling temperature.  Default 0.2.
        max_prompt_tokens: Truncate each prompt to this many tokens.  Default 2048.

    Returns:
        List of predicted next lines, one per prompt (same order, single line each).
    """
    if not prompts:
        return []

    # Causal LMs need left-padding so the actual code sits at the right edge.
    prev_pad  = _set_padding_side(tokenizer, "left")
    prev_trunc = getattr(tokenizer, "truncation_side", None)
    if prev_trunc is not None:
        tokenizer.truncation_side = "left"

    try:
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_tokens,
            padding=True,
        ).to(device)
    finally:
        _restore_padding_side(tokenizer, prev_pad)
        if prev_trunc is not None:
            tokenizer.truncation_side = prev_trunc

    prompt_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    results: List[str] = []
    for seq in outputs:
        generated_ids   = seq[prompt_len:]
        generated_text  = tokenizer.decode(generated_ids, skip_special_tokens=True)
        first_line      = generated_text.split("\n")[0]
        results.append(first_line.rstrip())

    return results

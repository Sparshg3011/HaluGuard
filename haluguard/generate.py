"""
generate.py — Code generation wrapper for next-line prediction.

Wraps DeepSeek-Coder (or any causal LM) for the RepoBench next-line
prediction task.  The prompt format follows the RepoBench convention:
cross-file context snippets are prepended to the cropped code.

Usage in notebooks::

    from haluguard.generate import build_completion_prompt, generate_next_line

    prompt = build_completion_prompt(
        cropped_code=example["cropped_code"],
        import_statement=example["import_statement"],
        selected_snippets=[ctx["snippet"] for ctx in selected],
    )
    predicted = generate_next_line(prompt, tokenizer, model, device="cuda")
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch


def build_completion_prompt(
    cropped_code: str,
    import_statement: str,
    selected_snippets: List[str],
) -> str:
    """Assemble a prompt for next-line code completion.

    Prepends cross-file context snippets to the current file's code so the
    LLM can reference definitions, signatures, and imports from other files.

    Args:
        cropped_code:      Code written so far in the current file (the query).
        import_statement:  Import statements from the current file.
        selected_snippets: Code snippets from other files, selected by HCCS
                           or a baseline method.

    Returns:
        Formatted prompt string ready for a causal LM.
    """
    parts: List[str] = []

    for snippet in selected_snippets:
        parts.append(f"# Cross-file context:\n{snippet}")

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
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
    ).to(device)

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

"""
chunker.py — Split repository files into overlapping text chunks.

Each chunk carries a header comment identifying its source file and line range,
so the LLM and the HCCS scorer always have provenance information.

Example output header:
    # File: weather_app/api.py (lines 1-30)
"""

from typing import Dict, List


def chunk_repo(
    repo_files: Dict[str, str],
    max_lines: int = 30,
    stride: int = 15,
) -> List[str]:
    """Split a repository into text chunks suitable for embedding and retrieval.

    Small files (line count <= max_lines) produce a single chunk.  Larger
    files are split into overlapping windows of size ``max_lines`` with a
    step of ``stride`` lines, so context at window boundaries is not lost.

    Args:
        repo_files: Mapping of relative filepath to file source contents.
        max_lines:  Maximum source lines per chunk.  Default 30.
        stride:     Step size in lines between consecutive windows.  Must be
                    <= max_lines.  Set equal to max_lines for non-overlapping
                    windows.  Default 15 (50% overlap).

    Returns:
        Flat list of chunk strings.  Each chunk is prefixed with a header of
        the form ``# File: path/to/file.py (lines N-M)`` where N and M are
        1-indexed line numbers.

    Raises:
        ValueError: If stride > max_lines.
    """
    if stride > max_lines:
        raise ValueError(
            f"stride ({stride}) must be <= max_lines ({max_lines})"
        )

    chunks: List[str] = []

    for filepath, source in repo_files.items():
        lines = source.splitlines()
        n_lines = len(lines)

        if n_lines == 0:
            continue

        if n_lines <= max_lines:
            # Entire file fits in a single chunk (1-indexed in header)
            header = f"# File: {filepath} (lines 1-{n_lines})"
            chunks.append(header + "\n" + source)
        else:
            # Sliding window with overlap
            start = 0
            while start < n_lines:
                end = min(start + max_lines, n_lines)
                # Headers use 1-based line numbers (matches editors/tracebacks)
                header = f"# File: {filepath} (lines {start + 1}-{end})"
                window_text = "\n".join(lines[start:end])
                chunks.append(header + "\n" + window_text)
                if end == n_lines:
                    break
                start += stride

    return chunks


def chunk_text(
    filepath: str,
    source: str,
    max_lines: int = 30,
    stride: int = 15,
) -> List[str]:
    """Convenience wrapper to chunk a single file.

    Args:
        filepath: Relative path used in the chunk header.
        source:   File contents as a string.
        max_lines: Maximum lines per chunk.
        stride:   Step size between windows.

    Returns:
        List of chunk strings for this file.
    """
    return chunk_repo({filepath: source}, max_lines=max_lines, stride=stride)

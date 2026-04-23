"""
artifacts.py — Cached evaluation artifact discovery, loading, and compatibility.

For RepoBench, embeddings are produced by Notebook 01 (``01_data_pipeline.ipynb``)
and this module only validates and loads them.

For CrossCodeEval, the official cached-evaluation workflow only *loads* existing
embeddings. A separate utility remains available for one-off offline cache
creation, but nothing in the default notebook or matrix runner will invoke it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


REPOBENCH_QUERY_BACKENDS: Sequence[str] = ("codebert", "unixcoder")
REPOBENCH_QUERY_VIEWS: Sequence[str] = ("last3", "full")
CCEVAL_LEGACY_DEFAULT_QUERY_VIEW = "full"


@dataclass
class ArtifactStatus:
    """One file required or optionally used by cached evaluation."""

    name: str
    path: Path
    required: bool
    exists: bool
    role: str
    note: str = ""

    def to_row(self) -> Dict[str, Any]:
        """Return a JSON/table friendly row."""
        size_mb: Optional[float] = None
        if self.exists:
            size_mb = round(self.path.stat().st_size / (1024 * 1024), 2)
        return {
            "name": self.name,
            "path": str(self.path),
            "required": self.required,
            "exists": self.exists,
            "role": self.role,
            "size_mb": size_mb,
            "note": self.note,
        }


def default_project_root() -> Path:
    """Resolve the project root from ``HALUGUARD_ROOT`` or the current file."""
    override = os.environ.get("HALUGUARD_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def default_data_dir(root: Optional[Path] = None) -> Path:
    """Resolve the data directory used for cached artifacts."""
    override = os.environ.get("HALUGUARD_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    base = root if root is not None else default_project_root()
    return Path(base) / "data"


def embeddings_dir(data_dir: Optional[Path] = None) -> Path:
    """Return ``data/embeddings`` for a data directory."""
    base = data_dir if data_dir is not None else default_data_dir()
    return Path(base) / "embeddings"


def repobench_required_paths(data_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Return required cached RepoBench artifacts."""
    emb = embeddings_dir(data_dir)
    return {
        "gold_indices": emb / "gold_indices.pt",
        "test_indices": emb / "test_indices.pt",
        "chunk_codebert": emb / "chunk_embeddings__codebert.pt",
        "chunk_unixcoder": emb / "chunk_embeddings__unixcoder.pt",
        "query_codebert_last3": emb / "query_embeddings__codebert__last3.pt",
        "query_codebert_full": emb / "query_embeddings__codebert__full.pt",
        "query_unixcoder_last3": emb / "query_embeddings__unixcoder__last3.pt",
        "query_unixcoder_full": emb / "query_embeddings__unixcoder__full.pt",
    }


def cceval_embedding_paths(data_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Return optional cached CrossCodeEval embedding artifact paths."""
    emb = embeddings_dir(data_dir)
    return {
        "query": emb / "cceval_query_embeddings.pt",
        "chunks": emb / "cceval_chunk_embeddings.pt",
        "query_meta": emb / "cceval_query_embeddings.meta.json",
        "chunks_meta": emb / "cceval_chunk_embeddings.meta.json",
    }


def cceval_embeddings_available(data_dir: Optional[Path] = None) -> bool:
    """Return True when CrossCodeEval query and chunk embeddings are cached."""
    paths = cceval_embedding_paths(data_dir)
    return paths["query"].exists() and paths["chunks"].exists()


def _normalise_backend_name(value: Any) -> Optional[str]:
    """Map encoder/backend labels onto the project's backend names."""
    if value is None:
        return None
    text = str(value).strip().lower()
    aliases = {
        "codebert": "codebert",
        "microsoft/codebert-base": "codebert",
        "unixcoder": "unixcoder",
        "microsoft/unixcoder-base": "unixcoder",
    }
    return aliases.get(text)


def cceval_cache_metadata(data_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Return normalized CrossCodeEval cache metadata when available.

    The cached CrossCodeEval format stores a single embedding backend and query
    view. Legacy caches written before query-view metadata existed are treated
    as ``query_view="full"`` because they embed the full ``cropped_code``.
    """
    if not cceval_embeddings_available(data_dir):
        return None

    paths = cceval_embedding_paths(data_dir)
    query_meta = read_json_if_exists(paths["query_meta"])
    chunk_meta = read_json_if_exists(paths["chunks_meta"])
    raw_meta = query_meta or chunk_meta or {}

    query_backend = _normalise_backend_name(
        (query_meta or {}).get("backend")
        or (query_meta or {}).get("encoder")
    )
    chunk_backend = _normalise_backend_name(
        (chunk_meta or {}).get("backend")
        or (chunk_meta or {}).get("encoder")
    )
    query_backend = query_backend or chunk_backend
    chunk_backend = chunk_backend or query_backend

    query_query_view = (query_meta or {}).get("query_view")
    chunk_query_view = (chunk_meta or {}).get("query_view")
    inferred_legacy_query_view = False
    if query_query_view is None and chunk_query_view is None:
        query_query_view = CCEVAL_LEGACY_DEFAULT_QUERY_VIEW
        chunk_query_view = CCEVAL_LEGACY_DEFAULT_QUERY_VIEW
        inferred_legacy_query_view = True
    elif query_query_view is None:
        query_query_view = chunk_query_view
    elif chunk_query_view is None:
        chunk_query_view = query_query_view

    metadata_consistent = (
        query_backend == chunk_backend
        and str(query_query_view) == str(chunk_query_view)
    )

    backend = query_backend if metadata_consistent else None
    query_view = str(query_query_view) if metadata_consistent else None
    n_examples = raw_meta.get("n_examples")
    hidden_size = raw_meta.get("hidden_size")

    return {
        "backend": backend,
        "query_view": query_view,
        "encoder": raw_meta.get("encoder"),
        "n_examples": n_examples,
        "hidden_size": hidden_size,
        "metadata_consistent": metadata_consistent,
        "legacy_query_view_inferred": inferred_legacy_query_view,
        "query_meta_present": query_meta is not None,
        "chunks_meta_present": chunk_meta is not None,
    }


def cceval_cache_supports(
    data_dir: Optional[Path],
    backend: Optional[str],
    query_view: Optional[str],
) -> bool:
    """Return True when the cached CrossCodeEval artifacts match a method."""
    if not cceval_embeddings_available(data_dir):
        return False
    meta = cceval_cache_metadata(data_dir)
    if meta is None or meta.get("backend") is None or meta.get("query_view") is None:
        return False
    return (
        str(meta["backend"]) == str(backend)
        and str(meta["query_view"]) == str(query_view)
    )


def cceval_method_skip_reason(
    data_dir: Optional[Path],
    backend: Optional[str],
    query_view: Optional[str],
) -> Optional[str]:
    """Return the exact skip reason for a CrossCodeEval cached-embedding method."""
    if not cceval_embeddings_available(data_dir):
        return "skipped_missing_cached_embeddings"
    if not cceval_cache_supports(data_dir, backend, query_view):
        return "skipped_incompatible_cached_embeddings"
    return None


def build_artifact_manifest(
    data_dir: Optional[Path] = None,
    checkpoint_dirs: Optional[Sequence[Path]] = None,
) -> List[Dict[str, Any]]:
    """Build a preflight manifest for notebooks and reports."""
    rows: List[ArtifactStatus] = []
    for name, path in repobench_required_paths(data_dir).items():
        rows.append(
            ArtifactStatus(
                name=name,
                path=path,
                required=True,
                exists=path.exists(),
                role="repobench_cached_embedding",
            )
        )

    for name, path in cceval_embedding_paths(data_dir).items():
        rows.append(
            ArtifactStatus(
                name=f"cceval_{name}",
                path=path,
                required=False,
                exists=path.exists(),
                role="crosscodeeval_optional_embedding",
                note="enables CrossCodeEval cosine/HaluGuard rows",
            )
        )

    cceval_meta = cceval_cache_metadata(data_dir)
    if cceval_meta is not None:
        rows.append(
            ArtifactStatus(
                name="cceval_cache_variant",
                path=embeddings_dir(data_dir),
                required=False,
                exists=True,
                role="crosscodeeval_cache_metadata",
                note=(
                    f"backend={cceval_meta.get('backend') or 'unknown'}, "
                    f"query_view={cceval_meta.get('query_view') or 'unknown'}"
                ),
            )
        )

    for ckpt_dir in checkpoint_dirs or []:
        ckpt_dir = Path(ckpt_dir)
        rows.append(
            ArtifactStatus(
                name=f"checkpoint_dir:{ckpt_dir.name}",
                path=ckpt_dir,
                required=False,
                exists=ckpt_dir.exists(),
                role="checkpoint_directory",
            )
        )

    return [row.to_row() for row in rows]


def validate_repobench_cached_artifacts(data_dir: Optional[Path] = None) -> None:
    """Raise ``FileNotFoundError`` if any required RepoBench cache is missing."""
    missing = [
        str(path)
        for path in repobench_required_paths(data_dir).values()
        if not path.exists()
    ]
    if missing:
        joined = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "Missing required cached RepoBench artifacts. Run the data pipeline "
            "or point HALUGUARD_DATA_DIR at an existing cache:\n"
            f"{joined}"
        )


def _torch_load(path: Path, map_location: str = "cpu") -> Any:
    """Load a torch artifact with a local import to keep module import cheap."""
    import torch

    return torch.load(Path(path), map_location=map_location)


def load_repobench_cached_artifacts(
    data_dir: Optional[Path] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load all cached RepoBench tensors into a dictionary."""
    validate_repobench_cached_artifacts(data_dir)
    paths = repobench_required_paths(data_dir)
    payload: Dict[str, Any] = {
        "gold_indices": _torch_load(paths["gold_indices"], map_location=map_location),
        "test_indices": _torch_load(paths["test_indices"], map_location=map_location),
        "query_embeddings": {},
        "chunk_embeddings": {},
    }
    for backend in REPOBENCH_QUERY_BACKENDS:
        payload["chunk_embeddings"][backend] = _torch_load(
            paths[f"chunk_{backend}"],
            map_location=map_location,
        )
        for view in REPOBENCH_QUERY_VIEWS:
            payload["query_embeddings"][(backend, view)] = _torch_load(
                paths[f"query_{backend}_{view}"],
                map_location=map_location,
            )
    return payload


def load_cceval_cached_artifacts(
    data_dir: Optional[Path] = None,
    map_location: str = "cpu",
) -> Optional[Dict[str, Any]]:
    """Load optional CrossCodeEval embeddings, returning ``None`` if absent."""
    if not cceval_embeddings_available(data_dir):
        return None
    paths = cceval_embedding_paths(data_dir)
    return {
        "query_embeddings": _torch_load(paths["query"], map_location=map_location),
        "chunk_embeddings": _torch_load(paths["chunks"], map_location=map_location),
        "meta": cceval_cache_metadata(data_dir),
    }


def read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON object if present, returning ``None`` otherwise."""
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compute_and_save_cceval_embeddings(
    loader: Any,
    encoder: Any,
    tokenizer: Any,
    device: str = "cpu",
    data_dir: Optional[Path] = None,
    limit: int = 500,
    batch_size: int = 32,
    encoder_name: str = "unixcoder",
) -> Dict[str, Any]:
    """Embed every CrossCodeEval example and save to ``data/embeddings/``.

    Produces two ``torch`` files (query tensor + list of chunk tensors) and two
    companion ``.meta.json`` files recording the encoder name, example count,
    and timestamp.

    This utility is intentionally *not* used by the official cached-evaluation
    notebook. It exists for explicit offline precomputation only.

    The function is idempotent: if the files already exist they are
    **overwritten** so callers can force a refresh by deleting the old files.

    Args:
        loader:       A ``CrossCodeEvalLoader`` (or compatible) instance whose
                      ``iter_examples`` yields :class:`~haluguard.benchmarks.base.Example`
                      objects.
        encoder:      Pre-loaded encoder model (e.g. CodeBERT ``AutoModel``).
        tokenizer:    Matching tokenizer.
        device:       PyTorch device string (``"cpu"`` or ``"cuda"``).
        data_dir:     Root data directory.  Defaults to the project ``data/``
                      folder resolved via :func:`default_data_dir`.
        limit:        Maximum number of examples to embed.
        batch_size:   Mini-batch size passed to :func:`~haluguard.hccs.batch_embed`.
        encoder_name: Label written into the meta JSON so you know which
                      encoder produced the cache.  Should match
                      ``DEFAULT_SCORER_BACKEND`` in ``eval_matrix.py``
                      (default ``"unixcoder"``).

    Returns:
        Dict with keys ``"query_embeddings"`` (list of numpy arrays, one per
        example, shape ``(hidden,)``) and ``"chunk_embeddings"`` (list of numpy
        arrays, shape ``(n_chunks_i, hidden)`` per example).

        Note: the *saved* files store these as PyTorch tensors, so
        :func:`load_cceval_cached_artifacts` returns
        ``Tensor(N, hidden)`` / ``List[Tensor]`` rather than numpy arrays.
        For immediate post-computation use the numpy return value is
        sufficient; for evaluation always load via
        :func:`load_cceval_cached_artifacts`.
    """
    import numpy as np
    import torch

    from haluguard.hccs import batch_embed, embed_code

    emb_dir = embeddings_dir(data_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)
    paths = cceval_embedding_paths(data_dir)

    query_embs: List[Any] = []
    chunk_embs_list: List[Any] = []

    print(f"[artifacts] Computing CrossCodeEval embeddings (limit={limit}) …")
    for i, example in enumerate(loader.iter_examples(limit=limit)):
        # Query embedding: 1-D array shape (hidden_size,)
        q_emb = embed_code(
            example.cropped_code,
            tokenizer,
            encoder,
            device=device,
        )
        query_embs.append(q_emb)

        # Chunk embeddings: 2-D array shape (n_chunks, hidden_size)
        if example.context_chunks:
            snippets = [c["snippet"] for c in example.context_chunks]
            c_embs = batch_embed(
                snippets,
                tokenizer,
                encoder,
                device=device,
                batch_size=batch_size,
            )
        else:
            c_embs = np.empty((0, q_emb.shape[0]), dtype=np.float32)
        chunk_embs_list.append(c_embs)

        if (i + 1) % 50 == 0:
            print(f"[artifacts]   embedded {i + 1}/{limit} examples")

    # Stack query embeddings into a single (N, hidden) tensor for compact storage.
    query_tensor = torch.tensor(np.stack(query_embs, axis=0), dtype=torch.float32)

    # Chunk embeddings are ragged (different n_chunks per example) — save as list.
    chunk_tensors = [
        torch.tensor(c, dtype=torch.float32) for c in chunk_embs_list
    ]

    torch.save(query_tensor, paths["query"])
    torch.save(chunk_tensors, paths["chunks"])
    print(f"[artifacts] Saved query tensor {tuple(query_tensor.shape)} → {paths['query']}")
    print(f"[artifacts] Saved {len(chunk_tensors)} chunk tensors → {paths['chunks']}")

    # Write companion meta files.
    meta = {
        "encoder": encoder_name,
        "backend": _normalise_backend_name(encoder_name) or str(encoder_name),
        "query_view": CCEVAL_LEGACY_DEFAULT_QUERY_VIEW,
        "n_examples": len(query_embs),
        "hidden_size": int(query_tensor.shape[1]),
        "limit": limit,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    paths["query_meta"].write_text(json.dumps(meta, indent=2))
    paths["chunks_meta"].write_text(json.dumps(meta, indent=2))

    return {
        "query_embeddings": [q_emb for q_emb in query_embs],
        "chunk_embeddings": chunk_embs_list,
    }


def discover_checkpoint_dirs(
    root: Optional[Path] = None,
    extra_dirs: Optional[Iterable[Path]] = None,
) -> List[Path]:
    """Return candidate checkpoint directories in priority order."""
    base = root if root is not None else default_project_root()
    candidates: List[Path] = []

    env_dir = os.environ.get("HALUGUARD_CKPT_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.append(Path(base) / "checkpoints")

    drive_root = os.environ.get("HALUGUARD_DRIVE_ROOT")
    if drive_root:
        candidates.append(Path(drive_root).expanduser() / "checkpoints")

    if extra_dirs:
        candidates.extend(Path(p).expanduser() for p in extra_dirs)

    seen: Set[str] = set()
    out: List[Path] = []
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            out.append(path)
            seen.add(key)
    return out

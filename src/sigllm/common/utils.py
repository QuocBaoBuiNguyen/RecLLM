from datetime import datetime
from pathlib import Path
import os
from urllib.parse import urlparse

from sigllm.common import registry

def now():
    return datetime.now().strftime("%Y%m%d%H%M")[:-1]


def derive_job_id_from_llm(cfg, fallback=None):
    """Derive a stable run id from the LLM backbone slug, e.g. ``qwen2-7b-base``.

    Replaces the timestamp-based job_id so checkpoint paths become
    ``<output_dir>/qwen2-7b-base/checkpoint_best.pth`` instead of
    ``<output_dir>/20260523041/checkpoint_best.pth``. Easier to reference,
    no manual path edits when chaining Stage 3 Step 1 → Step 2.

    Trade-off: re-running with the same LLM overwrites the previous ckpt.
    Archive manually to Drive if multiple runs need to coexist.
    """
    llm_path = ""
    try:
        llm_path = cfg.model_cfg.get("llm_model", "")
    except AttributeError:
        try:
            llm_path = cfg.model.get("llm_model", "")
        except AttributeError:
            llm_path = ""

    if llm_path:
        slug = os.path.basename(str(llm_path).rstrip("/"))
        if slug:
            return slug

    return fallback if fallback is not None else now()

def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def get_abs_path(rel_path):
    return os.path.join(registry.get_path("library_root"), rel_path)


def resolve_hf_model_path(path: str) -> tuple[str, bool]:
    """Return a resolved model path and whether it should be loaded locally."""
    candidate = Path(path).expanduser()
    if candidate.exists():
        # When using local model directories, ensure HF hub stays offline
        # to avoid treating the path as a repo id and triggering validation.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        return str(candidate), True
    return path, False

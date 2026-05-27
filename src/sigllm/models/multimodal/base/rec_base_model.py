
import contextlib
import os
from typing import Optional

import torch
from sigllm.common.dist_utils import download_cached_file
from sigllm.common.logging_utils import NotebookLogger
from sigllm.common.utils import is_url
from sigllm.models.multimodal.base.base_model import BaseModel
from sigllm.models.rec.matrix_factorization import MatrixFactorization

LOGGER = NotebookLogger.rich_logger("sigllm.rec_base_model")

def log_step(title: str, detail: Optional[str] = None) -> None:
    """Emit a compact log line with optional detail string."""

    message = title if detail is None else f"{title} | {detail}"
    LOGGER.info(message)


class Rec2Base(BaseModel):
    
    def to_be_trained(self):
        pass

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.amp.autocast('cuda', dtype=dtype)
        else:
            return contextlib.nullcontext()
        
    def init_rec_encoder(self, rec_model, config):
        if rec_model == "MF":
            log_step("Initializing Matrix Factorization model")
            rec_model = MatrixFactorization(config)
        else:
            raise NotImplementedError(f"Rec model {rec_model} not implemented.")

        return rec_model
    
    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        msg = self.load_state_dict(state_dict, strict=False)
        log_step("Load checkpoint from %s" % url_or_filename)

        return msg
    

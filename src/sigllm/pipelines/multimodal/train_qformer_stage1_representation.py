import argparse
import random
import torch
from torch.optim import Adam
import omegaconf
import os
import numpy as np
from typing import Optional

from sigllm.common import NotebookLogger, EarlyStopping
from sigllm.common.config import Config
from sigllm.datasets.qformer.qformer_loader import build_qformer_loaders
from sigllm.models.rec.matrix_factorization import MatrixFactorization
from sigllm.models.q_former.hf_qformer_adapter import HFQFormerAdapter
from sigllm.models.projection.qformer_alignment_model import QRecInstructAlignmentModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LOGGER = NotebookLogger.rich_logger("sigllm.train_qformer_stage1")


def log_step(title: str, detail: Optional[str] = None) -> None:
    message = title if detail is None else f"{title} | {detail}"
    LOGGER.info(message)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def _init_rec_model(cfg, device):
    """
    Initializes the recommendation model, loads pretrained weights if available,
    and freezes parameters if configured.
    """
    mf_config = omegaconf.OmegaConf.create({
        # TEMP_DISABLED_USER_CF: user_num is still needed to build/load MF weights,
        # but stage 1 no longer calls mf.user_encoder().
        "user_num": int(cfg.user_num),
        "item_num": int(cfg.item_num),
        "embedding_size": int(cfg.embedding_size)
    })
    mf = MatrixFactorization(mf_config).to(device)

    pretrained_rec_path = cfg.pretrained_rec_path
    if mf is not None and os.path.exists(pretrained_rec_path):
        mf.load_state_dict(torch.load(pretrained_rec_path, map_location="cpu"))
        print(f"Successfully loaded the pretrained rec model from {pretrained_rec_path}")

    if cfg.freeze_rec and mf is not None:
        for param in mf.parameters():
            param.requires_grad = False
        mf.eval()
        mf.train = disabled_train.__get__(mf, MatrixFactorization)
        print("Freeze rec encoder completed")

    return mf


def _init_qformer(cfg, d_model, device):
    """
    Initializes the Q-Former model.
    """
    qformer_output_dim = cfg.qformer_output_dim
    if qformer_output_dim is None:
        qformer_output_dim = d_model

    return HFQFormerAdapter(
        d_cf=cfg.embedding_size,
        d_model=d_model,
        num_queries=cfg.num_queries,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        output_dim=int(qformer_output_dim),
        qformer_text_model_name=cfg.qformer_text_model_name,
        max_instruction_length=cfg.get("max_instruction_length", 48),
    ).to(device)


def _init_optimizer(model, lr, weight_decay=0.0):
    """
    Initializes the Adam optimizer for trainable parameters.
    """
    return Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )


def _log_batch_preview(batch, prefix: str = "train_step"):
    """Print a compact preview of the current batch for debugging."""
    batch_size = batch["i_left"].size(0)
    type_counts = {sample_type: batch["sample_type"].count(sample_type) for sample_type in set(batch["sample_type"])}

    print(
        f"[{prefix}] batch_size={batch_size} type_counts={type_counts} "
        f"u.shape={tuple(batch['u'].shape)} i_left.shape={tuple(batch['i_left'].shape)} "
        f"i_right.shape={tuple(batch['i_right'].shape)}"
    )
    if batch_size > 0:
        print(
            f"[{prefix}] sample[0] type={batch['sample_type'][0]} u={batch['u'][0].item()} "
            f"i_left={batch['i_left'][0].item()} i_right={batch['i_right'][0].item()}"
        )
        print(f"[{prefix}] sample[0] instruction={batch['instruction'][0]}")
        print(f"[{prefix}] sample[0] text={batch['text'][0]}")


def _move_batch_to_device(batch, device):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


def _subset_batch(batch, indices):
    index_list = indices.detach().cpu().tolist()
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value[indices]
        else:
            out[key] = [value[i] for i in index_list]
    return out


def _indices_for_type(batch, sample_type: str, device):
    indices = [idx for idx, current in enumerate(batch["sample_type"]) if current == sample_type]
    return torch.tensor(indices, dtype=torch.long, device=device)


# Which sample type each metric depends on. A batch holding none of that type
# produces no loss for the metric, so it must be excluded from the average
# rather than contributing a zero.
METRIC_GROUPS = {
    "L_itc": "item_text", "itc_top1": "item_text",
    "L_itm": "item_text", "itm_acc": "item_text",
    "L_itg": "item_text", "itg_acc": "item_text",
    "L_llm": "item_text", "llm_top1": "item_text",
    "L_ii": "item_item", "ii_top1": "item_item",
    "L_ui": "user_item", "ui_top1": "user_item",
}

SAMPLE_TYPES = ("item_text", "item_item", "user_item")


class MetricAccumulator:
    """Average each metric over the batches where its sample type was present.

    Averaging over *all* batches scales every metric by the fraction of
    batches carrying its type, and that fraction differs sharply between
    splits: the train loader shuffles (types interleave, so every batch holds
    a few item_text rows) while val/test do not (the builder emits item_text
    as one contiguous block, so a handful of val batches are almost entirely
    item_text and the rest hold none). Diluted averages are therefore not
    comparable across splits.

    ``n_<type>`` records the mean number of rows of that type per contributing
    batch. In-batch contrastives draw negatives from exactly those rows, so
    chance accuracy is ``1 / n`` — without ``n``, a top-1 number cannot be
    read at all.
    """

    def __init__(self):
        self.loss_total = 0.0
        self.steps = 0
        self.sums = {key: 0.0 for key in METRIC_GROUPS}
        self.active = {key: 0 for key in METRIC_GROUPS}
        self.n_sums = {group: 0.0 for group in SAMPLE_TYPES}
        self.n_active = {group: 0 for group in SAMPLE_TYPES}

    def update(self, loss, logs, counts):
        self.loss_total += float(loss.item())
        self.steps += 1
        for key, group in METRIC_GROUPS.items():
            if counts[group] >= 2:
                self.sums[key] += float(logs[key].item())
                self.active[key] += 1
        for group in SAMPLE_TYPES:
            if counts[group] >= 2:
                self.n_sums[group] += float(counts[group])
                self.n_active[group] += 1

    def result(self):
        out = {"loss": self.loss_total / self.steps if self.steps else 0.0}
        for key in METRIC_GROUPS:
            out[key] = self.sums[key] / self.active[key] if self.active[key] else 0.0
        for group in SAMPLE_TYPES:
            out[f"n_{group}"] = (
                self.n_sums[group] / self.n_active[group] if self.n_active[group] else 0.0
            )
            out[f"frac_{group}"] = self.n_active[group] / self.steps if self.steps else 0.0
        return out


def train_step(
    batch,
    model: QRecInstructAlignmentModel,
    w_itc: float = 1.0,
    w_itm: float = 1.0,
    w_itg: float = 1.0,
    w_ii: float = 1.0,
    w_ui: float = 0.0,
    w_llm: float = 0.0,
    tau_itc: float = 0.07,
    tau_ii: float = 0.07,
    tau_ui: float = 0.07,
    tau_llm: float = 0.07,
    debug_batch: bool = False,
):
    """BLIP-2 stage-1 step: ITC + ITM + ITG on item-text samples, plus the
    SigLLM-specific item-item and (ILM-style) user-item contrastives."""

    device = batch["i_left"].device

    if debug_batch:
        _log_batch_preview(batch)

    zero = next(model.parameters()).sum() * 0.0
    logs = {
        "L_itc": zero,
        "L_itm": zero,
        "L_itg": zero,
        "L_ii": zero,
        "L_ui": zero,
        "L_llm": zero,
        "itc_top1": zero.detach(),
        "itm_acc": zero.detach(),
        "itg_acc": zero.detach(),
        "ii_top1": zero.detach(),
        "ui_top1": zero.detach(),
        "llm_top1": zero.detach(),
    }
    losses = []

    item_text_idx = _indices_for_type(batch, "item_text", device)
    if item_text_idx.numel() >= 2:
        item_text_batch = _subset_batch(batch, item_text_idx)
        item_ids = item_text_batch["i_left"]
        text_list = item_text_batch["text"]

        loss_itc, sim_matrix, itc_top1 = model.loss_itc(item_ids, text_list, tau=tau_itc)
        logs["L_itc"] = loss_itc
        logs["itc_top1"] = itc_top1.detach()
        losses.append(w_itc * loss_itc)

        if w_itm > 0.0:
            loss_itm, itm_acc = model.loss_itm(item_ids, text_list, sim_matrix)
            logs["L_itm"] = loss_itm
            logs["itm_acc"] = itm_acc.detach()
            losses.append(w_itm * loss_itm)

        if w_itg > 0.0:
            loss_itg, itg_acc = model.loss_itg(item_ids, text_list)
            logs["L_itg"] = loss_itg
            logs["itg_acc"] = itg_acc.detach()
            losses.append(w_itg * loss_itg)

        if w_llm > 0.00 and getattr(model, "has_llm_align", False):
            loss_llm, llm_top1 = model.loss_llm_align(item_ids, tau=tau_llm)
            logs["L_llm"] = loss_llm
            logs["llm_top1"] = llm_top1.detach()
            losses.append(w_llm * loss_llm)

    item_item_idx = _indices_for_type(batch, "item_item", device)
    if w_ii > 0.0 and item_item_idx.numel() >= 2:
        item_item_batch = _subset_batch(batch, item_item_idx)
        loss_ii, ii_top1 = model.loss_item_item_ilm(
            item_item_batch["i_left"], item_item_batch["i_right"], tau=tau_ii
        )
        logs["L_ii"] = loss_ii
        logs["ii_top1"] = ii_top1.detach()
        losses.append(w_ii * loss_ii)

    user_item_idx = _indices_for_type(batch, "user_item", device)
    if w_ui > 0.0 and user_item_idx.numel() >= 2:
        user_item_batch = _subset_batch(batch, user_item_idx)
        loss_ui, ui_top1 = model.loss_user_item(
            user_item_batch["u"], user_item_batch["i_left"], tau=tau_ui
        )
        logs["L_ui"] = loss_ui
        logs["ui_top1"] = ui_top1.detach()
        losses.append(w_ui * loss_ui)

    loss = sum(losses, zero)
    counts = {
        "item_text": int(item_text_idx.numel()),
        "item_item": int(item_item_idx.numel()),
        "user_item": int(user_item_idx.numel()),
    }
    return loss, logs, counts


def evaluate_loss(
    model,
    loader,
    w_itc=1.0,
    w_itm=1.0,
    w_itg=1.0,
    w_ii=1.0,
    w_ui=0.0,
    w_llm=0.0,
    tau_itc=0.07,
    tau_ii=0.07,
    tau_ui=0.07,
    tau_llm=0.07,
):
    model.eval()
    device = next(model.parameters()).device
    accumulator = MetricAccumulator()

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            loss, logs, counts = train_step(
                batch,
                model,
                w_itc=w_itc,
                w_itm=w_itm,
                w_itg=w_itg,
                w_ii=w_ii,
                w_ui=w_ui,
                w_llm=w_llm,
                tau_itc=tau_itc,
                tau_ii=tau_ii,
                tau_ui=tau_ui,
                tau_llm=tau_llm,
            )
            accumulator.update(loss, logs, counts)

    model.train()
    return accumulator.result()


def _save_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    epoch,
    val_logs,
):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            **{f"val_{key}": value for key, value in val_logs.items()},
        },
        checkpoint_path,
    )

def _load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def train_qformer_stage1_representation(cfg):
    set_seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = build_qformer_loaders(cfg, data_dir=cfg.data_dir)
    
    mf = _init_rec_model(cfg, device)
    qformer_d_model = int(cfg.get("qformer_d_model", 768))
    qformer = _init_qformer(cfg, qformer_d_model, device)

    w_llm = float(cfg.get("w_llm", 0.0))
    tau_llm = float(cfg.get("tau_llm", 0.07))
    item_llm_emb = None
    d_llm = None
    item_llm_emb_path = cfg.get("item_llm_emb_path", None)
    if w_llm > 0.0:
        if not item_llm_emb_path or not os.path.exists(item_llm_emb_path):
            raise FileNotFoundError(f"item_llm_emb_path is required and must exist when w_llm > 0.0, but got: {item_llm_emb_path}")
        blob = torch.load(item_llm_emb_path, map_location="cpu")
        item_llm_emb = blob["item_llm_emb"] if isinstance(blob, dict) else blob
        d_llm = int(item_llm_emb.size(-1))
        log_step("Loaded item LLM embeddings", f"path={item_llm_emb_path}, shape={tuple(item_llm_emb.shape)}")

    model = QRecInstructAlignmentModel(
        mf=mf,
        qformer=qformer,
        item_llm_emb=item_llm_emb,
        d_llm=d_llm,
    ).to(device)

    opt = _init_optimizer(model, cfg.lr, weight_decay=cfg.weight_decay)

    outdir = cfg.output_dir
    os.makedirs(outdir, exist_ok=True)
    best_checkpoint_path = os.path.join(outdir, cfg.best_checkpoint_name)
    stopper = EarlyStopping(
        ref_metric="val_loss",
        monitor_mode="min",
        patience=cfg.early_stopping_patience,
        min_delta=cfg.early_stopping_min_delta,
    )
    log_step("Training setup", f"seed={cfg.seed}, output_dir={outdir}")

    w_ui = float(cfg.get("w_ui", 0.0))
    tau_ui = float(cfg.get("tau_ui", 0.07))

    for epoch in range(cfg.epoch):
        model.train()
        accumulator = MetricAccumulator()
        train_steps = 0
        for batch in train_loader:
            batch = _move_batch_to_device(batch, device)
            opt.zero_grad()

            loss, logs, counts = train_step(
                batch,
                model,
                w_itc=cfg.w_itc,
                w_itm=cfg.w_itm,
                w_itg=cfg.w_itg,
                w_ii=cfg.w_ii,
                w_ui=w_ui,
                w_llm=w_llm,
                tau_itc=cfg.tau_itc,
                tau_ii=cfg.tau_ii,
                tau_ui=tau_ui,
                tau_llm=tau_llm,
                debug_batch=cfg.debug_batch and epoch == 0 and train_steps < cfg.debug_batch_max_steps,
            )
            loss.backward()
            opt.step()

            accumulator.update(loss, logs, counts)
            train_steps += 1

        if (epoch + 1) % cfg.log_epoch == 0:
            avg_train = accumulator.result()
            val_logs = evaluate_loss(
                model,
                val_loader,
                w_itc=cfg.w_itc,
                w_itm=cfg.w_itm,
                w_itg=cfg.w_itg,
                w_ii=cfg.w_ii,
                w_ui=w_ui,
                w_llm=w_llm,
                tau_itc=cfg.tau_itc,
                tau_ii=cfg.tau_ii,
                tau_ui=tau_ui,
                tau_llm=tau_llm,
            )
            print(
                f"epoch {epoch+1} | "
                f"Train Loss={avg_train['loss']:.4f} "
                f"L_itc={avg_train['L_itc']:.4f} L_itm={avg_train['L_itm']:.4f} "
                f"L_itg={avg_train['L_itg']:.4f} L_ii={avg_train['L_ii']:.4f} "
                f"L_ui={avg_train['L_ui']:.4f} L_llm={avg_train['L_llm']:.4f} "
                f"ITC@1={avg_train['itc_top1']:.4f} ITM_acc={avg_train['itm_acc']:.4f} "
                f"ITG_acc={avg_train['itg_acc']:.4f} II@1={avg_train['ii_top1']:.4f} "
                f"UI@1={avg_train['ui_top1']:.4f} LLM@1={avg_train['llm_top1']:.4f} "
                f"n_it={avg_train['n_item_text']:.1f}(chance={1.0 / max(avg_train['n_item_text'], 1.0):.4f}) "
                f"n_ii={avg_train['n_item_item']:.1f} n_ui={avg_train['n_user_item']:.1f} | "
                f"Val Loss={val_logs['loss']:.4f} "
                f"L_itc={val_logs['L_itc']:.4f} L_itm={val_logs['L_itm']:.4f} "
                f"L_itg={val_logs['L_itg']:.4f} L_ii={val_logs['L_ii']:.4f} "
                f"L_ui={val_logs['L_ui']:.4f} L_llm={val_logs['L_llm']:.4f} "
                f"ITC@1={val_logs['itc_top1']:.4f} ITM_acc={val_logs['itm_acc']:.4f} "
                f"ITG_acc={val_logs['itg_acc']:.4f} II@1={val_logs['ii_top1']:.4f} "
                f"UI@1={val_logs['ui_top1']:.4f} LLM@1={val_logs['llm_top1']:.4f} "
                f"n_it={val_logs['n_item_text']:.1f}(chance={1.0 / max(val_logs['n_item_text'], 1.0):.4f}) "
                f"n_ii={val_logs['n_item_item']:.1f} n_ui={val_logs['n_user_item']:.1f} | "
                f"w_itc={cfg.w_itc:.3f} w_itm={cfg.w_itm:.3f} w_itg={cfg.w_itg:.3f} "
                f"w_ii={cfg.w_ii:.3f} w_ui={w_ui:.3f} w_llm={w_llm:.3f} "
                f"tau_itc={cfg.tau_itc:.3f} tau_ii={cfg.tau_ii:.3f} tau_ui={tau_ui:.3f}"
            )

            metrics = {
                "epoch": epoch + 1,
                **{f"val_{key}": value for key, value in val_logs.items()},
                **{f"train_{key}": value for key, value in avg_train.items()},
            }
            improved = stopper.update(metrics)

            if improved:
                _save_checkpoint(
                    best_checkpoint_path,
                    model,
                    opt,
                    epoch + 1,
                    val_logs,
                )
                log_step("Saved new best checkpoint", f"epoch={epoch + 1}, path={best_checkpoint_path}")
            else:
                best_epoch = stopper.best_full_metric["epoch"] if stopper.best_full_metric is not None else "n/a"
                best_val_loss = stopper.best_metric_val
                log_step(
                    "No validation improvement",
                    f"counter={stopper.counter}, best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f}",
                )

            if stopper.should_stop:
                best_epoch = stopper.best_full_metric["epoch"] if stopper.best_full_metric is not None else "n/a"
                best_val_loss = stopper.best_metric_val
                log_step(
                    "Early stopping triggered",
                    f"epoch={epoch + 1}, best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f}",
                )
                break

    best_checkpoint = None
    if stopper.best_full_metric is not None and os.path.exists(best_checkpoint_path):
        best_checkpoint = _load_checkpoint(best_checkpoint_path, model)
        log_step(
            "Loaded best checkpoint",
            (
                f"epoch={best_checkpoint['epoch']}, val_loss={best_checkpoint['val_loss']:.4f}, "
                f"itc_top1={best_checkpoint.get('val_itc_top1', 0.0):.4f}, "
                f"itm_acc={best_checkpoint.get('val_itm_acc', 0.0):.4f}, "
                f"itg_acc={best_checkpoint.get('val_itg_acc', 0.0):.4f}, "
                f"ii_top1={best_checkpoint.get('val_ii_top1', 0.0):.4f}"
            ),
        )

    # Final Test
    log_step("Evaluating on Test Set")
    test_logs = evaluate_loss(
        model,
        test_loader,
        w_itc=cfg.w_itc,
        w_itm=cfg.w_itm,
        w_itg=cfg.w_itg,
        w_ii=cfg.w_ii,
        w_ui=w_ui,
        w_llm=w_llm,
        tau_itc=cfg.tau_itc,
        tau_ii=cfg.tau_ii,
        tau_ui=tau_ui,
        tau_llm=tau_llm,
    )
    log_step(
        "Test results",
        (
            f"loss={test_logs['loss']:.4f}, l_itc={test_logs['L_itc']:.4f}, "
            f"l_itm={test_logs['L_itm']:.4f}, l_itg={test_logs['L_itg']:.4f}, "
            f"l_ii={test_logs['L_ii']:.4f}, l_llm={test_logs['L_llm']:.4f}, "
            f"itc@1={test_logs['itc_top1']:.4f}, "
            f"itm_acc={test_logs['itm_acc']:.4f}, itg_acc={test_logs['itg_acc']:.4f}, "
            f"ii@1={test_logs['ii_top1']:.4f}, llm@1={test_logs['llm_top1']:.4f}, "
            f"n_it={test_logs['n_item_text']:.1f}"
        ),
    )

    if best_checkpoint is not None:
        best_qformer_path = os.path.join(outdir, cfg.best_qformer_weights_name)
        torch.save(model.qformer.state_dict(), best_qformer_path)
        log_step(
            "Exported best QFormer weights for stage2",
            f"path={best_qformer_path}, epoch={best_checkpoint['epoch']}, val_loss={best_checkpoint['val_loss']:.4f}",
        )
        if getattr(model, "has_llm_align", False):
            best_align_proj_path = os.path.join(
                outdir, cfg.get("best_align_proj_weights_name", "qformer_stage1_best_align_proj.pth")
            )
            torch.save(model.llm_align_proj.state_dict(), best_align_proj_path)
            log_step(
                "Exported aligned projection for stage2 warm-start",
                f"path={best_align_proj_path}",
            )
    else:
        log_step("Skipped best QFormer export", "No best checkpoint was selected during training")
    
    return model


def main():
    cfg = Config(parse_args())
    stage1_cfg = cfg.run_cfg.get("qformer_stage1")
    if stage1_cfg is None:
        raise KeyError("Missing 'run.qformer_stage1' section in configuration.")

    required_keys = [
        "batch_size",
        "num_workers",
        "embedding_size",
        "user_num",
        "item_num",
        "num_queries",
        "num_heads",
        "num_layers",
        "qformer_output_dim",
        "lr",
        "w_itc",
        "w_itm",
        "w_itg",
        "w_ii",
        "tau_itc",
        "tau_ii",
        "weight_decay",
        "debug_batch",
        "debug_batch_max_steps",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "best_checkpoint_name",
        "best_qformer_weights_name",
        "log_epoch",
        "epoch",
        "pretrained_rec_path",
        "freeze_rec",
        "output_dir",
        "seed",
    ]
    missing_keys = [key for key in required_keys if key not in stage1_cfg]
    if missing_keys:
        raise KeyError(
            "Missing required keys in 'run.qformer_stage1': " + ", ".join(missing_keys)
        )

    first_dataset_key = list(cfg.datasets_cfg.keys())[0]
    stage1_cfg.data_dir = cfg.datasets_cfg[first_dataset_key].path

    train_qformer_stage1_representation(stage1_cfg)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Q-Former stage 1 representation model")
    parser.add_argument(
        "--cfg-path",
        default="configs/config.yaml",
        type=str,
        help="Path to the config file.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="Override config settings in key=value format.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

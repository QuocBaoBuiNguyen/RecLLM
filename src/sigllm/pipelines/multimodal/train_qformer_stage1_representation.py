import argparse
import random
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import omegaconf
import os
import numpy as np
from typing import Optional

from sigllm.common import NotebookLogger, EarlyStopping
from sigllm.common.config import Config
from sigllm.datasets.qformer.qformer_alignment_builder import QFormerAlignmentBuilder
from sigllm.datasets.qformer.qformer_alignment_dataset import QFormerAlignmentDataset
from sigllm.models.rec.matrix_factorization import MatrixFactorization
# from sigllm.models.q_former.q_former import QFormer
from sigllm.models.q_former.hf_qformer_adapter import HFQFormerAdapter
from sigllm.models.q_former.text_encoder import LlamaEmbeddingTextEncoder
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


def collate(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = [b[k] for b in batch]
    return out


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


def _init_dataset(cfg, filename: str, shuffle: bool = True):
    """
    Initializes the dataset and dataloader.
    """
    dataset = QFormerAlignmentDataset(filename=filename)
    loader = DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        shuffle=shuffle, 
        collate_fn=collate, 
        num_workers=cfg.num_workers
    )
    return loader


def _init_text_encoder(cfg, device):
    """
    Initializes the frozen LLaMA embedding text encoder.
    """
    text_encoder = LlamaEmbeddingTextEncoder(
        model_name=cfg.llama_model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    
    # Freeze if necessary
    if cfg.freeze_text_encoder:
        for p in text_encoder.parameters():
            p.requires_grad = False
        text_encoder.eval()
        text_encoder.train = disabled_train.__get__(text_encoder, type(text_encoder))

    return text_encoder, getattr(text_encoder, "hidden_size", text_encoder.model.config.hidden_size)


def _init_qformer(cfg, d_model, device):
    """
    Initializes the Q-Former model.
    """
    # return QFormer(
    #     d_cf=cfg.embedding_size,
    #     d_model=d_model,
    #     num_queries=cfg.num_queries,
    #     num_heads=cfg.num_heads,
    #     num_layers=cfg.num_layers
    # ).to(device)
    return HFQFormerAdapter(
        d_cf=cfg.embedding_size, 
        d_model=d_model, 
        num_queries=cfg.num_queries, 
        num_heads=cfg.num_heads, 
        num_layers=cfg.num_layers
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


def _compute_item_text_metrics(i_pos_vec, t_vec, tau_it: float):
    """Compute Top-1 accuracy for the temporary item-text-only stage-1 objective."""
    pos_selected, _ = QRecInstructAlignmentModel.select_query_by_text(i_pos_vec, t_vec)
    pos = QRecInstructAlignmentModel.l2norm(pos_selected)
    text = QRecInstructAlignmentModel.l2norm(t_vec)

    it_logits = (pos @ text.T) / tau_it
    it_labels = torch.arange(pos.size(0), device=pos.device)
    it_predictions = it_logits.argmax(dim=1)
    it_top1 = (it_predictions == it_labels).float().mean()

    return {
        "ui_top1": i_pos_vec.new_zeros(()),
        "it_top1": it_top1,
    }


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


def train_step(
    batch,
    model: QRecInstructAlignmentModel,
    w_ui: float = 1.0,
    w_it: float = 1.0,
    w_ii: float = 1.0,
    tau_ui: float = 0.07,
    tau_ii: float = 0.07,
    tau_it: float = 0.2,
    debug_batch: bool = False,
    enable_user_item: bool = False,
):
    device = batch["i_left"].device

    if debug_batch:
        _log_batch_preview(batch)

    zero = next(model.parameters()).sum() * 0.0
    logs = {
        "L_it": zero,
        "L_ii": zero,
        "L_ui": zero,
        "it_top1": zero.detach(),
        "ii_top1": zero.detach(),
        "ui_top1": zero.detach(),
    }
    losses = []

    item_text_idx = _indices_for_type(batch, "item_text", device)
    if item_text_idx.numel() >= 2:
        item_text_batch = _subset_batch(batch, item_text_idx)
        ins_tok_emb = model.ins_tokens(item_text_batch["instruction"], device)
        item_vec = model.enc_item(item_text_batch["i_left"], ins_tok_emb)
        text_vec = model.text_vec(item_text_batch["text"], device)
        logs["L_it"] = model.loss_item_text(item_vec, text_vec, tau=tau_it)
        logs["it_top1"] = _compute_item_text_metrics(item_vec, text_vec, tau_it)["it_top1"]
        losses.append(w_it * logs["L_it"])

    item_item_idx = _indices_for_type(batch, "item_item", device)
    if item_item_idx.numel() >= 2:
        item_item_batch = _subset_batch(batch, item_item_idx)
        ins_tok_emb = model.ins_tokens(item_item_batch["instruction"], device)
        left_vec = model.enc_item(item_item_batch["i_left"], ins_tok_emb)
        right_vec = model.enc_item(item_item_batch["i_right"], ins_tok_emb)
        logs["L_ii"] = model.loss_item_item_ilm(left_vec, right_vec, tau=tau_ii)
        logs["ii_top1"] = model.multiquery_inbatch_top1(left_vec, right_vec, tau=tau_ii)
        losses.append(w_ii * logs["L_ii"])

    user_item_idx = _indices_for_type(batch, "user_item", device)
    if enable_user_item and w_ui > 0.0 and user_item_idx.numel() >= 2:
        user_item_batch = _subset_batch(batch, user_item_idx)
        ins_tok_emb = model.ins_tokens(user_item_batch["instruction"], device)
        user_vec = model.enc_user(user_item_batch["u"], ins_tok_emb)
        item_vec = model.enc_item(user_item_batch["i_left"], ins_tok_emb)
        logs["L_ui"] = model.loss_multiquery_inbatch_symmetric(user_vec, item_vec, tau=tau_ui)
        logs["ui_top1"] = model.multiquery_inbatch_top1(user_vec, item_vec, tau=tau_ui)
        losses.append(w_ui * logs["L_ui"])

    loss = sum(losses, zero)
    return loss, logs


def evaluate_loss(
    model,
    loader,
    w_ui=1.0,
    w_it=1.0,
    w_ii=1.0,
    tau_ui=0.07,
    tau_ii=0.07,
    tau_it=0.2,
    enable_user_item: bool = False,
):
    model.eval()
    device = next(model.parameters()).device
    totals = {
        "loss": 0.0,
        "L_it": 0.0,
        "L_ii": 0.0,
        "L_ui": 0.0,
        "it_top1": 0.0,
        "ii_top1": 0.0,
        "ui_top1": 0.0,
    }
    steps = 0

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            loss, logs = train_step(
                batch,
                model,
                w_ui=w_ui,
                w_it=w_it,
                w_ii=w_ii,
                tau_ui=tau_ui,
                tau_ii=tau_ii,
                tau_it=tau_it,
                enable_user_item=enable_user_item,
            )
            totals["loss"] += loss.item()
            for key in logs:
                totals[key] += logs[key].item()
            steps += 1

    if steps == 0:
        return {key: 0.0 for key in totals}
    return {key: value / steps for key, value in totals.items()}


def _save_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    epoch,
    val_logs,
):
    trainable_param_names = {
        name for name, param in model.named_parameters() if param.requires_grad
    }
    trainable_state_dict = {
        name: param
        for name, param in model.state_dict().items()
        if name in trainable_param_names
    }
    torch.save(
        {
            "epoch": epoch,
            "model": trainable_state_dict,
            "optimizer": optimizer.state_dict(),
            **{f"val_{key}": value for key, value in val_logs.items()},
        },
        checkpoint_path,
    )

def _load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def train_qformer_stage1_representation(cfg):
    set_seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = _init_dataset(cfg, filename=os.path.join(cfg.data_dir, "train_qformer_ood2.pkl"), shuffle=True)
    val_loader = _init_dataset(cfg, filename=os.path.join(cfg.data_dir, "valid_qformer_ood2.pkl"), shuffle=False)
    test_loader = _init_dataset(cfg, filename=os.path.join(cfg.data_dir, "test_qformer_ood2.pkl"), shuffle=False)
    
    mf = _init_rec_model(cfg, device)
    text_encoder, d_model = _init_text_encoder(cfg, device)
    qformer = _init_qformer(cfg, d_model, device)

    model = QRecInstructAlignmentModel(mf, qformer, text_encoder).to(device)
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

    for epoch in range(cfg.epoch):
        model.train()
        train_totals = {
            "loss": 0.0,
            "L_it": 0.0,
            "L_ii": 0.0,
            "L_ui": 0.0,
            "it_top1": 0.0,
            "ii_top1": 0.0,
            "ui_top1": 0.0,
        }
        train_steps = 0
        for batch in train_loader:
            batch = _move_batch_to_device(batch, device)
            opt.zero_grad()

            loss, logs = train_step(
                batch,
                model,
                w_ui=cfg.w_ui,
                w_it=cfg.w_it,
                w_ii=cfg.w_ii,
                tau_ui=cfg.tau_ui,
                tau_ii=cfg.tau_ii,
                tau_it=cfg.tau_it,
                debug_batch=cfg.debug_batch and epoch == 0 and train_steps < cfg.debug_batch_max_steps,
                enable_user_item=bool(cfg.get("include_user_item", False)),
            )
            loss.backward()
            opt.step()
            
            train_totals["loss"] += loss.item()
            for key in logs:
                train_totals[key] += logs[key].item()
            train_steps += 1    

        if (epoch + 1) % cfg.log_epoch == 0:
            avg_train = {
                key: value / train_steps if train_steps > 0 else 0.0
                for key, value in train_totals.items()
            }
            val_logs = evaluate_loss(
                model,
                val_loader,
                w_ui=cfg.w_ui,
                w_it=cfg.w_it,
                w_ii=cfg.w_ii,
                tau_ui=cfg.tau_ui,
                tau_ii=cfg.tau_ii,
                tau_it=cfg.tau_it,
                enable_user_item=bool(cfg.get("include_user_item", False)),
            )
            print(
                f"epoch {epoch+1} | "
                f"Train Loss={avg_train['loss']:.4f} L_it={avg_train['L_it']:.4f} "
                f"L_ii={avg_train['L_ii']:.4f} L_ui={avg_train['L_ui']:.4f} "
                f"IT@1={avg_train['it_top1']:.4f} II@1={avg_train['ii_top1']:.4f} "
                f"UI@1={avg_train['ui_top1']:.4f} | "
                f"Val Loss={val_logs['loss']:.4f} L_it={val_logs['L_it']:.4f} "
                f"L_ii={val_logs['L_ii']:.4f} L_ui={val_logs['L_ui']:.4f} "
                f"IT@1={val_logs['it_top1']:.4f} II@1={val_logs['ii_top1']:.4f} "
                f"UI@1={val_logs['ui_top1']:.4f} | "
                f"w_it={cfg.w_it:.3f} w_ii={cfg.w_ii:.3f} w_ui={cfg.w_ui:.3f} "
                f"tau_it={cfg.tau_it:.3f} tau_ii={cfg.tau_ii:.3f} tau_ui={cfg.tau_ui:.3f}"
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
                f"it_top1={best_checkpoint.get('val_it_top1', 0.0):.4f}, "
                f"ii_top1={best_checkpoint.get('val_ii_top1', 0.0):.4f}, "
                f"ui_top1={best_checkpoint.get('val_ui_top1', 0.0):.4f}"
            ),
        )

    # Final Test
    log_step("Evaluating on Test Set")
    test_logs = evaluate_loss(
        model,
        test_loader,
        w_ui=cfg.w_ui,
        w_it=cfg.w_it,
        w_ii=cfg.w_ii,
        tau_ui=cfg.tau_ui,
        tau_ii=cfg.tau_ii,
        tau_it=cfg.tau_it,
        enable_user_item=bool(cfg.get("include_user_item", False)),
    )
    log_step(
        "Test results",
        (
            f"loss={test_logs['loss']:.4f}, l_it={test_logs['L_it']:.4f}, "
            f"l_ii={test_logs['L_ii']:.4f}, l_ui={test_logs['L_ui']:.4f}, "
            f"it@1={test_logs['it_top1']:.4f}, ii@1={test_logs['ii_top1']:.4f}, "
            f"ui@1={test_logs['ui_top1']:.4f}"
        ),
    )

    if best_checkpoint is not None:
        best_qformer_path = os.path.join(outdir, cfg.best_qformer_weights_name)
        torch.save(model.qformer.state_dict(), best_qformer_path)
        log_step(
            "Exported best QFormer weights for stage2",
            f"path={best_qformer_path}, epoch={best_checkpoint['epoch']}, val_loss={best_checkpoint['val_loss']:.4f}",
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
        "lr",
        "w_ui",
        "w_it",
        "w_ii",
        "tau_ui",
        "tau_ii",
        "tau_it",
        "weight_decay",
        "debug_batch",
        "debug_batch_max_steps",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "best_checkpoint_name",
        "best_qformer_weights_name",
        "log_epoch",
        "epoch",
        "llama_model_name",
        "pretrained_rec_path",
        "freeze_rec",
        "freeze_text_encoder",
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
    item_pair_window = int(stage1_cfg.get("item_pair_window", 2))
    max_item_item_pairs = stage1_cfg.get("max_item_item_pairs", None)
    max_user_item_pairs = stage1_cfg.get("max_user_item_pairs", None)
    include_user_item = bool(stage1_cfg.get("include_user_item", False))

    QFormerAlignmentBuilder.build_qformer_alignment_samples(        
        input_pkl_path=os.path.join(stage1_cfg.data_dir, "train_ood2.pkl"),
        output_path=os.path.join(stage1_cfg.data_dir, "train_qformer_ood2.pkl"),
        seed=stage1_cfg.seed,
        item_pair_window=item_pair_window,
        max_item_item_pairs=max_item_item_pairs,
        max_user_item_pairs=max_user_item_pairs,
        include_user_item=include_user_item,
    )
    QFormerAlignmentBuilder.build_qformer_alignment_samples(        
        input_pkl_path=os.path.join(stage1_cfg.data_dir, "valid_ood2.pkl"),
        output_path=os.path.join(stage1_cfg.data_dir, "valid_qformer_ood2.pkl"),
        seed=stage1_cfg.seed,
        item_pair_window=item_pair_window,
        max_item_item_pairs=max_item_item_pairs,
        max_user_item_pairs=max_user_item_pairs,
        include_user_item=include_user_item,
    )
    QFormerAlignmentBuilder.build_qformer_alignment_samples(        
        input_pkl_path=os.path.join(stage1_cfg.data_dir, "test_ood2.pkl"),
        output_path=os.path.join(stage1_cfg.data_dir, "test_qformer_ood2.pkl"),
        seed=stage1_cfg.seed,
        item_pair_window=item_pair_window,
        max_item_item_pairs=max_item_item_pairs,
        max_user_item_pairs=max_user_item_pairs,
        include_user_item=include_user_item,
    )

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

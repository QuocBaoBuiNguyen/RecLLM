import argparse
import os
import numpy as np
import time as time
import random
import omegaconf
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from typing import Optional
from torch.utils.data import DataLoader

from sigllm.common import NotebookLogger, EarlyStopping
from sigllm.common.config import Config
from sigllm.models.rec.matrix_factorization import MatrixFactorization

LOGGER = NotebookLogger.rich_logger("sigllm.train_rec_baseline")

def log_step(title: str, detail: Optional[str] = None) -> None:
    """Emit a compact log line with optional detail string."""

    message = title if detail is None else f"{title} | {detail}"
    LOGGER.info(message)

def calculate_user_auc(user_ids, y_pred, y_true):
    """Calculate User AUC (uAUC) for recommendation tasks."""
    
    y_pred = np.asarray(y_pred).squeeze()
    y_true = np.asarray(y_true).squeeze()
    user_ids = np.asarray(user_ids)

    start_time = time.time()

    users, inverse, counts = np.unique(user_ids, return_inverse=True, return_counts=True)
    sort_indices = np.argsort(inverse)

    auc_list = []
    computed_users = []
    only_one_interaction = 0
    only_one_class = 0
    current_pos = 0

    for i, user_id in enumerate(users):
        user_count = counts[i]
        user_indices = sort_indices[current_pos:current_pos + user_count]

        user_y_true = y_true[user_indices]
        user_y_pred = y_pred[user_indices]

        current_pos += user_count

        if user_count < 2:
            only_one_interaction += 1
            continue

        if np.all(user_y_true == user_y_true[0]):
            only_one_class += 1
            continue

        score = roc_auc_score(user_y_true, user_y_pred)
        auc_list.append(score)
        computed_users.append(user_id)

    auc_array = np.array(auc_list)
    avg_uauc = auc_array.mean()
    log_step("Users with only one interaction", str(only_one_interaction))
    log_step("Users with only one class", str(only_one_class))
    log_step("Computed user AUC count", str(len(auc_list)))
    log_step("Average User AUC", f"{avg_uauc:.6f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    log_step("Time taken to calculate user AUC", f"{elapsed_time:.2f} seconds")

    return avg_uauc, computed_users, auc_array

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_model_predictions(model, data_loader, device):
    model.eval()
    all_preds, all_labels, all_users = [], [], []

    with torch.no_grad():
        for batch_data in data_loader:
            batch_data = batch_data.to(device)
            # Forward pass
            ui_matching = model(batch_data[:, 0].long(), batch_data[:, 1].long())
            
            all_users.append(batch_data[:, 0].cpu().numpy())
            all_preds.append(ui_matching.detach().cpu().numpy())
            all_labels.append(batch_data[:, -1].cpu().numpy())
            
    return (np.concatenate(all_users), 
            np.concatenate(all_preds), 
            np.concatenate(all_labels))

def parse_args():
    parser = argparse.ArgumentParser(description="Train MF baseline for recommendation")
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


def train_baseline_model(
    train_config,
    data_dir,
    save_file=None,
    need_train=True,
    warm_or_cold=None,
    seed=None,
    train_file="train_ood2.pkl",
):
    # 1. Setup Environment
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load and Filter Data
    # `train_file` lets the MF teacher be pretrained on an extended window
    # (e.g. train_ext_ood2.pkl = months 0-23) while valid/test stay on the
    # untouched OOD split — isolating CF quality as the only changed variable.
    log_step("MF train file", train_file)
    # Guard the dense-teacher checkpoint: an "_ext" save path with a non-ext
    # train pool would overwrite mf_ext_model.pth with a sparse-trained MF —
    # silently poisoning every pipeline run that loads it as the CF teacher.
    if save_file and "ext" in os.path.basename(save_file) and "ext" not in train_file:
        raise ValueError(
            f"save_file={save_file!r} looks like the extended-pool teacher but "
            f"train_file={train_file!r} is not the extended pool. Refusing to "
            "overwrite; set save_file and train_file together."
        )
    train_data = pd.read_pickle(os.path.join(data_dir, train_file))[['uid','iid','label']].values
    valid_data = pd.read_pickle(os.path.join(data_dir, "valid_ood2.pkl"))[['uid','iid','label']].values
    test_data = pd.read_pickle(os.path.join(data_dir, "test_ood2.pkl"))[['uid','iid','label']].values

    user_num = max(train_data[:,0].max(), valid_data[:,0].max(), test_data[:,0].max()) + 1
    item_num =  max(train_data[:,1].max(), valid_data[:,1].max(), test_data[:,1].max()) + 1
    log_step("User num:", str(user_num))
    log_step("Item num:", str(item_num))

    if warm_or_cold is not None:
        if warm_or_cold == 'warm':
            test_data = pd.read_pickle(os.path.join(data_dir, "test_warm_cold_ood2.pkl"))[['uid','iid','label', 'warm']]
            test_data = test_data[test_data['warm'].isin([1])][['uid','iid','label']].values
            log_step("warm data size:", str(test_data.shape[0]))
        else:
            test_data = pd.read_pickle(os.path.join(data_dir, "test_warm_cold_ood2.pkl"))[['uid','iid','label', 'cold']]
            test_data = test_data[test_data['cold'].isin([1])][['uid','iid','label']].values
            log_step("cold data size:", str(test_data.shape[0]))

    train_loader = DataLoader(train_data, batch_size=train_config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=train_config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_data, batch_size=train_config['batch_size'], shuffle=False)
    
    # 3. Model & Optimizer Initialization
    mf_config = omegaconf.OmegaConf.create({
        "user_num": int(user_num),
        "item_num": int(item_num),
        "embedding_size": int(train_config['embedding_size'])
    })

    model = MatrixFactorization(mf_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'], weight_decay=train_config['wd'])
    # Select the MF teacher by valid uAUC (the headline ranking metric), matching
    # the pipeline's agg_metrics=uauc. AUC and uAUC diverge, so selecting on AUC
    # would save an MF checkpoint that is not the best per-user ranker.
    stopper = EarlyStopping(ref_metric='valid_uauc', monitor_mode='max', patience=train_config['patience'])
    criterion = nn.BCEWithLogitsLoss()

    #4. Inference only
    if not need_train:
        log_step("Starting evaluation only mode.")
        model.load_state_dict(torch.load(save_file))
        
        v_users, v_preds, v_labels = get_model_predictions(model, valid_loader, device)
        valid_auc = roc_auc_score(v_labels, v_preds)
        valid_uauc, _, _ = calculate_user_auc(v_users, v_preds, v_labels)
        
        t_users, t_preds, t_labels = get_model_predictions(model, test_loader, device)
        test_auc = roc_auc_score(t_labels, t_preds)
        test_uauc, _, _ = calculate_user_auc(t_users, t_preds, t_labels)

        threshold = 0.1
        acc = ((v_preds >= threshold) == v_labels).mean()
        
        log_step(f"Valid AUC: {valid_auc:.4f}, Valid uAUC: {valid_uauc:.4f}, Test AUC: {test_auc:.4f}, Test uAUC: {test_uauc:.4f}, Acc: {acc:.4f}")
        return
    
    #5. Training loop
    for epoch in range(train_config['epoch']):
        model.train()
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            
            ui_matching = model(batch_data[:, 0].long(), batch_data[:, 1].long())
            loss = criterion(ui_matching.squeeze(), batch_data[:, -1].float())

            loss.backward()
            optimizer.step()

        if epoch % train_config['eval_epoch'] == 0:
            v_users, v_preds, v_labels = get_model_predictions(model, valid_loader, device)
            valid_auc = roc_auc_score(v_labels, v_preds)
            valid_uauc, _, _ = calculate_user_auc(v_users, v_preds, v_labels)
            
            t_users, t_preds, t_labels = get_model_predictions(model, test_loader, device)
            test_auc = roc_auc_score(t_labels, t_preds)
            test_uauc, _, _ = calculate_user_auc(t_users, t_preds, t_labels)

            threshold = 0.1
            acc = ((v_preds >= threshold) == v_labels).mean()
            
            log_step(f"Epoch {epoch}: Valid AUC: {valid_auc:.4f}, Valid uAUC: {valid_uauc:.4f}, Test AUC: {test_auc:.4f}, Test uAUC: {test_uauc:.4f}, Acc: {acc:.4f}")

            metrics = {
                'valid_auc': valid_auc, 'valid_uauc': valid_uauc,
                'test_auc': test_auc, 'test_uauc': test_uauc, 'epoch': epoch
            }

            improved = stopper.update(metrics)

            if improved:
                log_step(f"New best model found at epoch {epoch} with Valid uAUC: {valid_uauc:.4f}")
                if save_file is not None:
                    torch.save(model.state_dict(), save_file)
                    log_step(f"Model saved to {save_file}")

            if stopper.should_stop:
                log_step("Early stopping triggered. Ending training.")
                break

            if epoch > 500 and stopper.best_metric_val < 0.52:
                log_step("Training fails to converge (Valid uAUC < 0.52 at epoch 500)")
                break

    # 6. Final Logging
    final_log = f"Train Config: {train_config}\nBest Results: {stopper.best_full_metric}"
    log_step(final_log)

    return stopper.best_full_metric


def main():
    cfg = Config(parse_args())
    baseline_cfg = cfg.run_cfg.get("rec_baseline")
    if baseline_cfg is None:
        raise KeyError("Missing 'run.rec_baseline' section in configuration.")

    first_dataset_key = list(cfg.datasets_cfg.keys())[0]
    required_keys = [
        "save_file",
        "need_train",
        "warm_or_cold",
        "seed",
        "lr",
        "wd",
        "embedding_size",
        "epoch",
        "eval_epoch",
        "patience",
        "batch_size",
    ]
    missing_keys = [key for key in required_keys if key not in baseline_cfg]
    if missing_keys:
        raise KeyError(
            "Missing required keys in 'run.rec_baseline': " + ", ".join(missing_keys)
        )

    train_config = baseline_cfg
    data_dir = cfg.datasets_cfg[first_dataset_key].path
    save_file = baseline_cfg["save_file"]
    need_train = bool(baseline_cfg["need_train"])
    warm_or_cold = baseline_cfg["warm_or_cold"]
    seed = int(baseline_cfg["seed"])
    # Optional: pretrain MF on an extended window. Defaults to the original
    # train_ood2.pkl so existing runs are unchanged.
    train_file = baseline_cfg.get("train_file", "train_ood2.pkl")

    log_step("Loaded config", f"cfg_path={cfg.args.cfg_path}")
    log_step("Dataset dir", str(data_dir))
    log_step("Dataset key", str(first_dataset_key))
    log_step("Baseline mode", f"need_train={need_train}, warm_or_cold={warm_or_cold}")

    if save_file is not None:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)

    train_baseline_model(
        train_config=train_config,
        data_dir=data_dir,
        save_file=save_file,
        need_train=need_train,
        warm_or_cold=warm_or_cold,
        seed=seed,
        train_file=train_file,
    )

if __name__ == "__main__":
    main()

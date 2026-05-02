import argparse
from html import parser
import os
import random
import pandas as pd
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from torch.distributed.elastic.multiprocessing.errors import record

from sigllm import tasks
from sigllm.common import registry
from sigllm.common.config import Config
from sigllm.common.utils import now
from sigllm.common.dist_utils import get_rank, init_distributed_mode
from datetime import datetime
from sigllm.runners.runner_base_rec import RecRunnerBase

def parse_args():
    parser = argparse.ArgumentParser(description="Train LLM for recommendation")
    parser.add_argument("--cfg-path", default="/content/SigLLM/configs/config.yaml",type=str, required=True, help="Path to the config file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

@record
def main():
    job_id = now()
    cfg = Config(parse_args())
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)

    task = tasks.setup_task(cfg=cfg)

    datasets = task.build_datasets(cfg=cfg)

    first_dataset_key = list(cfg.datasets_cfg.keys())[0]
    data_dir = cfg.datasets_cfg[first_dataset_key].path
    train_ = pd.read_pickle(os.path.join(data_dir, "train_ood2.pkl"))
    valid_ = pd.read_pickle(os.path.join(data_dir, "valid_ood2.pkl"))
    test_ = pd.read_pickle(os.path.join(data_dir, "test_ood2.pkl"))
    user_num = max(train_.uid.max(), valid_.uid.max(), test_.uid.max()) + 1
    item_num = max(train_.iid.max(), valid_.iid.max(), test_.iid.max()) + 1
    # TEMP_DISABLED_USER_CF: keep user_num for dataset/eval and pretrained MF loading,
    # but QRecLLM no longer injects mf.user_encoder(UserID) into the LLM prompt.
    cfg.model_cfg.rec_config.user_num = int(user_num)
    cfg.model_cfg.rec_config.item_num = int(item_num)
    cfg.pretty_print()

    model = task.build_model(cfg=cfg)
    
    runner = task.build_runner(cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)

    runner.train()

if __name__ == "__main__":
    main()

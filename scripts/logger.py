#scripts/logger.py

import os
import wandb

class Logger:
    def __init__(self, experiment_name, project="mm_ff_project",
                 log_dir="/users/adgs898/sharedscratch/multimodal_ff_project/.wandb_offline"):
        os.makedirs(log_dir, exist_ok=True)
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DIR"] = log_dir
        self.logger = wandb.init(project=project, name=experiment_name)

    def log(self, metrics: dict, step: int | None = None):
        if step is not None:
            metrics = dict(metrics)
            metrics["step"] = step
        self.logger.log(metrics)

    def watch(self, model):
        try:
            self.logger.watch(model, log="gradients", log_freq=500)
        except Exception:
            pass

    def finish(self):
        self.logger.finish()
from transformers import Trainer
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
from transformers import PreTrainedModel
import torch
import torch.nn as nn
from collections import defaultdict

class CustomTrainer(Trainer):
    _stored_metrics = defaultdict(lambda: defaultdict(list))

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)
    
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        loss, others = model(**inputs)
        acc = (inputs["class_label"] == others["pred"]).float().mean()
        metrics = {
            "router_z_loss": others.get("router_z_loss", 0.),
            "load_balancing_loss": others.get("balancing_loss", 0.),
            "ce_loss": others["ce_loss"],
            "accuracy" : acc
        }

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, metrics = self.compute_loss(model, inputs, return_outputs=True)
            metrics = {"eval_"+k:v for k,v in metrics.items()}

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="eval")

        # if prediction_loss_only:
        return (loss.detach(), None, None)


        # return (loss.detach(), logits, labels)
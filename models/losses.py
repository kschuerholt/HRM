from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn


IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(
        logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1
    ).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(
        logits.to(torch.float32).reshape(-1, logits.shape[-1]),
        labels.to(torch.long).reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).reshape(labels.shape)


class ACTLossHead(nn.Module):
    """
    Loss head for the ACT model.
    # NOTE: for some weird reason, they use the loss class as wrapper around the model. So this is really the model that does inference and returns the loss
    """

    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        # Correctness
        with torch.no_grad():
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            # Metrics (only for halted sequences - those that just completed their problems)
            # NOTE: valid_metrics filters to sequences that halted AND have valid labels
            # This ensures we only evaluate sequences that just finished working on a problem
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (
                    valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)
                ).sum(),
                # NOTE: Steps metric shows how many batches sequences spent on their problems
                # Higher values = harder problems that required more temporal computation
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

            # Enhanced H/L Module Metrics
            if hasattr(self.model, "_hrm_metrics"):
                hrm_metrics = self.model._hrm_metrics

                # H residual norms (convert tensors to scalars here)
                if hrm_metrics.get("h_residual_norms"):
                    h_residuals = [
                        t.item() for t in hrm_metrics["h_residual_norms"]
                    ]  # Convert tensors to floats
                    if h_residuals:
                        metrics.update(
                            {
                                "h_residual_mean": torch.tensor(
                                    sum(h_residuals) / len(h_residuals), device="cuda"
                                ),
                                "h_residual_max": torch.tensor(max(h_residuals), device="cuda"),
                                "h_residual_min": torch.tensor(min(h_residuals), device="cuda"),
                            }
                        )

                        # Track residual evolution over H cycles
                        for i, residual in enumerate(h_residuals):
                            metrics[f"h_residual_cycle_{i}"] = torch.tensor(residual, device="cuda")

                # L residual norms (convert tensors to scalars here)
                if hrm_metrics.get("l_residual_norms"):
                    l_residuals = [
                        t.item() for t in hrm_metrics["l_residual_norms"]
                    ]  # Convert tensors to floats
                    if l_residuals:
                        metrics.update(
                            {
                                "l_residual_mean": torch.tensor(
                                    sum(l_residuals) / len(l_residuals), device="cuda"
                                ),
                                "l_residual_max": torch.tensor(max(l_residuals), device="cuda"),
                                "l_residual_min": torch.tensor(min(l_residuals), device="cuda"),
                            }
                        )

                        # Track residual evolution over L cycles
                        for i, residual in enumerate(l_residuals):
                            metrics[f"l_residual_cycle_{i}"] = torch.tensor(residual, device="cuda")

                # Activation norms (convert tensors to scalars here)
                if hrm_metrics.get("h_activation_norms"):
                    h_activations = [
                        t.item() for t in hrm_metrics["h_activation_norms"]
                    ]  # Convert tensors to floats
                    if h_activations:
                        metrics["h_activation_mean"] = torch.tensor(
                            sum(h_activations) / len(h_activations), device="cuda"
                        )

                if hrm_metrics.get("l_activation_norms"):
                    l_activations = [
                        t.item() for t in hrm_metrics["l_activation_norms"]
                    ]  # Convert tensors to floats
                    if l_activations:
                        metrics["l_activation_mean"] = torch.tensor(
                            sum(l_activations) / len(l_activations), device="cuda"
                        )

                # Cycle execution counts (simple integers)
                h_cycles = hrm_metrics.get("h_cycles_executed", 0)
                l_cycles = hrm_metrics.get("l_cycles_executed", 0)
                metrics.update(
                    {
                        "h_cycles_executed": torch.tensor(
                            float(h_cycles), device="cuda", dtype=torch.float32
                        ),
                        "l_cycles_executed": torch.tensor(
                            float(l_cycles), device="cuda", dtype=torch.float32
                        ),
                    }
                )

        # Losses
        # FIXME: Assuming the batch is always full
        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum"
        )

        # Loss component tracking with weights (clean approach)
        metrics.update(
            {
                "lm_loss": lm_loss.detach(),
                "q_halt_loss": q_halt_loss.detach(),
                "lm_loss_weight": torch.tensor(1.0, device="cuda", dtype=torch.float32),
                "q_halt_loss_weight": torch.tensor(0.5, device="cuda", dtype=torch.float32),
            }
        )

        # Q continue (bootstrapping target loss)
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum"
            )

            metrics["q_continue_loss"] = q_continue_loss.detach()
            metrics["q_continue_loss_weight"] = torch.tensor(0.5, device="cuda", dtype=torch.float32)

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return (
            new_carry,
            lm_loss + 0.5 * (q_halt_loss + q_continue_loss),
            metrics,
            detached_outputs,
            new_carry.halted.all(),  # all_finish: True when ALL sequences in batch have halted
        )

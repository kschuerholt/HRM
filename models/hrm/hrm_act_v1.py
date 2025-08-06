from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"


class HierarchicalReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        # Self Attention
        # NOTE:
        # # resiudal self attention
        # # RMS-norm as more efficient replacement for LayerNorm
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        # Fully Connected
        # # Residual Feedforward step, so far so common
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    """
    Wrapper around a list of Self-Attention blocks. Nothing fancy, except for the input injection at the very beginning.
    """

    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV1Block]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # NOTE: What's this input injection?
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    """
    Summary:
    - takes data, computes embeddings and passes through the H_level and L_level, both of which are just transformer blocks.
    - key 1: several gradient-free iterations of H_level and L_level, followed by one gradient-enabled iteration of H_level and L_level.
    - key 2: the z_H is input `injection` to the L_level, and z_L is input `injection` to the H_level. injection is basically just an addition to the input.
    - key 2.5: z_L receives several steps / updates (faster), while z_H receives fewer.
    - key 3: the output is computed from z_H, and sliced to remove the puzzle embeddings.
    - key 4: the q_head is a linear layer, that takes the first token of z_H and outputs a 2-dimensional tensor.
    """

    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        # NOTE: - this is just an embedding layer
        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        # NOTE: - lm_head + q head, to linear layers
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )
        else:
            raise NotImplementedError()

        # Reasoning Layers
        ## NOTE: we instantiate two modules, both of which are just transformer blocks. Called H_level and L_level.
        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.H_layers)]
        )
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)]
        )

        # Initial states
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        ## NOTE: - self.embed_tokens makes a lot of sense, but whate are puzzle_identifiers?
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2
            )

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
            z_L=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_ACTV1InnerCarry):
        ## NOTE: - reset_flag is a boolean tensor, that is used to reset the carry. Affects both z_H and z_L, but not the steps or halted.
        ## NOTE: - self.H_init and self.L_init are the initial states for z_H and z_L.
        ## NOTE: effictively, this finds the sequences with a 'reset_flag' and replaces the z_H and z_L with the initial states.
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(
        self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Takes a 'carry' and a a batch of inputs as a dict. Returns a carry, a logits tensor, and a tuple of q_logits.
        Args:
            carry: The carry from the previous step.
            batch: A dictionary containing the `inputs` and `puzzle_identifiers`.
        Returns:
            A tuple containing the new carry, the logits tensor, and a tuple of q_logits.
        """
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding - # compute embeddings
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Initialize metrics tracking (only during training to avoid eval performance impact)
        track_metrics = self.training
        h_residual_norms = []
        l_residual_norms = []
        h_activation_norms = []
        l_activation_norms = []

        # Forward iterations
        # NOTE: gradient-free forward passes through H-level and L-level to update z_H and z_L.
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            # Track initial activation norms (only during training)
            if track_metrics:
                h_activation_norms.append(torch.norm(z_H, dim=-1).mean().detach().cpu())
                l_activation_norms.append(torch.norm(z_L, dim=-1).mean().detach().cpu())

            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        ## NOTE: - z_H + input_embeddings are the 'injection' to the z_L, z_L is the input to z_L
                        z_L_prev = z_L
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

                        # Track L-level residual norm (only during training)
                        if track_metrics:
                            l_residual_norms.append(torch.norm(z_L - z_L_prev, dim=-1).mean().detach().cpu())
                            l_activation_norms.append(torch.norm(z_L, dim=-1).mean().detach().cpu())

                if not (_H_step == self.config.H_cycles - 1):
                    ## NOTE: - z_L is the injection to the H_Level, z_H is the input to the H_Level
                    z_H_prev = z_H
                    z_H = self.H_level(z_H, z_L, **seq_info)

                    # Track H-level residual norm (only during training)
                    if track_metrics:
                        h_residual_norms.append(torch.norm(z_H - z_H_prev, dim=-1).mean().detach().cpu())
                        h_activation_norms.append(torch.norm(z_H, dim=-1).mean().detach().cpu())

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad
        ## NOTE: One further step of z_L and z_H through the H_Level and L_Level, this time with gradients.
        z_L_prev_grad = z_L
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

        z_H_prev_grad = z_H
        z_H = self.H_level(z_H, z_L, **seq_info)

        # Track final gradient step residuals (only during training)
        if track_metrics:
            l_residual_norms.append(torch.norm(z_L - z_L_prev_grad, dim=-1).mean().detach().cpu())
            h_residual_norms.append(torch.norm(z_H - z_H_prev_grad, dim=-1).mean().detach().cpu())

            # Store metrics in a global dict for the loss head to access
            if not hasattr(self, "_hrm_metrics"):
                self._hrm_metrics = {}

            self._hrm_metrics.update(
                {
                    "h_residual_norms": h_residual_norms,
                    "l_residual_norms": l_residual_norms,
                    "h_activation_norms": h_activation_norms,
                    "l_activation_norms": l_activation_norms,
                    "h_cycles_executed": len(h_residual_norms),
                    "l_cycles_executed": len(l_residual_norms),
                }
            )
        else:
            # Clear metrics during eval to avoid stale data
            if hasattr(self, "_hrm_metrics"):
                self._hrm_metrics = {}

        # LM Outputs
        # the carry is updated, but detached
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=z_H.detach(), z_L=z_L.detach()
        )  # New carry no grad
        ## NOTE: - output is computed from z_H, and sliced to remove the puzzle embeddings?
        output = self.lm_head(z_H)[:, self.puzzle_emb_len :]

        # Q head
        # NOTE: - q_head is a linear layer, that takes the first token of z_H and outputs a 2-dimensional tensor.
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(
                batch_size
            ),  # Empty is expected, will be reset since all sequences start halted
            steps=torch.zeros((batch_size,), dtype=torch.int32),  # Step counter per sequence
            halted=torch.ones(
                (batch_size,), dtype=torch.bool
            ),  # Default to halted - all sequences start with fresh problems from first batch
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        """
        Questions:
        - how is the carry initialized? Are all sequences halted? If so, how are new_steps computed?
        """
        # Update data, carry (temporal multiplexing across batches)
        # NOTE: This implements the core ACT mechanism where sequences work on problems for variable time
        # - Halted sequences get reset and receive fresh problems from the new batch
        # - Non-halted sequences continue working on their existing problems across batches
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        # NOTE: Step counter tracks how many batches each sequence has spent on its current problem
        # Halted sequences reset to 0 (starting fresh), non-halted sequences increment their counter
        new_steps = torch.where(carry.halted, 0, carry.steps)

        # NOTE: Critical temporal multiplexing logic - mix old and new data based on halting state
        # halted[i]=True:  use batch[k][i] (fresh problem from new batch)
        # halted[i]=False: use v[i] (continue same problem from carry.current_data)
        # This allows hard problems to persist across batches while easy ones cycle through quickly
        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

        # Forward inner model # NOTE: this is the actual forward pass through the H_level and L_level.
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data
        )

        # NOTE: Outputs from inner model - logits for next token prediction and Q-values for halting decisions
        outputs = {"logits": logits, "q_halt_logits": q_halt_logits, "q_continue_logits": q_continue_logits}

        # NOTE: Halting decision logic - determines which sequences should stop working on their current problems
        # This operates per-sequence: each position in the batch can independently decide to halt
        with torch.no_grad():
            # Step counter: increment for all sequences (tracks batches spent on current problem)
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            # Default: force halt when max steps reached (prevents infinite computation)
            halted = is_last_step

            # Training mode: use Q-learning to decide halting + exploration
            if self.training and (self.config.halt_max_steps > 1):
                # Q-learning halt signal: sequence halts if Q(halt) > Q(continue)
                # NOTE: This learns to predict when the sequence has "solved" its current problem
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration: randomly force some sequences to halt early or late
                # NOTE: Prevents over-fitting to specific timing patterns during training
                min_halt_steps = (
                    torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob
                ) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)

                # Final halting decision: combine Q-learning decision with exploration
                # NOTE: Sequences halt when they want to AND they've spent minimum exploration time
                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q
                # NOTE: No replay buffer and target networks for computing target Q-value.
                # As batch_size is large, there're many parallel envs.
                # Similar concept as PQN https://arxiv.org/abs/2407.04811
                next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]

                # NOTE: No idea what this is. maybe the target q value is trained to take on a specific value at the end?
                outputs["target_q_continue"] = torch.sigmoid(
                    torch.where(
                        is_last_step,
                        next_q_halt_logits,
                        torch.maximum(next_q_halt_logits, next_q_continue_logits),
                    )
                )

        return HierarchicalReasoningModel_ACTV1Carry(
            new_inner_carry, new_steps, halted, new_current_data
        ), outputs

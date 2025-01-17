import math
import re
from collections import OrderedDict
from functools import partial
from typing import Optional, Tuple, Union

from transformers.utils import is_torch_available


if is_torch_available():
    import torch
    import torch.distributed
    import torch.nn.functional as F
    from torch import zeros, randn, zeros_like


def is_at_least_volta_gpu():
    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(0)
        if gpu_properties.major >= 7:
            return True
    return False


TRITON_AVAIL = False
HAS_APPROX_TOP_K = False
try:
    from .triton_src import CVMMSel, cvmm, cvmm_prepare_sel2

    if not is_at_least_volta_gpu():
        raise ImportError("GPU must at least be Volta")
    TRITON_AVAIL = True
except ImportError:
    from transformers.utils import logging

    logger = logging.get_logger(__name__)
    if torch.cuda.is_available():
        logger.warning(
            "Could not import triton_src.moe_layer.cvmm. Using cuda_src.moe_layer.cvmm instead."
        )
    else:
        logger.warning(
            "GPU not available. Falling back to CPU verison of the MoE layer, which is equally fast as a dense layer."
        )
    from .cuda_src import CVMMSel, cvmm, cvmm_prepare_sel
except RuntimeError as e:
    print(f"Error importing triton:\n{e}")
    print("Trying to import fast router code")
    from .triton_src import ApproximateTopkRouter
    HAS_APPROX_TOP_K = True

if not HAS_APPROX_TOP_K:
    try:
        from .triton_src import ApproximateTopkRouter
        HAS_APPROX_TOP_K = True
    except Exception as e:
        print(f"Could not load approximate top-k router: {e}")


def dist_logsumexp(x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    # Calculate numerically stable distributed logsumexp
    xmax = x.max(dim=dim, keepdim=True).values
    torch.distributed.all_reduce(xmax, op=torch.distributed.ReduceOp.MAX)

    xe = (x - xmax).exp().sum(dim=dim, keepdim=True)
    torch.distributed.all_reduce(xe, op=torch.distributed.ReduceOp.SUM)

    res = xmax + xe.log()
    if not keepdim:
        res = res.squeeze(dim)

    return res


def log_mean(x: torch.Tensor, dim: int = 0):
    if torch.distributed.is_initialized():
        xlse = dist_logsumexp(x, dim=dim)

        # Normalize
        n = torch.tensor(x.shape[dim]).to(x.device)
        torch.distributed.all_reduce(n, op=torch.distributed.ReduceOp.SUM)
        return xlse - n.log()
    else:
        return x.logsumexp(dim) - math.log(x.shape[dim])


def entropy_l(l: torch.Tensor) -> torch.Tensor:
    return -(l * l.exp()).sum(-1)


def non_traceable_to_traceable(state_dict: OrderedDict, prefix: str):
    """Convert non-traceable state dict into traceable state dict in-place"""
    keys_to_delete = [prefix + "keys", prefix + "values"]
    keys = state_dict[prefix + "keys"]
    values = state_dict[prefix + "values"]
    bias = None
    if prefix + "bias" in state_dict:
        keys_to_delete.append(prefix + "bias")
        bias = state_dict[prefix + "bias"]
    n_experts = len(keys)
    for i in range(n_experts):
        state_dict[prefix + f"keys.{i}.weight"] = keys[i].T
        state_dict[prefix + f"values.{i}.weight"] = values[i].T
        if bias is not None:
            state_dict[prefix + f"keys.{i}.bias"] = bias[i]
    state_dict[prefix + "expert_sel.weight"] = state_dict[prefix + "expert_sel.weight"]
    for key_to_delete in keys_to_delete:
        state_dict.pop(key_to_delete)


def traceable_to_non_traceable(state_dict: OrderedDict, prefix: str):
    """Convert traceable to non-traceable state dict"""
    key_names = [
        k for k in state_dict if re.match(rf"^{re.escape(prefix)}keys\.\d+\.weight$", k)
    ]
    key_biases = [
        k for k in state_dict if re.match(rf"^{re.escape(prefix)}keys\.\d+\.bias$", k)
    ]
    values_names = [
        k
        for k in state_dict
        if re.match(rf"^{re.escape(prefix)}values\.\d+\.weight$", k)
    ]
    keys_to_delete = [*key_names, *values_names, *key_biases]
    bias = None
    if prefix + "keys.0.bias" in state_dict:
        bias = torch.stack([state_dict[k] for k in key_biases])
        state_dict[prefix + "bias"] = bias
    state_dict[prefix + "keys"] = torch.stack(
        [state_dict[k] for k in key_names]
    ).transpose(2, 1)
    state_dict[prefix + "values"] = torch.stack(
        [state_dict[k] for k in values_names]
    ).transpose(2, 1)
    state_dict[prefix + "expert_sel.weight"] = state_dict[prefix + "expert_sel.weight"]
    for key_to_delete in keys_to_delete:
        state_dict.pop(key_to_delete)


def load_state_dict_pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
    traceable,
):
    if prefix + "keys" in state_dict:
        state_dict_is_for_non_traceable = True
    else:
        state_dict_is_for_non_traceable = False

    if traceable and state_dict_is_for_non_traceable:
        # convert state dict to traceable
        non_traceable_to_traceable(state_dict=state_dict, prefix=prefix)
    elif not traceable and not state_dict_is_for_non_traceable:
        traceable_to_non_traceable(state_dict=state_dict, prefix=prefix)


class SigmaMoELayer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        expert_size: int,
        k: int,
        dropout: float = 0,
        selection_mode: str = "sigmoid",
        activation_after_topk: bool = False,
        activation=F.relu,
        bias: bool = False,
        v_dim: Optional[int] = None,
        sinkhorn_n_iters: int = 3,
        expert_dropout: float = 0.0,
        traceable: bool = False,
        approximate: bool = False,
        triton_approximate: bool = False,
    ):
        super().__init__()
        self.k_dim = d_model
        self.v_dim = v_dim if v_dim is not None else d_model
        self.n_experts = n_experts
        self.expert_size = expert_size
        self.size = self.n_experts * self.expert_size
        self.dropout = dropout
        self.selection_mode = selection_mode
        self.k_vec_dim = self.k_dim
        self.n_heads = k
        self.activation_after_topk = activation_after_topk
        self.activation = activation
        self.sinkhorn_n_iters = sinkhorn_n_iters
        self.expert_dropout = expert_dropout
        self.traceable = traceable
        self.approximate = approximate
        self.triton_approximate = triton_approximate
        
        assert n_experts % k == 0, "Num experts must be divisible by top-k"
        self.bucket_size = n_experts // k
        self.n_buckets = n_experts // self.bucket_size

        if self.triton_approximate:
            assert self.bucket_size >= 16, "Too small bucket size. Your n_experts must be at least 16x higher than your k"
        
        if approximate:
            assert k % self.n_buckets == 0, "top-k must be divisible by num buckets, which is ceil(E / bucket size)"

        if self.selection_mode not in {"softmax", "sigmoid", "sinkmoid"}:
            raise ValueError(f"Unknown selection mode {self.selection_mode}")

        self.expert_sel = torch.nn.Linear(
            in_features=self.k_vec_dim, out_features=self.n_experts, bias=False
        )

        self._register_load_state_dict_pre_hook(
            partial(load_state_dict_pre_hook, traceable=traceable)
        )

        if traceable:
            self.keys = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        in_features=self.k_vec_dim,
                        out_features=self.expert_size,
                        bias=bias,
                    )
                    for _ in range(self.n_experts)
                ]
            )
            self.values = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        in_features=self.expert_size,
                        out_features=self.k_vec_dim,
                        bias=False,
                    )
                    for _ in range(self.n_experts)
                ]
            )
            if bias:
                self.o_bias = torch.nn.Parameter(zeros(self.v_dim))
                self.bias = None
            else:
                self.o_bias = None
                self.bias = None
        else:
            self.keys = torch.nn.Parameter(
                randn(self.n_experts, self.k_vec_dim, self.expert_size)
            )
            self.values = torch.nn.Parameter(
                randn(self.n_experts, self.expert_size, self.v_dim)
            )
            if bias:
                self.bias = torch.nn.Parameter(zeros(self.n_experts, self.expert_size))
                self.o_bias = torch.nn.Parameter(zeros(self.v_dim))
            else:
                self.o_bias = None
                self.bias = None

    def renorm_keep_std(self, weight: torch.Tensor, dim: int = 0):
        with torch.no_grad():
            std = weight.std()
            weight.div_(weight.norm(dim=dim, keepdim=True))
            weight.mul_(std / weight.std())

    def entropy_reg(self, sel: torch.Tensor, is_softmaxed: bool = False) -> float:
        # Everything is done in log scale
        sel = sel.flatten(0, -2)
        
        # reverse the softmax
        if is_softmaxed:
            # this is e^{raw-router-logits}
            sel = (sel / (1 - sel)).clamp_min(1e-6)
            # this is the log-softmax
            sel = (sel / sel.sum(dim=-1, keepdim=True)).log()
        else:
            sel = F.log_softmax(sel, dim=-1)
        sel = log_mean(sel, -2)
        return -entropy_l(sel).mean()

    def cvmm_wrapper(
        self,
        inputs: torch.Tensor,
        sel_indices: Union["CVMMSel", torch.Tensor],
        weights: torch.Tensor,
    ):
        """
        TODO

        Args:
            inputs (torch.Tensor): Shape [bsz, seq_len, d_model]
            sel_indices (Union[CVMMSel,torch.Tensor]): _description_
            weights (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        return cvmm(inputs, sel_indices, weights)

    def compute_scores_traceable(
        self, inp: torch.Tensor, index: torch.Tensor
    ) -> torch.Tensor:
        
        bsz, seq_len, d_model = inp.shape
        scores = torch.zeros(
            (bsz, seq_len, self.expert_size),
            device=inp.device,
            dtype=inp.dtype
        )
        scores = scores.view(-1, self.expert_size)
        index, sorting_indices = index.view(-1).sort()
        inp = inp.view(-1, d_model)
        for expert_idx in range(self.n_experts):
            tokens = inp[sorting_indices[index == expert_idx]]
            if tokens.numel() > 0:
                scores[sorting_indices[index == expert_idx]] = self.keys[expert_idx](
                    tokens
                )
        scores = scores.view(bsz, seq_len, self.expert_size)
        scores = F.relu(scores)
        
        return scores

    def compute_scores(
        self,
        inp: torch.Tensor,
        index: Union["CVMMSel", torch.Tensor],
        expert_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        IS_CUDA = inp.is_cuda
        if IS_CUDA:
            scores = self.cvmm_wrapper(inp, index, self.keys)
            if self.bias is not None:
                scores = scores + self.bias[index.raw_sel]
        else:
            scores = index * F.linear(inp, self.keys, None)
            if self.bias is not None:
                scores = scores + index * self.bias

        scores = self.activation(scores)
        if expert_scores is not None:
            scores = scores * expert_scores[..., None]

        if self.dropout > 0:
            # Standard dropout on the "up-projected scores"
            scores = F.dropout(scores, self.dropout, training=self.training)

        return scores

    def sel_activation(self, sel: torch.Tensor) -> torch.Tensor:
        if self.selection_mode == "sinkmoid":
            if self.training:
                with torch.no_grad():
                    sel = self.sinkhorn_unnorm(sel)
            else:
                sel = torch.sigmoid(sel)
        elif self.selection_mode == "sigmoid":
            sel = torch.sigmoid(sel)
        elif self.selection_mode == "softmax":
            sel = F.softmax(sel, dim=-1)
        else:
            assert False

        return sel

    def sinkhorn_unnorm(self, x: torch.Tensor) -> torch.Tensor:
        # Based on https://arxiv.org/abs/2202.01169. Unnormalized verison
        A, B = x.shape[-2:]

        a = zeros_like(x[..., 0, :])
        b = zeros_like(x[..., 0])

        for _ in range(self.sinkhorn_n_iters):
            b = math.log(A) - (x - a[..., None, :]).logsumexp(-1)
            if torch.distributed.is_initialized():
                a = math.log(B) - dist_logsumexp(x - b[..., None], -2)
            else:
                a = math.log(B) - (x - b[..., None]).logsumexp(-2)

        return (a[..., None, :] + b[..., None] + x).exp()

    def create_index(self, index: torch.Tensor) -> torch.Tensor:
        bs, seq_len = index.shape
        one_hot = torch.nn.functional.one_hot(index, num_classes=self.n_experts)
        return (
            one_hot.unsqueeze(-1)
            .expand(bs, seq_len, self.n_experts, self.expert_size)
            .reshape((bs, seq_len, -1))
        )

    def forward(self, inp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # control flow variables
        use_triton_kernel = TRITON_AVAIL and inp.is_cuda
        use_cuda_kernel = not use_triton_kernel and inp.is_cuda and not self.traceable
        use_fast_cpu = (
            not use_cuda_kernel and not use_triton_kernel and not self.traceable
        )
        if self.traceable:
            use_cuda_kernel = False
            use_triton_kernel = False
            use_fast_cpu = False

        if use_fast_cpu:
            if self.keys.ndim > 2:
                self.keys.data = torch.reshape(
                    self.keys.transpose(1, 2),
                    (int(self.n_experts * self.expert_size), self.k_vec_dim),
                )
            if self.values.ndim > 2:
                self.values.data = torch.reshape(
                    self.values,
                    (int(self.n_experts * self.expert_size), self.k_vec_dim),
                ).T
            if self.bias is not None and self.bias.ndim > 1:
                self.bias.data = torch.reshape(
                    self.bias, (int(self.n_experts * self.expert_size),)
                )

        # use the fused router with approx top-k
        if self.approximate and self.triton_approximate:
            assert self.expert_dropout <= 0, "Expert dropout not supported for this mode"
            assert not self.activation_after_topk, "Act. after top-k not supported for fast triton"
            assert HAS_APPROX_TOP_K, "Could not import approximate top-k router in triton"
            sel_raw, sel_val, sel_index = ApproximateTopkRouter.apply(
                inp, self.expert_sel.weight.T, self.bucket_size, self.n_heads, self.n_experts, self.training
            )
            if self.training:
                reg_loss = self.entropy_reg(sel_raw, is_softmaxed=True)
            else:
                # we don't need the loss during inference
                reg_loss = torch.tensor(0.0).to(device=inp.device, dtype=inp.dtype)
        else:
            sel = sel_raw = self.expert_sel(inp)
            reg_loss = self.entropy_reg(sel_raw)

            # Selection activation and topk
            if (not self.activation_after_topk) or (self.selection_mode == "sinkmoid"):
                # Sinkhorn should be always applied before top-k
                sel = self.sel_activation(sel)

            if self.training and self.expert_dropout > 0:
                mask = torch.rand_like(sel) < self.expert_dropout
                sel = sel.masked_fill(mask, float("-inf"))

            # sel val and sel_index have shape (bs, seq_len, n_heads)
            # where n_heads is the number of experts we select
            # Example: sel_val[1,3,:] are the scores (ordered) of token 4 of sequence 2
            #     [0.69,0.42] are the scores
            # Example: sel_index[1,3,:] are the indices (ordered) of token 4 of sequence 2
            #     [2,1] are the indices
            if self.approximate:
                sel_val, sel_index_local = torch.stack(
                    sel.chunk(self.n_buckets, dim=-1), dim=-1
                ).topk(self.n_heads // self.n_buckets, dim=-2, sorted=False)
                sel_index = sel_index_local + torch.arange(self.n_buckets, device=sel.device) * self.bucket_size
                # technically, the transpose is not necessary. it just
                # aligns the local top-k next to each other.
                # probably faster to remove them
                sel_index = sel_index.transpose(-2, -1).flatten(start_dim=-2)
                sel_val = sel_val.transpose(-2, -1).flatten(start_dim=-2)

                # debug the overlap
                # _, sel_index_correct = sel.topk(self.n_heads, dim=-1, sorted=False)
                # coverage = torch.isin(sel_index, sel_index_correct).float().mean(-1)
            else:
                sel_val, sel_index = sel.topk(self.n_heads, dim=-1, sorted=False)

        if self.activation_after_topk or (self.selection_mode == "sinkmoid"):
            sel_val = torch.gather(sel_raw, -1, sel_index)
            # for sinkmoid, the score is always calculated by a sigmoid
            sel_val = (
                torch.sigmoid(sel_val)
                if self.selection_mode == "sinkmoid"
                else self.sel_activation(sel_val)
            )

        # Preprocess the selection indices. They will be needed for both layers and save some time
        if use_cuda_kernel or use_triton_kernel:
            if use_triton_kernel:
                sel_indices = cvmm_prepare_sel2(sel_index.int())
            else:
                sel_indices = [
                    cvmm_prepare_sel(sel_index[..., h].int(), self.n_experts)
                    for h in range(sel_index.shape[-1])
                ]
        elif use_fast_cpu:
            sel_indices = [
                self.create_index(sel_index[..., h].long())
                for h in range(sel_index.shape[-1])
            ]

        if use_triton_kernel:
            # "Up-projection" layer for each head
            scores = self.compute_scores(inp, sel_indices)

            # Down projection layer for each head
            sel_indices = sel_indices.clone()
            sel_indices.reduction_weight = sel_val
            sel_indices.sel_index = sel_indices.out_index
            sel_indices.out_index = None
            out = self.cvmm_wrapper(scores, sel_indices, self.values)
            res = out.view(*inp.shape[:-1], self.v_dim)
        elif use_cuda_kernel or use_fast_cpu:
            # "Up-projection" layer for each head
            scores_l = [
                self.compute_scores(inp, sel_indices[h], sel_val[..., h])
                for h in range(sel_index.shape[-1])
            ]
            # Down projection layer for each head
            res = 0
            for scores in scores_l:
                # we don't need to mask with the indices here since the
                # hidden activations of the non-used experts are zero
                res = res + F.linear(scores, self.values, None)
        else:
            # traceable
            scores_l = [
                self.compute_scores_traceable(inp, sel_index[..., h].long())
                for h in range(sel_index.shape[-1])
            ]

            # Down projection layer for each head
            res = torch.zeros_like(inp)
            bsz, seq_len, d_model = inp.shape
            res = res.view(-1, d_model)
            for h, scores in enumerate(scores_l):
                scores = scores.view(-1, self.expert_size)
                sel_index_h = sel_index[..., h].view(-1)
                sel_val_h = sel_val[..., h].view(-1)
                sel_index_h, sort_indices = sel_index_h.sort()
                for expert_idx in range(self.n_experts):
                    tgt_ind = sort_indices[sel_index_h == expert_idx]
                    tokens = scores[tgt_ind]
                    res[tgt_ind] = res[tgt_ind] + sel_val_h[tgt_ind].unsqueeze(
                        -1
                    ) * self.values[expert_idx](tokens)

            res = res.view(bsz, seq_len, d_model)

        if self.o_bias is not None:
            res = res + self.o_bias
        return res, reg_loss

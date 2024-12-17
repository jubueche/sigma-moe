import math
import torch
from torch import Tensor
from torch.autograd import Function
import triton
import triton.language as tl
import triton.language.core as core
from triton.language.standard import _log2, sum, zeros_like


@triton.jit
def _compare_and_swap(x, ids, flip, i: core.constexpr, n_dims: core.constexpr):
    n_outer: core.constexpr = x.numel >> n_dims
    shape: core.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]
    y = core.reshape(x, shape)
    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = core.arange(0, 2)[None, :, None]
    left = core.broadcast_to(sum(y * (1 - mask), 1)[:, None, :], shape)
    right = core.broadcast_to(sum(y * mask, 1)[:, None, :], shape)
    left = core.reshape(left, x.shape)
    right = core.reshape(right, x.shape)

    # idx
    y_idx = core.reshape(ids, shape)
    left_idx = core.broadcast_to(sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = core.broadcast_to(sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = core.reshape(left_idx, x.shape)
    right_idx = core.reshape(right_idx, x.shape)

    # actual compare-and-swap
    idtype = core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) ^ flip

    ret = ix ^ core.where(cond, ileft ^ iright, zeros_like(ix))

    new_ids = ids ^ core.where(cond, left_idx ^ right_idx, zeros_like(ids))

    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(
    x, ids, stage: core.constexpr, order: core.constexpr, n_dims: core.constexpr
):
    """
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    """
    n_outer: core.constexpr = x.numel >> n_dims
    core.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        shape: core.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = core.reshape(
            core.broadcast_to(core.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in core.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def argsort(
    x, ids, dim: core.constexpr = None, descending: core.constexpr = core.CONSTEXPR_0
):
    # handle default dimension or check that it is the most minor dim
    _dim: core.constexpr = len(x.shape) - 1 if dim is None else dim
    core.static_assert(
        _dim == len(x.shape) - 1, "only minor dimension is currently supported"
    )
    # iteratively run bitonic merge-sort steps
    n_dims: core.constexpr = _log2(x.shape[_dim])

    for i in core.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < n_dims else descending, n_dims)
    return x, ids


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    if "tiles_per_update" in args:
        ret["name"] = (
            f"{kernel.name} [M={M}, N={N}, K={K}, tiles_per_update={args['tiles_per_update']:02}]"
        )
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


@triton.jit(launch_metadata=_matmul_launch_metadata)
def kernel_matmul_sigmoid_topk(
    a_ptr,
    b_ptr,
    c_ptr,  #
    top_k_vals_ptr,
    top_k_inds_ptr,
    M,
    N,
    K,  #
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,  #
    stride_topk_m,
    stride_topk_k,
    kb,
    training,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply sigmoid activation to the output block
    accumulator = tl.sigmoid(accumulator)

    if c_ptr.dtype.element_ty == tl.float8e4nv:
        c = accumulator.to(tl.float8e4nv)
    elif c_ptr.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator

    # if we are not training, avoid storing back the mvm result
    if training:
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

        tl.store(c_ptrs, c, mask=c_mask)

    sorted_vals = tl.max(c, axis=-1)
    sorted_indices = tl.argmax(c, axis=-1)
    sorted_indices = start_n + sorted_indices

    offs_kb_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    top_k_address = stride_topk_m * offs_kb_m + stride_topk_k * pid_n
    top_k_vals_ptrs = top_k_vals_ptr + top_k_address
    top_k_inds_ptrs = top_k_inds_ptr + top_k_address
    
    top_k_mask = offs_kb_m < M
    tl.store(top_k_vals_ptrs, sorted_vals, mask=top_k_mask)
    tl.store(top_k_inds_ptrs, sorted_indices, mask=top_k_mask)


    # ids = tl.broadcast_to(
    #     tl.arange(0, BLOCK_SIZE_N)[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N)
    # )
    # sorted_vals, sorted_indices = argsort(c, ids, descending=True)

    # # we need to offset the indices by the block-id
    # sorted_indices = start_n + sorted_indices

    # offs_kb_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # offs_kb_k = tl.arange(0, BLOCK_SIZE_N)
    # moved_k_offs = pid_n * kb + offs_kb_k

    # top_k_address = (
    #     stride_topk_m * offs_kb_m[:, None]
    #     + stride_topk_k * moved_k_offs[None, :]
    # )
    # top_k_vals_ptrs = top_k_vals_ptr + top_k_address
    # top_k_inds_ptrs = top_k_inds_ptr + top_k_address
    # top_k_mask = (offs_kb_m[:, None] < M) & (offs_kb_k[None, :] < kb)

    # sorted_indices = sorted_indices.to(tl.float32)
    # tl.store(top_k_vals_ptrs, sorted_vals, mask=top_k_mask)
    # tl.store(top_k_inds_ptrs, sorted_indices, mask=top_k_mask)


def matmul_sigmoid_topk(a: Tensor, b: Tensor, bucket_size: int, k: int, n_experts: int, training: bool):
    """
    Performs matrix multiplication of input tensors `a` and `b`, applies the sigmoid function to the results, 
    and computes approximate top-k values and their indices per token.

    This function uses Triton kernels for efficient computation with specialized configurations based on 
    the data type of the input tensor `a`.

    Args:
        a (Tensor): Input tensor of shape `(..., K)`, where `K` is the model-dim.
        b (Tensor): Weight tensor of shape `(K, N)`, where `N` is the output dimension. Must have 2 dimensions.
        bucket_size (int): Size of the bucket to divide the output dimension `N`. `N` must be divisible by `bucket_size`.
        k (int): Normal top-k in MoEs.
        n_experts (int): Number of experts, used to determine the number of buckets for the top-k operation.
        training (bool): Are we in training mode.

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            - `c` (Tensor): Output tensor of the matrix multiplication with shape `(..., N)`.
            - `top_k_vals` (Tensor): Tensor of shape `(..., k)` containing the top-k values per token after applying sigmoid.
            - `top_k_indices` (Tensor, long): Tensor of shape `(..., k)` containing the indices of the top-k values per token.

    Raises:
        AssertionError: If `b` is not a 2D tensor.
        AssertionError: If `N % bucket_size != 0`.
        AssertionError: If `k % n_buckets != 0`, where `n_buckets = ceil(n_experts / bucket_size)`.

    Notes:
        - Triton kernels are configured with optimal block sizes and other parameters depending on the data type of `a`.
        - The configurations for `torch.float8_e4m3fn`, `torch.float16`, and `torch.float32` specify different block sizes 
          (`BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`), group sizes (`GROUP_SIZE_M`), and other kernel parameters.
        - The output `c` is computed as the result of `a @ b`.
        - The sigmoid activation function is applied element-wise on `c` before extracting top-k values.
    """
    configs = {
        torch.float8_e4m3fn: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": bucket_size,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
            "num_stages": 4,
            "num_warps": 8,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": bucket_size,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float32: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": bucket_size,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
    }

    assert b.ndim == 2, "Num. dimension of weight must be 2"
    a_shape = a.shape
    dtype = a.dtype
    out_shape_c = a_shape[:-1] + (b.size(-1),)
    out_shape_k = a_shape[:-1] + (k,)

    a = a.flatten(end_dim=-2)
    a = a.contiguous()
    b = b.contiguous()

    M, K = a.shape
    K, N = b.shape

    assert (
        N % bucket_size == 0
    ), "Output dimension N must be divisible by block size B_OUT"

    n_buckets = math.ceil(n_experts / bucket_size)
    assert (
        k % n_buckets == 0
    ), "top-k must be divisible by num buckets, which is ceil(E / bucket size)"
    kb = k // n_buckets

    # Output tensors
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    top_k_vals = torch.empty((M, k), device=a.device, dtype=torch.float32)
    top_k_indices = torch.empty((M, k), device=a.device, dtype=torch.float32)

    # Define grid and block size
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    kernel_matmul_sigmoid_topk[grid](
        a,
        b,
        c,
        top_k_vals,
        top_k_indices,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        top_k_vals.stride(0),
        top_k_vals.stride(1),
        kb,  # top-k per block
        training,  # training mode
        BLOCK_SIZE_M=configs[dtype]["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=configs[dtype]["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=configs[dtype]["BLOCK_SIZE_K"],
        GROUP_SIZE_M=configs[dtype]["GROUP_SIZE_M"],
        num_stages=configs[dtype]["num_stages"],
        num_warps=configs[dtype]["num_warps"],
    )
    c = c.to(dtype=a.dtype).view(out_shape_c)
    top_k_vals = top_k_vals.to(dtype=a.dtype).view(out_shape_k)
    top_k_indices = top_k_indices.to(torch.long).view(out_shape_k)
    return c, top_k_vals, top_k_indices


class ApproximateTopkRouter(Function):
    @staticmethod
    def forward(ctx, inp: Tensor, weight: Tensor, bucket_size: int, k: int, n_experts: int, training: bool):
        """
        Forward pass: Call the matmul_sigmoid_topk function.
        Expects the inp as shape [...,K] and the weights are [K, N] where
        N is the output dimension.
        """
        # Call the given function
        c, top_k_vals, top_k_indices = matmul_sigmoid_topk(inp, weight, bucket_size, k, n_experts, training)
        
        # Save for backward pass
        ctx.save_for_backward(inp, weight, c, top_k_vals, top_k_indices)
        ctx.bucket_size = bucket_size
        ctx.k = k
        ctx.n_experts = n_experts
        
        return c, top_k_vals, top_k_indices

    @staticmethod
    def backward(ctx, grad_c: Tensor, grad_top_k_vals: Tensor, grad_top_k_indices: Tensor):
        """
        Backward pass: Compute gradients w.r.t. a and b.
        """
        a, b, c, top_k_vals, top_k_indices = ctx.saved_tensors
        bucket_size = ctx.bucket_size
        k = ctx.k
        n_experts = ctx.n_experts

        # 1. Backprop through sigmoid: grad_B = grad_C * sigmoid(C) * (1 - sigmoid(C))
        sigmoid_c = torch.sigmoid(c)
        grad_b = grad_c * sigmoid_c * (1 - sigmoid_c)

        # 2. Backprop through top-k
        grad_a_topk = torch.zeros_like(c)
        grad_a_topk.scatter_add_(-1, top_k_indices, grad_top_k_vals)

        # Combine gradients from top-k and sigmoid
        grad_a_combined = grad_a_topk + grad_b

        # 3. Backprop through matrix multiplication A = XW
        grad_a = grad_a_combined @ b.T
        grad_b = a.T @ grad_a_combined

        return grad_a, grad_b, None, None, None, None


def pytorch_fwd(tokens, weights, bucket_size, k, n_experts):
    n_buckets = math.ceil(n_experts / bucket_size)
    sel = tokens @ weights
    sel = torch.nn.functional.sigmoid(sel)
    sel_val, sel_index = torch.stack(sel.chunk(n_buckets, dim=-1), dim=-1).topk(
        k // n_buckets, dim=-2, sorted=False
    )
    sel_index += torch.arange(n_buckets) * bucket_size
    # technically, the transpose is not necessary. it just
    # aligns the local top-k next to each other.
    # probably faster to remove them
    sel_index = sel_index.transpose(-2, -1).flatten(start_dim=-2)
    sel_val = sel_val.transpose(-2, -1).flatten(start_dim=-2)
    return sel, sel_val, sel_index


def test_forward_correctness(
    bsz,
    d_model,
    n_tokens,
    n_experts,
    bucket_size,
    k,
):
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    dd_dict = {"device": device, "dtype": dtype}

    tokens = torch.randn(bsz, n_tokens, d_model, **dd_dict)
    weights = torch.randn(d_model, n_experts, **dd_dict) / d_model

    c, sel_vals, sel_indices = matmul_sigmoid_topk(
        tokens, weights, bucket_size=bucket_size, k=k, n_experts=n_experts, training=True
    )

    c_torch, sel_vals_torch, sel_indices_torch = pytorch_fwd(
        tokens=tokens,
        weights=weights,
        bucket_size=bucket_size,
        k=k,
        n_experts=n_experts,
    )

    assert torch.allclose(c, c_torch, atol=1e-4)
    test_correctness_between_ind_sel(
        sel_indices=sel_indices,
        sel_vals=sel_vals,
        sel_indices_torch=sel_indices_torch,
        sel_vals_torch=sel_vals_torch,
    )


def test_forward_backward(
    bsz,
    d_model,
    n_tokens,
    n_experts,
    bucket_size,
    k,
):
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    dd_dict = {"device": device, "dtype": dtype}

    tokens = torch.randn(bsz, n_tokens, d_model, **dd_dict)
    weights = torch.randn(d_model, n_experts, **dd_dict) / d_model

    pass

    # c, sel_vals, sel_indices = matmul_sigmoid_topk(
    #     tokens, weights, bucket_size=bucket_size, k=k, n_experts=n_experts
    # )

    # c_torch, sel_vals_torch, sel_indices_torch = pytorch_fwd(
    #     tokens=tokens,
    #     weights=weights,
    #     bucket_size=bucket_size,
    #     k=k,
    #     n_experts=n_experts,
    # )

    # assert torch.allclose(c, c_torch, atol=1e-4)
    # test_correctness_between_ind_sel(
    #     sel_indices=sel_indices,
    #     sel_vals=sel_vals,
    #     sel_indices_torch=sel_indices_torch,
    #     sel_vals_torch=sel_vals_torch,
    # )


def test_correctness_between_ind_sel(
    sel_indices, sel_vals, sel_indices_torch, sel_vals_torch
):
    sel_indices = sel_indices.flatten(end_dim=-2)
    sel_vals = sel_vals.flatten(end_dim=-2)

    sel_indices_torch = sel_indices_torch.flatten(end_dim=-2)
    sel_vals_torch = sel_vals_torch.flatten(end_dim=-2)

    for input_idx in range(sel_indices_torch.size(0)):
        si, si_idx = torch.sort(sel_indices[input_idx])
        si_torch, si_torch_idx = torch.sort(sel_indices_torch[input_idx])
        assert torch.allclose(si, si_torch, atol=1e-4)

        sv = sel_vals[input_idx, si_idx]
        sv_torch = sel_vals_torch[input_idx, si_torch_idx]
        assert torch.allclose(sv, sv_torch, atol=1e-4)


if __name__ == "__main__":
    bsz = 10
    d_model = 512
    n_tokens = 1
    n_experts = 512
    bucket_size = 32
    k = 16

    test_forward_correctness(
        bsz=bsz,
        d_model=d_model,
        n_tokens=n_tokens,
        n_experts=n_experts,
        bucket_size=bucket_size,
        k=k,
    )

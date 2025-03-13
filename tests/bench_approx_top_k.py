import math
from dataclasses import dataclass
from functools import partial
from scipy.optimize import minimize_scalar
import numpy as np

import torch
import triton
import triton.testing

from sigma_moe.moe_layer import SigmaMoELayer
from sigma_moe import SigmaMoEForCausalLM, SigmaMoEConfiguration


@dataclass
class Architecture:
    granularity: int
    n_layers: int
    top_k: int
    n_experts: int
    d_model: int
    d_ff: int
    d_expert: int


def get_architecture(flops_budget: int):
    # apply scaling laws
    possible_granularities = [4, 8, 16, 32, 64]

    losses_and_n_layers = [
        find_best_moe(flops_budget, granularity)
        for granularity in possible_granularities
    ]
    arg_min = np.argmin([loss for loss, _ in losses_and_n_layers])
    granularity = possible_granularities[arg_min]
    n_layers = losses_and_n_layers[arg_min][1]
    top_k = granularity
    n_experts = 64 * granularity
    n_params_moe = num_params(n_layers = n_layers, E=64) / 1e9
    d_model = int(n_layers * 64)
    d_ff = int(4 * d_model)
    d_expert = int(d_ff / granularity)
    arch = Architecture(
        granularity=granularity,
        n_layers=n_layers,
        top_k=top_k,
        n_experts=n_experts,
        d_model=d_model,
        d_ff=d_ff,
        d_expert=d_expert
    )
    return arch


def loss_moe(N, D, G):
    a = 18.1
    alpha = 0.115
    b = 30.8
    beta = 0.147
    g = 2.1
    gamma = 0.58
    c = 0.47
    return c + (g / (G**gamma) + a) * 1 / N**alpha + b / D**beta


def loss_dense(N, D):
    a = 16.3
    alpha = 0.126
    b = 26.7
    beta = 0.127
    c = 0.47
    return c + a / N**alpha + b / D**beta


def num_params(n_layers, E):
    # E = 1 is dense
    d_model = 64 * n_layers
    return d_model + n_layers * (4 + 8 * E) * d_model**2


def moe_objective(n_layers, flops_budget, cf, expansion, granularity, cr):
    # we optimize w.r.t. the number of layers
    d_model = 64 * n_layers
    number_of_tokens = flops_budget / (
        (12 * d_model**2 * cf + d_model * expansion * granularity * cr) * n_layers
    )
    num_parameters = d_model**2 * (8 * expansion + 4) * n_layers
    return loss_moe(N=num_parameters, D=number_of_tokens, G=granularity)


def dense_objective(n_layers, flops_budget, cf, cr):
    # we optimize w.r.t. the number of layers
    d_model = 64 * n_layers
    number_of_tokens = flops_budget / (12 * d_model**2 * cf * n_layers)
    num_parameters = 12 * d_model**2 * n_layers
    return loss_dense(N=num_parameters, D=number_of_tokens)


def find_best_moe(flops_budget: int, granularity: int):
    cf = 6
    cr = 14
    expansion = 64
    objective_fun = partial(
        moe_objective,
        flops_budget=flops_budget,
        cf=cf,
        expansion=expansion,
        granularity=granularity,
        cr=cr,
    )
    moe_res = minimize_scalar(
        objective_fun,
        bracket=(
            1,
            100
            if math.log10(flops_budget) > 23
            else 10
            if math.log10(flops_budget) > 16
            else 5,
            1000,
        ),
        method="brent",
    )
    loss_ceil = objective_fun(math.ceil(moe_res.x))
    return loss_ceil, math.ceil(moe_res.x)


def find_best_dense(flops_budget: int):
    cf = 6
    cr = 14
    objective_fun = partial(dense_objective, flops_budget=flops_budget, cf=cf, cr=cr)
    moe_res = minimize_scalar(
        objective_fun,
        bracket=(1, 100 if math.log10(flops_budget) > 23 else 10, 1000),
        method="brent",
    )
    loss_ceil = objective_fun(math.ceil(moe_res.x))
    return loss_ceil, math.ceil(moe_res.x)


def get_layer(
    d_model,
    approximate,
    triton_approximate,
    n_experts,
    expert_size,
    top_k,
    training,
    device,
    dtype,
):
    # Initialize the layer
    layer = SigmaMoELayer(
        d_model=d_model,
        n_experts=n_experts,
        expert_size=expert_size,
        k=top_k,
        approximate=approximate,
        triton_approximate=triton_approximate,
    ).to(device=device, dtype=dtype)
    if not training:
        layer.eval()

    return layer


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["flops_budget"],  # The varying dimension
        x_vals=[np.log10(f * 10.0**e) for e in range(18, 21) for f in np.arange(1, 10, 2)],
        line_arg="mode",  # The mode we compare
        line_vals=[
            "non_approximate",
            "approximate_with_triton"
        ],
        line_names=[
            "Non-Approximate",
            "Approximate (Triton Enabled)"
        ],
        ylabel="Time per forward pass (s)",
        xlabel="FLOPs budget (log10)",
        plot_name="sigma_moe_benchmark",
        args={},
    )
)
def benchmark_layer_fwd(flops_budget, mode):
    flops_budget = 10**flops_budget

    # Define constants
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    # inp sequence
    context_length = 8192
    batch_size = 32
    num_tokens = context_length * batch_size
    arch = get_architecture(flops_budget)

    training = False

    # Select mode configuration
    if mode == "non_approximate":
        approximate = False
        triton_approximate = False
    elif mode == "approximate_with_triton":
        approximate = True
        triton_approximate = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    x = torch.randn(num_tokens, arch.d_model, device=device, dtype=dtype)

    layer = get_layer(
        d_model=arch.d_model,
        approximate=approximate,
        triton_approximate=triton_approximate,
        n_experts=arch.n_experts,
        expert_size=arch.d_expert,
        top_k=arch.top_k,
        training=training,
        device=device,
        dtype=dtype,
    )

    @torch.no_grad()
    def fwd_layer(x):
        return layer(x)

    for _ in range(10):
        fwd_layer(x)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: fwd_layer(x), quantiles=quantiles)
    print(f"{mode} G: {arch.granularity} E: {arch.n_experts} top-k {arch.top_k} d-model {arch.d_model} d_expert {arch.d_expert} took {ms}")
    return ms, max_ms, min_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["flops_budget"],  # The varying dimension
        x_vals=[np.log10(f * 10.0**e) for e in range(18, 21) for f in np.arange(1, 10, 2)],
        line_arg="mode",  # The mode we compare
        line_vals=[
            "non_approximate",
            "approximate_with_triton"
        ],
        line_names=[
            "Non-Approximate",
            "Approximate (Triton Enabled)"
        ],
        ylabel="Time per forward pass (s)",
        xlabel="FLOPs budget (log10)",
        plot_name="sigma_moe_model_benchmark",
        args={},
    )
)
def benchmark_model_fwd(flops_budget, mode):
    flops_budget = 10**flops_budget

    # Define constants
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    # inp sequence
    context_length = 8192
    batch_size = 32
    num_tokens = context_length * batch_size

    arch = get_architecture(flops_budget)

    training = False

    # Select mode configuration
    if mode == "non_approximate":
        approximate = False
        triton_approximate = False
    elif mode == "approximate_with_triton":
        approximate = True
        triton_approximate = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    x = torch.randint(5, 1000, (num_tokens, arch.d_model), device=device, dtype=torch.long)

    config = SigmaMoEConfiguration(
        d_model=arch.d_model,
        n_experts=arch.n_experts,
        top_k_experts=arch.top_k,
        num_hidden_layers=arch.n_layers,
        d_ff=arch.d_ff,
        expert_size=arch.d_expert,
        approximate=approximate,
        triton_approximate=triton_approximate,
    )
    moe = SigmaMoEForCausalLM(config=config)
    moe.to(device=device, dtype=dtype)

    @torch.no_grad()
    def fwd_model(x):
        return moe(x)

    for _ in range(10):
        fwd_model(x)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: fwd_model(x), quantiles=quantiles)
    print(f"{mode} layers: {arch.n_layers} G: {arch.granularity} E: {arch.n_experts} top-k {arch.top_k} d-model {arch.d_model} d_expert {arch.d_expert} took {ms}")
    return ms, max_ms, min_ms

if __name__ == "__main__":
    benchmark_model_fwd.run(show_plots=False, print_data=True, save_path="/u/jub/sigma-moe/tests/plots")
    benchmark_layer_fwd.run(show_plots=False, print_data=True, save_path="/u/jub/sigma-moe/tests/plots")

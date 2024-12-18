import torch
import triton
import triton.testing

from sigma_moe.moe_layer import SigmaMoELayer


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
        x_names=["n_experts"],  # The varying dimension
        x_vals=[2**i for i in range(11, 15)],
        line_arg="mode",  # The mode we compare
        line_vals=[
            "non_approximate",
            "approximate_no_triton",
            "approximate_with_triton"
        ],
        line_names=[
            "Non-Approximate",
            "Approximate (Triton Disabled)",
            "Approximate (Triton Enabled)"
        ],
        ylabel="Time per forward pass (s)",
        plot_name="sigma_moe_benchmark",
        args={},
    )
)
def benchmark(n_experts, mode):
    """
    Main benchmarking function for different d_model values and modes.
    """
    # Define constants
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    d_model = 128
    context_length = 8192
    batch_size = 1
    num_tokens = context_length * batch_size
    expert_size = 128
    top_k = 64
    training = False

    # Select mode configuration
    if mode == "non_approximate":
        approximate = False
        triton_approximate = False
    elif mode == "approximate_no_triton":
        approximate = True
        triton_approximate = False
    elif mode == "approximate_with_triton":
        approximate = True
        triton_approximate = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    x = torch.randn(num_tokens, d_model, device=device, dtype=dtype)

    layer = get_layer(
        d_model=d_model,
        approximate=approximate,
        triton_approximate=triton_approximate,
        n_experts=n_experts,
        expert_size=expert_size,
        top_k=top_k,
        training=training,
        device=device,
        dtype=dtype,
    )

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: layer(x), quantiles=quantiles)
    print(f"{mode} took {ms}")
    return ms, max_ms, min_ms


if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True)

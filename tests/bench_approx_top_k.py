import torch
import time
import triton
import triton.testing

from sigma_moe.moe_layer import SigmaMoELayer

def get_layer(d_model, approximate, triton_approximate, n_experts, expert_size, top_k, bucket_size, device, dtype):
    # Initialize the layer
    layer = SigmaMoELayer(
        d_model=d_model,
        n_experts=n_experts,
        expert_size=expert_size,
        k=top_k,
        approximate=approximate,
        triton_approximate=triton_approximate,
        bucket_size=bucket_size,
    ).to(device=device, dtype=dtype)
    return layer


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["d_model"],  # The varying dimension
        # x_vals=[2**i for i in range(9, 14)],  # d_model = 512 to 8192 (2^9 to 2^13)
        x_vals=[2**i for i in range(9, 11)],  # d_model = 512 to 8192 (2^9 to 2^13)
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
        args={}
    )
)
def benchmark(d_model, mode):
    """
    Main benchmarking function for different d_model values and modes.
    """
    # Define constants
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    context_length = 8192
    batch_size = 32
    num_tokens = context_length * batch_size
    expert_size = 128
    feedforward_dimension = 4 * d_model
    n_experts = feedforward_dimension // expert_size
    top_k = int(0.25 * n_experts)
    bucket_size = 128

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
        bucket_size=bucket_size,
        device=device,
        dtype=dtype
    )

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: layer(x), quantiles=quantiles)
    return ms, max_ms, min_ms


if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True)
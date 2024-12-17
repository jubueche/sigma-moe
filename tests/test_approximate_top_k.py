import torch
from sigma_moe.moe_layer import SigmaMoELayer
from sigma_moe import SigmaMoEForCausalLM, SigmaMoEConfiguration


def test_approximate_top_k():
    torch.manual_seed(0)
    
    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d_model = 1024
    n_experts = 1024
    top_k = 16
    bsz = 2

    layer = SigmaMoELayer(
        d_model=d_model,
        n_experts=n_experts,
        expert_size=2,
        k=top_k,
        approximate=True,
        triton_approximate=True,
    )
    layer.to(device=device, dtype=dtype)
    layer.eval()

    input = torch.randn(bsz, 10, d_model, dtype=dtype, device=device)
    layer(input)

    config = SigmaMoEConfiguration(
        d_model=d_model,
        n_experts=n_experts,
        top_k_experts=top_k,
        num_hidden_layers=1,
        d_ff=int(4*d_model),
        approximate=True,
        triton_approximate=True,
    )
    moe = SigmaMoEForCausalLM(config=config)
    moe.to(device=device, dtype=dtype)
    token_ids = torch.randint(low=0, high=10000, size=(bsz, 10))
    attn_mask = torch.ones((bsz, 10), dtype=dtype)
    moe(token_ids, attn_mask)


if __name__ == "__main__":
    test_approximate_top_k()

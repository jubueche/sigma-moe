import torch
from sigma_moe.moe_layer import SigmaMoELayer
from sigma_moe import SigmaMoEForCausalLM, SigmaMoEConfiguration


def test_approximate_top_k():
    d_model = 1024
    n_experts = 1024
    top_k = 16
    bsz = 2

    layer = SigmaMoELayer(
        d_model=d_model,
        n_experts=n_experts,
        expert_size=2,
        k=top_k,
        bias=True,
        approximate=True,
        bucket_size=128
    )

    input = torch.randn(bsz, 10, d_model)
    layer(input)

    config = SigmaMoEConfiguration(
        d_model=d_model,
        n_experts=n_experts,
        top_k_experts=top_k,
        num_hidden_layers=1,
        d_ff=int(4*d_model),
        approximate=True,
        bucket_size=128
    )
    moe = SigmaMoEForCausalLM(config=config)
    token_ids = torch.randint(low=0, high=10000, size=(bsz, 10))
    attn_mask = torch.ones((bsz, 10))
    moe(token_ids, attn_mask)


if __name__ == "__main__":
    test_approximate_top_k()

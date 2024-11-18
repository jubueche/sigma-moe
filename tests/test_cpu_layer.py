import torch
from sigma_moe.moe_layer import SigmaMoELayer
from sigma_moe import SigmaMoEForCausalLM, SigmaMoEConfiguration


def test_load_cpu():
    layer_traceable = SigmaMoELayer(
        d_model=5,
        n_experts=2,
        expert_size=2,
        k=2,
        bias=True,
        traceable=True,
    )
    layer = SigmaMoELayer(
        d_model=5,
        n_experts=2,
        expert_size=2,
        k=2,
        bias=True,
    )

    layer_state_dict = layer.state_dict()
    layer_traceable.load_state_dict(layer_state_dict)

    layer_state_dict = layer_traceable.state_dict()
    layer.load_state_dict(layer_state_dict)

    print(layer_traceable.state_dict())
    print(layer.state_dict())

    config = SigmaMoEConfiguration(traceable=True)
    traceable_model = SigmaMoEForCausalLM(config=config)

    config = SigmaMoEConfiguration()
    normal_model = SigmaMoEForCausalLM(config=config)

    normal_sd = normal_model.state_dict()
    traceable_model.load_state_dict(normal_sd)

    traceable_sd = traceable_model.state_dict()
    normal_model.load_state_dict(traceable_sd)

    normal_model.load_state_dict(normal_model.state_dict())
    traceable_model.load_state_dict(traceable_model.state_dict())


def test_traceable_correct_forward():
    d_model = 5
    n_experts = 5
    top_k = 2
    bsz = 2

    layer_traceable = SigmaMoELayer(
        d_model=d_model,
        n_experts=n_experts,
        expert_size=2,
        k=top_k,
        bias=True,
        traceable=True,
    )
    layer = SigmaMoELayer(
        d_model=d_model,
        n_experts=n_experts,
        expert_size=2,
        k=top_k,
        bias=True,
    )

    layer_state_dict = layer.state_dict()
    layer_traceable.load_state_dict(layer_state_dict)

    input = torch.randn(bsz, 10, d_model)
    out_traceable = layer_traceable(input)
    out = layer(input)

    assert torch.allclose(out[0], out_traceable[0], atol=1e-5)


if __name__ == "__main__":
    test_traceable_correct_forward()

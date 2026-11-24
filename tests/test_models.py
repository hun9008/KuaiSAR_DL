import pytest
import torch

from kuaisar_xtr.models import MODEL_REGISTRY, build_model


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_model_forward_and_backward(name: str) -> None:
    vocab = [31, 41, 7, 7, 7, 7, 11, 13, 5, 5, 25, 8]
    kwargs = {}
    if name == "lightgcn":
        kwargs = {
            "edge_users": torch.tensor([1, 1, 2, 3]),
            "edge_items": torch.tensor([1, 2, 2, 3]),
            "lightgcn_layers": 2,
        }
    model = build_model(name, vocab, dim=8, hidden_dims=(16, 8), **kwargs)
    x = torch.stack([torch.randint(1, size, (6,)) for size in vocab], dim=1)
    output = model(x)
    assert output.shape == (6,)
    torch.nn.functional.binary_cross_entropy_with_logits(output, torch.randint(0, 2, (6,)).float()).backward()


def test_lightgcn_performs_graph_propagation() -> None:
    model = build_model(
        "lightgcn", [5, 6], dim=4, hidden_dims=(8,),
        edge_users=torch.tensor([1, 1, 2]), edge_items=torch.tensor([1, 2, 2]),
        lightgcn_layers=2,
    )
    propagated_user, propagated_item = model.propagate()
    assert not torch.allclose(propagated_user, model.user.weight)
    assert not torch.allclose(propagated_item, model.item.weight)

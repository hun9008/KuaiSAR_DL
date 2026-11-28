import pytest
import torch

from kuaisar_xtr.model import MODEL_REGISTRY, ModelInput, build_model


STATIC_MODELS = sorted(set(MODEL_REGISTRY) - {"hyformer"})


@pytest.mark.parametrize("name", STATIC_MODELS)
def test_static_model_forward_and_backward(name: str) -> None:
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
    torch.nn.functional.binary_cross_entropy_with_logits(
        output, torch.randint(0, 2, (6,)).float()
    ).backward()


def _hyformer_kwargs() -> dict:
    # One sequence domain, one query and two NS tokens gives T=3; d_model=12
    # satisfies the original full RankMixer constraint d_model % T == 0.
    return {
        "user_int_feature_specs": [(10, 0, 1)],
        "item_int_feature_specs": [(20, 0, 1)],
        "user_dense_dim": 0,
        "item_dense_dim": 0,
        "seq_vocab_sizes": {"interaction": [30, 8]},
        "user_ns_groups": [[0]],
        "item_ns_groups": [[0]],
        "d_model": 12,
        "emb_dim": 4,
        "num_queries": 1,
        "num_hyformer_blocks": 2,
        "num_heads": 3,
        "seq_encoder_type": "transformer",
        "hidden_mult": 2,
        "dropout_rate": 0.0,
        "seq_top_k": 3,
        "seq_causal": False,
        "action_num": 1,
        "num_time_buckets": 6,
        "rank_mixer_mode": "full",
        "use_rope": True,
        "ns_tokenizer_type": "group",
    }


def _hyformer_input() -> ModelInput:
    batch_size, seq_len = 3, 4
    return ModelInput(
        user_int_feats=torch.tensor([[1], [2], [3]]),
        item_int_feats=torch.tensor([[4], [5], [6]]),
        user_dense_feats=torch.empty(batch_size, 0),
        item_dense_feats=torch.empty(batch_size, 0),
        seq_data={
            "interaction": torch.tensor(
                [
                    [[1, 2, 3, 0], [1, 2, 3, 0]],
                    [[4, 5, 0, 0], [2, 3, 0, 0]],
                    [[6, 7, 8, 9], [3, 4, 5, 6]],
                ]
            )
        },
        seq_lens={"interaction": torch.tensor([3, 2, 4])},
        seq_time_buckets={
            "interaction": torch.tensor([[1, 2, 3, 0], [1, 2, 0, 0], [1, 2, 3, 4]])
        },
    )


def test_hyformer_single_domain_forward_and_backward() -> None:
    model = build_model(
        "hyformer", [], dim=0, hidden_dims=(), hyformer_kwargs=_hyformer_kwargs()
    )
    logits = model(_hyformer_input())
    assert logits.shape == (3, 1)
    torch.nn.functional.binary_cross_entropy_with_logits(
        logits.squeeze(-1), torch.tensor([0.0, 1.0, 0.0])
    ).backward()


def test_hyformer_requires_sequence_schema() -> None:
    with pytest.raises(ValueError, match="sequence schema"):
        build_model("hyformer", [], dim=0, hidden_dims=())


def test_lightgcn_performs_graph_propagation() -> None:
    model = build_model(
        "lightgcn", [5, 6], dim=4, hidden_dims=(8,),
        edge_users=torch.tensor([1, 1, 2]), edge_items=torch.tensor([1, 2, 2]),
        lightgcn_layers=2,
    )
    propagated_user, propagated_item = model.propagate()
    assert not torch.allclose(propagated_user, model.user.weight)
    assert not torch.allclose(propagated_item, model.item.weight)

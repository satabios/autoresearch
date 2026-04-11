import pytest

from octopus.sharding import get_strategy
from octopus.sharding.tensor_parallel import TensorParallelStrategy
from octopus.sharding.pipeline_parallel import PipelineParallelStrategy


class TestStrategyRegistry:
    def test_get_tp_strategy(self):
        s = get_strategy("tp")
        assert isinstance(s, TensorParallelStrategy)

    def test_get_pp_strategy(self):
        s = get_strategy("pp")
        assert isinstance(s, PipelineParallelStrategy)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown sharding strategy"):
            get_strategy("invalid")  # type: ignore


class TestTensorParallelStrategy:
    def test_compute_gpus_needed(self):
        s = TensorParallelStrategy()
        # 100GB model * 1.10 overhead / 40GB per GPU = ceil(2.75) = 3
        assert s.compute_gpus_needed(100.0, 40.0) == 3

    def test_compute_gpus_needed_exact_fit(self):
        s = TensorParallelStrategy()
        # 80GB * 1.10 = 88, / 44 = 2.0 -> 2
        assert s.compute_gpus_needed(80.0, 44.0) == 2

    def test_compute_gpus_needed_single(self):
        s = TensorParallelStrategy()
        # 10GB * 1.10 = 11, / 80 = 0.1375 -> 1
        assert s.compute_gpus_needed(10.0, 80.0) == 1


class TestPipelineParallelStrategy:
    def test_compute_gpus_needed(self):
        s = PipelineParallelStrategy()
        # 100GB / 40GB = 2.5 -> 3
        assert s.compute_gpus_needed(100.0, 40.0) == 3

    def test_decompose_model_with_layers(self):
        import torch.nn as nn

        class FakeTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(100, 32)
                self.layers = nn.ModuleList([nn.Linear(32, 32) for _ in range(6)])
                self.lm_head = nn.Linear(32, 100)

            def forward(self, x):
                x = self.embed_tokens(x)
                for layer in self.layers:
                    x = layer(x)
                return self.lm_head(x)

        model = FakeTransformer()
        layers, emb, head = PipelineParallelStrategy._decompose_model(model)

        assert len(layers) == 6
        assert emb is model.embed_tokens
        assert head is model.lm_head

    def test_decompose_model_gpt2_style(self):
        import torch.nn as nn

        class FakeGPT2(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = nn.Module()
                self.transformer.wte = nn.Embedding(100, 32)
                self.transformer.h = nn.ModuleList([nn.Linear(32, 32) for _ in range(4)])
                self.lm_head = nn.Linear(32, 100)

            def forward(self, x):
                return x

        model = FakeGPT2()
        layers, emb, head = PipelineParallelStrategy._decompose_model(model)

        assert len(layers) == 4
        assert emb is model.transformer.wte
        assert head is model.lm_head

from collections import deque
from unittest.mock import MagicMock, patch

import pytest


class MockObjectRef:
    """Stands in for a ray.ObjectRef."""

    def __init__(self, value):
        self.value = value


class TestOrderedResultIterator:
    def test_yields_in_order(self):
        """Results should come back in submission order."""
        from octopus.iterator import OrderedResultIterator

        pool = MagicMock()
        refs = [MockObjectRef(i) for i in range(5)]
        pool.num_workers = 2
        pool.submit.side_effect = refs

        dataset = iter(range(5))

        with patch("octopus.iterator.ray") as mock_ray:
            mock_ray.get = lambda ref: ref.value
            mock_ray.exceptions = MagicMock()

            it = OrderedResultIterator(pool, dataset, prefetch=3)
            results = list(it)

        assert results == [0, 1, 2, 3, 4]

    def test_empty_dataset(self):
        """Empty dataset should produce empty results."""
        from octopus.iterator import OrderedResultIterator

        pool = MagicMock()
        pool.num_workers = 2

        with patch("octopus.iterator.ray"):
            it = OrderedResultIterator(pool, iter([]), prefetch=2)
            results = list(it)

        assert results == []


class TestUnorderedResultIterator:
    def test_yields_all_results(self):
        """All results should be returned (order may vary)."""
        from octopus.iterator import UnorderedResultIterator

        pool = MagicMock()
        refs = [MockObjectRef(i) for i in range(5)]
        pool.num_workers = 2
        pool.submit.side_effect = refs

        dataset = iter(range(5))

        with patch("octopus.iterator.ray") as mock_ray:
            mock_ray.exceptions = MagicMock()
            # ray.wait returns the first ref as ready
            def fake_wait(inflight, num_returns=1):
                return [inflight[0]], inflight[1:]
            mock_ray.wait = fake_wait
            mock_ray.get = lambda ref: ref.value

            it = UnorderedResultIterator(pool, dataset, prefetch=3)
            results = list(it)

        assert sorted(results) == [0, 1, 2, 3, 4]

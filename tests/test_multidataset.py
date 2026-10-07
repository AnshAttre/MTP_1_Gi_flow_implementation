"""Focused checks for channel alignment, segment boundaries, and SRG-flow."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from giflow.data.dataset import build_splits
from giflow.data.datasets import DatasetBundle, align_bundles
from giflow.data.graph import gcn_normalize, laplacian, line_graph_adjacency
from giflow.models.giflow import GiFlow


class JointDatasetTests(unittest.TestCase):
    def test_channel_union_and_graph_alignment(self):
        first = DatasetBundle(
            "first", np.ones((2, 30)), np.ones((2, 30)),
            np.array([[0, 1], [1, 0]], dtype=float),
            channel_names=("FP1", "FP2"),
        )
        second = DatasetBundle(
            "second", np.full((3, 20), 2.0), np.ones((3, 20)),
            np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float),
            channel_names=("FP2", "FZ", "O1"),
        )

        aligned, adjacency, names = align_bundles([first, second])

        self.assertEqual(names, ("FP1", "FP2", "FZ", "O1"))
        self.assertEqual(aligned[0].x.shape, (4, 30))
        self.assertTrue(np.all(aligned[0].observed_mask[2:] == 0))
        self.assertEqual(adjacency[0, 1], 1.0)
        self.assertEqual(adjacency[1, 2], 1.0)
        self.assertEqual(adjacency[2, 3], 1.0)

    def test_windows_respect_recording_segments(self):
        values = np.ones((2, 30))
        splits = build_splits(
            values,
            np.ones_like(values),
            window=5,
            stride=1,
            segments=((0, 15), (15, 30)),
            dataset_id=3,
        )
        train = splits[0]
        self.assertTrue(all(start + 5 <= 15 or start >= 15 for start in train.indices))
        self.assertEqual(train[0]["dataset_id"].item(), 3)

    def test_srg_flow_loss_and_sampling(self):
        nodes, steps = 6, 5
        adj_s = np.ones((nodes, nodes)) - np.eye(nodes)
        adj_t = line_graph_adjacency(steps)
        model = GiFlow(
            adj_s=adj_s,
            lap_s=laplacian(adj_s),
            lap_t=laplacian(adj_t),
            adj_s_norm=gcn_normalize(adj_s),
            adj_t_norm=gcn_normalize(adj_t),
            hidden=8,
            emb_dim=4,
            n_mp_layers=1,
            use_srg_guidance=True,
        )
        node_mask = torch.tensor([
            [1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1],
        ], dtype=torch.float32)
        truth = torch.randn(2, nodes, steps) * node_mask[:, :, None]
        mask = (torch.rand_like(truth) > 0.3).float() * node_mask[:, :, None]
        loss, parts = model.loss(
            truth, mask, mask, return_components=True, node_mask=node_mask
        )
        loss.backward()
        prediction = model.impute(truth, mask, n_steps=2, node_mask=node_mask)

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(parts["residual_loss"]))
        self.assertTrue(torch.isfinite(parts["smm_regularization"]))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.all(prediction * (1.0 - node_mask[:, :, None]) == 0))
        self.assertTrue(torch.allclose(prediction * mask, truth * mask, atol=1e-5))


if __name__ == "__main__":
    unittest.main()

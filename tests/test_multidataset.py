"""Focused checks for channel alignment, segment boundaries, and SRG-flow."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from giflow.data.dataset import build_splits, make_eval_mask, make_loaders
from giflow.data.datasets import DatasetBundle, align_bundles
from giflow.data.seed_iv import load_seed_iv
from giflow.data.graph import gcn_normalize, laplacian, line_graph_adjacency
from giflow.models.giflow import GiFlow
from giflow.train import TrainConfig, evaluate


class JointDatasetTests(unittest.TestCase):
    def test_named_eval_masks_match_after_channel_reordering_and_padding(self):
        base_ids = ("C", "A", "B")
        base_observed = np.ones((3, 90), dtype=np.float32)
        aligned_ids = ("B", "D", "C", "A")
        aligned_observed = np.ones((4, 90), dtype=np.float32)
        aligned_observed[1] = 0

        for strategy in ("point", "block", "channel", "stream"):
            with self.subTest(strategy=strategy):
                base = make_eval_mask(
                    base_observed, strategy, 0.5, seed=17, channel_ids=base_ids
                )
                aligned = make_eval_mask(
                    aligned_observed, strategy, 0.5, seed=17,
                    channel_ids=aligned_ids,
                )
                for channel in base_ids:
                    base_row = base[base_ids.index(channel)]
                    aligned_row = aligned[aligned_ids.index(channel)]
                    np.testing.assert_array_equal(base_row, aligned_row)

    def test_seed_iv_mat_trials_filter_subject_and_cap_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for session in ("1", "2"):
                session_dir = root / session
                session_dir.mkdir()
                savemat(
                    session_dir / "7_20210101.mat",
                    {"mz_eeg1": np.ones((62, 30), dtype=np.float32)},
                )
                savemat(
                    session_dir / "8_20210102.mat",
                    {"tyc_eeg1": np.full((62, 30), 2, dtype=np.float32)},
                )

            bundle = load_seed_iv(
                root=root,
                window=5,
                subjects=("07",),
                max_windows_per_subject=3,
                verbose=False,
            )

        self.assertEqual(bundle.x.shape, (62, 15))
        self.assertEqual(bundle.segment_group_ids, ("7", "7", "7"))
        self.assertEqual(len(bundle.segments), 3)
        self.assertTrue(np.all(bundle.x == 1.0))

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

    def test_subject_split_keeps_groups_disjoint_and_scaler_train_only(self):
        nodes, samples_per_group = 2, 40
        chunks = []
        segments = []
        group_ids = []
        for group in range(8):
            value = 1_000_000.0 if group == 7 else float(group * 10)
            start = group * samples_per_group
            chunks.append(np.full((nodes, samples_per_group), value))
            segments.append((start, start + samples_per_group))
            group_ids.append(str(group))
        values = np.concatenate(chunks, axis=1)
        train, validation, test, scaler, info = build_splits(
            values,
            np.ones_like(values),
            window=5,
            stride=5,
            rho=0.2,
            seed=0,
            segments=tuple(segments),
            segment_group_ids=tuple(group_ids),
            split_mode="subject",
            split_seed=17,
            test_groups=("7",),
        )

        def groups_for(dataset):
            return {
                group_ids[start // samples_per_group]
                for start in dataset.indices
            }

        train_groups = groups_for(train)
        validation_groups = groups_for(validation)
        test_groups = groups_for(test)
        self.assertEqual(test_groups, {"7"})
        self.assertEqual(test[0]["group_id"], "7")
        self.assertFalse(train_groups & validation_groups)
        self.assertFalse(train_groups & test_groups)
        self.assertFalse(validation_groups & test_groups)
        self.assertNotIn("7", info["subject_groups"]["train"])
        self.assertLess(scaler.mean, 1000.0)

    def test_window_split_reports_subject_level_metrics(self):
        values = np.random.default_rng(0).normal(size=(2, 60)).astype(np.float32)
        segments = ((0, 20), (20, 40), (40, 60))
        group_ids = ("a", "b", "c")
        splits = build_splits(
            values,
            np.ones_like(values),
            window=5,
            stride=5,
            rho=0.2,
            seed=1,
            segments=segments,
            segment_group_ids=group_ids,
            split_mode="window",
        )

        class PerfectReconstruction(torch.nn.Module):
            def impute(self, x_true, cond_mask, n_steps, timestamps=None, node_mask=None):
                return x_true

        _, _, test_loader = make_loaders(*splits[:3], batch_size=4)
        metrics = evaluate(
            PerfectReconstruction(),
            test_loader,
            splits[3],
            TrainConfig(euler_steps=1),
            "cpu",
            dataset_names=("test",),
        )
        self.assertEqual(metrics["subject_summary"]["n_subjects"], 1)
        self.assertEqual(set(metrics["per_subject"]), {"c"})
        self.assertEqual(metrics["nmse"], 0.0)

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

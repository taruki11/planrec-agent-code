import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from preprocess_amazon_2014 import can_reuse_processed
from planrec.data.amazon import (
    build_deterministic_id_mapping,
    filter_iterative_kcore,
    parse_raw_interactions,
    prepare_leave_one_out_splits,
    process_amazon_beauty_pipeline,
    sort_user_sequences,
)


class AmazonDataPipelineTests(unittest.TestCase):
    def test_standard_preprocess_reuse_requires_current_protocol(self) -> None:
        current = {
            "num_users": 500,
            "raw_sha256": "raw-sha",
            "padding_side": "right",
            "protocol_version": "planrec_loo_rightpad_v1",
        }
        self.assertTrue(can_reuse_processed(current, "raw-sha"))

        stale = dict(current)
        stale.pop("protocol_version")
        self.assertFalse(can_reuse_processed(stale, "raw-sha"))

        wrong_padding = dict(current, padding_side="left")
        self.assertFalse(can_reuse_processed(wrong_padding, "raw-sha"))

    def test_parse_amazon_2014_python_literal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_file = Path(tmp_dir) / "reviews_Beauty_5.json"
            raw_file.write_text(
                "{'reviewerID': 'U1', 'asin': 'I1', 'unixReviewTime': 123}\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_raw_interactions(raw_file), [("U1", "I1", 123)])

    def test_iterative_5core_filters_low_frequency(self) -> None:
        # Create user U1 with 6 items, U2 with 2 items
        interactions = [
            ("U1", "I1", 100),
            ("U1", "I2", 101),
            ("U1", "I3", 102),
            ("U1", "I4", 103),
            ("U1", "I5", 104),
            ("U1", "I6", 105),
            ("U2", "I1", 106),
            ("U2", "I2", 107),
        ]
        # Filter with min_freq=3
        filtered = filter_iterative_kcore(interactions, min_user_freq=3, min_item_freq=1)
        users = {u for u, _, _ in filtered}
        self.assertIn("U1", users)
        self.assertNotIn("U2", users)

    def test_deterministic_id_mapping_is_stable(self) -> None:
        interactions = [
            ("U_B", "I_Z", 10),
            ("U_A", "I_A", 20),
            ("U_A", "I_Z", 30),
        ]
        u_map1, i_map1 = build_deterministic_id_mapping(interactions)
        u_map2, i_map2 = build_deterministic_id_mapping(interactions)

        self.assertEqual(u_map1, u_map2)
        self.assertEqual(i_map1, i_map2)

        # Alphabetical sorting check: U_A -> 1, U_B -> 2
        self.assertEqual(u_map1["U_A"], 1)
        self.assertEqual(u_map1["U_B"], 2)
        self.assertEqual(i_map1["I_A"], 1)
        self.assertEqual(i_map1["I_Z"], 2)

    def test_leave_one_out_splits_without_future_leakage(self) -> None:
        # User U1 with sequence [10, 20, 30, 40, 50] (len=5)
        # Test target must be 50, val target must be 40, train targets must be 20, 30
        user_seqs = {
            1: [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]
        }
        train_ds, val_ds, test_ds = prepare_leave_one_out_splits(user_seqs, max_seq_len=4)

        # Test evaluation check
        test_x, test_y = test_ds[0]
        self.assertEqual(test_y.item(), 50)
        self.assertEqual(test_x.tolist(), [10, 20, 30, 40])

        # Validation check
        val_x, val_y = val_ds[0]
        self.assertEqual(val_y.item(), 40)
        self.assertEqual(val_x.tolist(), [10, 20, 30, 0])

        # Training samples check (targets: 20, 30)
        train_targets = [y.item() for _, y in train_ds]
        self.assertEqual(train_targets, [20, 30])

        # Train sample 1: target=20, prefix=[10] -> right padded [10, 0, 0, 0]
        train_x1, train_y1 = train_ds[0]
        self.assertEqual(train_y1.item(), 20)
        self.assertEqual(train_x1.tolist(), [10, 0, 0, 0])

        # Train sample 2: target=30, prefix=[10, 20] -> right padded [10, 20, 0, 0]
        train_x2, train_y2 = train_ds[1]
        self.assertEqual(train_y2.item(), 30)
        self.assertEqual(train_x2.tolist(), [10, 20, 0, 0])

    def test_sequence_sorting_stability(self) -> None:
        interactions = [
            ("U1", "I2", 100),
            ("U1", "I1", 100),  # Tie in timestamp
            ("U1", "I3", 90),   # Earlier
        ]
        u_map, i_map = build_deterministic_id_mapping(interactions)
        sorted_seqs = sort_user_sequences(interactions, u_map, i_map)

        # Items should be ordered by ts (90 first), then item_id for ties
        item_ids = [item_id for _, item_id in sorted_seqs[u_map["U1"]]]
        self.assertEqual(item_ids, [i_map["I3"], i_map["I1"], i_map["I2"]])

    def test_pipeline_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_file = tmp_path / "sample_beauty.jsonl"

            # Create mock interactions (5 users, each with 5 items to satisfy 5-core)
            records = []
            for u_idx in range(1, 6):
                for i_idx in range(1, 6):
                    records.append({
                        "user_id": f"USER_{u_idx}",
                        "parent_asin": f"ITEM_{i_idx}",
                        "timestamp": 1000 + u_idx * 10 + i_idx,
                    })

            with open(raw_file, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")

            stats = process_amazon_beauty_pipeline(
                raw_filepath=raw_file,
                output_dir=tmp_path,
                min_user_freq=5,
                min_item_freq=5,
                max_seq_len=10,
            )

            self.assertEqual(stats["num_users"], 5)
            self.assertEqual(stats["num_items"], 5)
            self.assertEqual(stats["num_test_samples"], 5)
            self.assertEqual(stats["num_val_samples"], 5)
            self.assertEqual(stats["num_train_samples"], 10)  # 5 users * (5-3) = 10
            self.assertEqual(stats["protocol_version"], "planrec_loo_rightpad_v1")
            self.assertEqual(stats["padding_side"], "right")

            processed_dir = tmp_path / "processed"
            self.assertTrue((processed_dir / "data_stats.json").exists())
            self.assertTrue((processed_dir / "id_mapping.json").exists())
            self.assertTrue((processed_dir / "datasets.pt").exists())


if __name__ == "__main__":
    unittest.main()

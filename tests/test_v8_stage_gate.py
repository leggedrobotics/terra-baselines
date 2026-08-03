import hashlib
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.euler_v8_deep_xattn_v1 import stage_gate

CAPABILITY = stage_gate.CAPABILITY_IDS
CORE = tuple(
    [f"v7-fnd-{index}-adjacent" for index in range(6)]
    + [f"v7-trn-{index}-adjacent" for index in range(7)]
)
FAMILY = {
    name: ("foundation" if name.startswith("v7-fnd") else "trench") for name in CORE
}


def evaluation_record(update, counts, *, checkpoint=None, integrity=True):
    checkpoint = checkpoint or f"/tmp/checkpoint_{update:06d}.pkl"
    return {
        "schema": stage_gate.EVAL_SCHEMA,
        "completion_contract": stage_gate.COMPLETION_CONTRACT,
        "checkpoint": checkpoint,
        "checkpoint_sha256": "a" * 64,
        "checkpoint_update": update,
        "treatment_fingerprint": {
            "contract": {
                "run": {"name": "v8_deep"},
                "architecture": {
                    "model_size": "medium",
                    "model_core": "mlp",
                    "map_encoder": "resnet_spatial_8x8_se",
                    "encoder_compute_dtype": "bfloat16",
                    "attention_compute_dtype": "encoder",
                    "token_mixer_residual_init_scale": 0.0,
                    "critic_hidden_dims": [512, 256],
                    "resnet_stage_channels": [24, 48, 64, 96],
                    "resnet_blocks_per_stage": [2, 2, 3, 3],
                    "loaded_max": 100,
                },
            }
        },
        "horizon": 450,
        "deterministic": True,
        "policy_mode": "deterministic",
        "exact_manifest_enumeration": True,
        "split": "promotion",
        "summary": {
            "overall": {"episodes": 16 * len(counts)},
            "by_primary_cell": {
                name: {"episodes": 16, "successes": successes}
                for name, successes in counts.items()
            },
            "integrity": {"passed": integrity},
        },
    }


def bind_frozen_treatment(records, bank, panel, stage="capability", seed=20260730):
    selected = set(CAPABILITY)
    if stage != "capability":
        selected.update(bank["core_ids"])
    train_by_id = {entry["condition_id"]: entry for entry in bank["dataset"]["train"]}
    architecture = records[0]["treatment_fingerprint"]["contract"]["architecture"]
    contract = {
        "schema": "terra_fixed_bank_treatment_v1",
        "run": {
            "name": f"v8_screen_{stage}_g_deep_v8_dense_warm_s{seed}",
            "seed": seed,
            "config_name": "G-V8-FIXED",
            "accepted_bank_arm": "G-UNIFORM",
        },
        "bank": {
            "terra_revision": stage_gate.TERRA_REVISION,
            "environment_protocol_sha256": bank["environment_protocol_sha256"],
            "source_registry_sha256": bank["source_registry_sha256"],
        },
        "ppo": {
            "num_devices": 4,
            "num_envs_per_device": 1024,
            "num_steps": 32,
            "update_epochs": 2,
            "num_minibatches": 32,
            "lr": 0.0003,
            "gamma": 0.9984,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "vf_coef": 2.0,
            "max_grad_norm": 0.5,
            "ent_schedule_start": 0.02,
            "ent_schedule_end": 0.005,
            "ent_schedule_steps": 10000,
            "use_value_clip": False,
            "flat_minibatch_shuffle": True,
        },
        "reward_action": {
            "agent_types": [0],
            "action_types": [0],
            "relocation_progress_mult": 1.5,
            "curriculum_levels": [
                {
                    "maps_path": train_by_id[name]["maps_path"],
                    "max_steps_in_episode": 450,
                    "rewards_type": 0,
                    "apply_trench_rewards": False,
                }
                for name in sorted(selected)
            ],
        },
        "sampler": {
            "enabled": True,
            "rule": "fixed",
            "update_interval": 150,
            "uniform_floor": 0.20,
            "mastery_threshold": 0.75,
            "temperature": 0.25,
            "min_episodes": 20,
            "competence_ema": 0.30,
            "max_mass": 0.15,
            "seed": seed,
        },
        "architecture": architecture,
    }
    fingerprint = {
        "contract": contract,
        "sha256": hashlib.sha256(
            json.dumps(
                contract, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest(),
    }
    accepted_bank = {
        "schema": "terra_curriculum_loader_bank_v1",
        "terra_revision": stage_gate.TERRA_REVISION,
        "environment_protocol_sha256": bank["environment_protocol_sha256"],
        "source_registry_sha256": bank["source_registry_sha256"],
        "diagnostic_control": False,
        "diagnostic_contract_sha256": None,
    }
    for record in records:
        record.update(
            {
                "treatment_fingerprint": fingerprint,
                "bank_root": panel["bank_root"],
                "accepted_bank": accepted_bank,
                "manifest": panel["manifest"],
                "manifest_sha256": panel["manifest_sha256"],
                "stratum": panel["stratum"],
                "seed": seed,
                "reset_verification": panel["reset_verification"],
            }
        )
    return {
        "stage": stage,
        "conditions": sorted(selected),
        "declared_weights": [1.0 / len(selected)] * len(selected),
        "probabilities": [1.0 / len(selected)] * len(selected),
        "maps_per_condition": 96,
        "sha256": "4" * 64,
    }


class V8StageGateTest(unittest.TestCase):
    def test_evaluation_is_bound_to_panel_seed_and_ppo_contract(self):
        train = [
            {"condition_id": name, "maps_path": f"train/{name}"} for name in CAPABILITY
        ]
        bank = {
            "core_ids": (),
            "dataset": {"train": train},
            "environment_protocol_sha256": "1" * 64,
            "source_registry_sha256": "2" * 64,
        }
        panel = {
            "bank_root": "/tmp/frozen-bank",
            "manifest": "/tmp/frozen-bank/promotion/manifest.jsonl",
            "manifest_sha256": "3" * 64,
            "stratum": "capability",
            "reset_verification": {"passed": True, "slots": 32},
        }
        records = [
            evaluation_record(update, dict.fromkeys(CAPABILITY, 12))
            for update in stage_gate.STAGE_UPDATES["capability"]
        ]
        sampling = bind_frozen_treatment(records, bank, panel)
        run_contract = {"seed": "20260730"}
        with patch.object(stage_gate, "_read_json", return_value=records):
            stage_gate.validate_evaluation(
                Path("/tmp/promotion.json"),
                "capability",
                "G-DEEP-V8-DENSE-WARM",
                bank,
                run_contract,
                panel,
                sampling,
            )
            records[-1]["manifest_sha256"] = "4" * 64
            with self.assertRaisesRegex(ValueError, "manifest_sha256"):
                stage_gate.validate_evaluation(
                    Path("/tmp/promotion.json"),
                    "capability",
                    "G-DEEP-V8-DENSE-WARM",
                    bank,
                    run_contract,
                    panel,
                    sampling,
                )

    def test_promote_revalidates_both_latest_checkpoints(self):
        records = [
            evaluation_record(update, dict.fromkeys(CAPABILITY, 12))
            for update in stage_gate.STAGE_UPDATES["capability"]
        ]
        run_contract = {
            "terra_baselines_revision": "1" * 40,
            "parent_checkpoint_sha256": "2" * 64,
            "initial_checkpoint_sha256": "3" * 64,
        }
        candidate = {
            "path": records[-1]["checkpoint"],
            "checkpoint_sha256": records[-1]["checkpoint_sha256"],
            "next_update": 2000,
            "architecture": "deep-se",
            "map_encoder": "resnet_spatial_8x8_se",
            "curriculum_stage": "capability",
            "warm_start_from": "/tmp/parent.pkl",
            "teacher_checkpoint": "/tmp/teacher.pkl",
        }
        args = SimpleNamespace(
            bank_root=Path("/tmp/bank"),
            stage="capability",
            arm="G-DEEP-V8-DENSE-WARM",
            prior_receipt=None,
            capability=None,
            run_contract=Path("/tmp/run_contract.env"),
            promotion=Path("/tmp/promotion.json"),
        )
        with (
            patch.object(stage_gate, "load_bank_contract", return_value={}),
            patch.object(
                stage_gate,
                "stage_sampling_contract",
                return_value={"conditions": list(CAPABILITY)},
            ),
            patch.object(stage_gate, "parse_run_contract", return_value=run_contract),
            patch.object(stage_gate, "validate_run_contract"),
            patch.object(stage_gate, "panel_contract", return_value={}),
            patch.object(stage_gate, "validate_evaluation", return_value=records),
            patch.object(stage_gate, "validate_panel_conditions"),
            patch.object(
                stage_gate,
                "validate_candidate_checkpoint",
                side_effect=[{**candidate, "next_update": 1500}, candidate],
            ) as validate_candidate,
            patch.object(stage_gate, "sha256_file", return_value="4" * 64),
        ):
            receipt = stage_gate.promote(args)

        self.assertTrue(receipt["passed"])
        self.assertEqual(validate_candidate.call_count, 2)
        self.assertIs(validate_candidate.call_args_list[0].args[0], records[-2])
        self.assertIs(validate_candidate.call_args_list[1].args[0], records[-1])

    def test_fixed_sampler_state_binds_condition_order_and_probabilities(self):
        sampling = {
            "conditions": list(CAPABILITY),
            "declared_weights": [0.5, 0.5],
            "probabilities": [0.5, 0.5],
            "maps_per_condition": 96,
        }
        state = {
            "schema": "terra_pooled_condition_sampler_state_v1",
            "conditions": list(CAPABILITY),
            "settings": {
                "rule": "fixed",
                "update_interval": 150,
                "uniform_floor": 0.20,
                "mastery_threshold": 0.75,
                "temperature": 0.25,
                "min_episodes": 20,
                "competence_ema": 0.30,
                "max_mass": 0.15,
                "seed": 20260730,
            },
            "maps_per_condition": [96, 96],
            "labels": {name: {"sampling_weight": 0.5} for name in CAPABILITY},
            "probabilities": [0.5, 0.5],
        }
        stage_gate._validate_sampler_state(
            state, sampling, 20260730, Path("/tmp/checkpoint.pkl")
        )
        state["probabilities"] = [0.6, 0.4]
        with self.assertRaisesRegex(ValueError, "probabilities changed"):
            stage_gate._validate_sampler_state(
                state, sampling, 20260730, Path("/tmp/checkpoint.pkl")
            )

    def test_capability_uses_latest_pair_and_twelve_of_sixteen_floor(self):
        records = [
            evaluation_record(500, dict.fromkeys(CAPABILITY, 16)),
            evaluation_record(1000, dict.fromkeys(CAPABILITY, 16)),
            evaluation_record(1500, dict.fromkeys(CAPABILITY, 12)),
            evaluation_record(2000, dict.fromkeys(CAPABILITY, 11)),
        ]
        decision = stage_gate.decide_capability(records)
        self.assertFalse(decision["passed"])
        self.assertEqual(decision["new_thresholds"], {})

        records[-1] = evaluation_record(2000, dict.fromkeys(CAPABILITY, 12))
        decision = stage_gate.decide_capability(records)
        self.assertTrue(decision["passed"])
        self.assertEqual(decision["frozen_thresholds"], dict.fromkeys(CAPABILITY, 12))

    def test_nearby_requires_family_cell_and_capability_retention(self):
        main = [
            evaluation_record(update, dict.fromkeys(CORE, 13))
            for update in stage_gate.STAGE_UPDATES["nearby"]
        ]
        capability = [
            evaluation_record(
                update,
                dict.fromkeys(CAPABILITY, 12),
                checkpoint=main[index]["checkpoint"],
            )
            for index, update in enumerate(stage_gate.STAGE_UPDATES["nearby"])
        ]
        prior = {"retention": {"frozen_thresholds": dict.fromkeys(CAPABILITY, 12)}}
        decision = stage_gate.decide_nearby(main, capability, prior, CORE, FAMILY)
        self.assertTrue(decision["passed"])
        self.assertEqual(
            decision["family_totals"][-1], {"foundation": 78, "trench": 91}
        )

        for record in capability[-2:]:
            record["summary"]["by_primary_cell"][CAPABILITY[0]]["successes"] = 11
        decision = stage_gate.decide_nearby(main, capability, prior, CORE, FAMILY)
        self.assertFalse(decision["passed"])
        self.assertTrue(decision["rollback_triggered"])

        for record in capability[-2:]:
            for condition_id in CAPABILITY:
                record["summary"]["by_primary_cell"][condition_id]["successes"] = 12
        capability[-2]["summary"]["by_primary_cell"][CAPABILITY[0]]["successes"] = 11
        capability[-1]["summary"]["by_primary_cell"][CAPABILITY[1]]["successes"] = 11
        decision = stage_gate.decide_nearby(main, capability, prior, CORE, FAMILY)
        self.assertTrue(decision["rollback_triggered"])

        for record in capability:
            for condition_id in CAPABILITY:
                record["summary"]["by_primary_cell"][condition_id]["successes"] = 12
        capability[0]["summary"]["by_primary_cell"][CAPABILITY[0]]["successes"] = 11
        capability[1]["summary"]["by_primary_cell"][CAPABILITY[1]]["successes"] = 11
        decision = stage_gate.decide_nearby(main, capability, prior, CORE, FAMILY)
        self.assertFalse(decision["passed"])
        self.assertTrue(decision["rollback_triggered"])
        self.assertEqual(decision["rollback_updates"], [500, 1000])

    def test_nearby_rejects_capability_checkpoint_mismatch(self):
        main = [
            evaluation_record(update, dict.fromkeys(CORE, 13))
            for update in stage_gate.STAGE_UPDATES["nearby"]
        ]
        capability = [
            evaluation_record(
                update,
                dict.fromkeys(CAPABILITY, 12),
                checkpoint=f"/tmp/capability_{update:06d}.pkl",
            )
            for update in stage_gate.STAGE_UPDATES["nearby"]
        ]
        prior = {"retention": {"frozen_thresholds": dict.fromkeys(CAPABILITY, 12)}}
        with self.assertRaisesRegex(ValueError, "different checkpoints"):
            stage_gate.decide_nearby(main, capability, prior, CORE, FAMILY)

    def test_receipt_and_checkpoint_provenance_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_path = root / "prior.json"
            receipt = {
                "schema": stage_gate.SCHEMA,
                "passed": True,
                "stage": "capability",
                "next_stage": "nearby",
                "arm": "G-DEEP-V8-DENSE-WARM",
                "release_id": stage_gate.RELEASE_ID,
                "terra_revision": stage_gate.TERRA_REVISION,
                "bank_archive_sha256": stage_gate.BANK_ARCHIVE_SHA256,
                "bank_dataset_sha256": stage_gate.BANK_DATASET_SHA256,
                "training_mixture_sha256": stage_gate.TRAINING_MIXTURE_SHA256,
                "sampling": {
                    "stage": "capability",
                    "sha256": stage_gate.STAGE_SAMPLING_SHA256["capability"],
                },
                "scheduled_updates": list(stage_gate.STAGE_UPDATES["capability"]),
                "evaluated_pair": [1500, 2000],
                "candidate": {
                    "path": str(
                        stage_gate.REMOTE_RUN_ROOT
                        / "revision/screen/capability/s1/G-DEEP-V8-DENSE-WARM"
                        / "checkpoints/candidate.pkl"
                    ),
                    "checkpoint_sha256": "b" * 64,
                    "next_update": 2000,
                    "architecture": "deep-se",
                    "map_encoder": "resnet_spatial_8x8_se",
                    "curriculum_stage": "capability",
                },
                "mastery": {name: [13, 12] for name in CAPABILITY},
                "integrity_pair": [True, True],
                "retention": {
                    "frozen_thresholds": dict.fromkeys(CAPABILITY, 12),
                    "rollback_triggered": False,
                },
            }
            receipt_path.write_text(json.dumps(receipt))
            stage_gate.validate_prior_receipt(
                receipt_path, "G-DEEP-V8-DENSE-WARM", "capability"
            )
            receipt["retention"]["frozen_thresholds"][CAPABILITY[0]] = 13
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "thresholds were modified"):
                stage_gate.validate_prior_receipt(
                    receipt_path, "G-DEEP-V8-DENSE-WARM", "capability"
                )

            checkpoint_path = root / "checkpoint.pkl"
            checkpoint = {
                "next_update": 2000,
                "model": {},
                "train_config": {
                    "accepted_bank": {
                        "release_id": stage_gate.RELEASE_ID,
                        "terra_revision": stage_gate.TERRA_REVISION,
                        "curriculum_stage": "capability",
                    },
                    "model_size": "medium",
                    "model_core": "mlp",
                    "map_encoder": "resnet_spatial_8x8_se",
                    "encoder_compute_dtype": "bfloat16",
                    "attention_compute_dtype": "encoder",
                    "token_mixer_residual_init_scale": 0.0,
                    "critic_hidden_dims": (512, 256),
                    "resnet_stage_channels": (24, 48, 64, 96),
                    "resnet_blocks_per_stage": (2, 2, 3, 3),
                    "loaded_max": 100,
                    "warm_start_from": "/tmp/parent.pkl",
                    "teacher_checkpoint": "/tmp/teacher.pkl",
                },
            }
            with checkpoint_path.open("wb") as handle:
                pickle.dump(checkpoint, handle)
            record = evaluation_record(
                2000,
                dict.fromkeys(CAPABILITY, 12),
                checkpoint=str(checkpoint_path),
            )
            record["checkpoint_sha256"] = stage_gate.sha256_file(checkpoint_path)
            stage_gate.validate_candidate_checkpoint(
                record, "capability", "G-DEEP-V8-DENSE-WARM"
            )
            checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"mutation")
            with self.assertRaisesRegex(ValueError, "hash does not match"):
                stage_gate.validate_candidate_checkpoint(
                    record, "capability", "G-DEEP-V8-DENSE-WARM"
                )


if __name__ == "__main__":
    unittest.main()

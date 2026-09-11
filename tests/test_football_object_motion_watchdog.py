import json
from pathlib import Path
from tempfile import TemporaryDirectory

from football_object_motion.smoke_contract import (
    RESOLVED_CRITICAL_PARAMETERS,
    SCHEMA,
    build_contract,
    validate_smoke_artifact,
)


REPO = Path(__file__).resolve().parents[1]


def _valid_artifact(contract_hash: str, generated_at: float = 101.0) -> dict:
    return {
        "state": "passed",
        "schema": SCHEMA,
        "generated_at_unix": generated_at,
        "contract_hash": contract_hash,
        "steps_completed": 2,
        "experiment_status": "teacher-supervised exploratory",
        "peak_cuda_memory_gib": 12.0,
        "pair_count": 3,
        "first_pair": {
            "legal": True,
            "same_pair_id": True,
            "same_video": True,
            "masks_valid": True,
            "coverage_min": 1.0,
        },
        "steps": [
            {
                "alpha": 0.0,
                "identity_bitwise": True,
                "anchor_identity_max_abs_error": 0.0,
                "detector_event_grad_norm": 0.0,
                "adapter_grad_norms": {"clip_residual": 0.2, "relation": 0.0, "ball_lora": 0.4},
            },
            {
                "alpha": 0.10,
                "adapter_grad_norms": {"frame_residual": 0.2, "relation": 0.3},
                "effective_residual": {"clip_abs_mean": 0.01, "frame_abs_mean": 0.02},
            },
        ],
    }


def test_v4_smoke_launcher_uses_the_formal_contract_and_output():
    source = (REPO / "scripts/run_object_motion_smoke_after_epoch2.sh").read_text()
    assert "football_object_motion.smoke_contract contract" in source
    assert "football_object_motion.smoke" in source
    assert "vitl16_d7c_object_motion_v4_causal33_720p" in source


def test_two_stage_smoke_validation_and_contract_rejection():
    payload = _valid_artifact("contract-a")
    assert validate_smoke_artifact(
        payload,
        expected_contract_hash="contract-a",
        smoke_started_at=100.0,
        now=102.0,
    ) == (True, "ok")
    invalid = json.loads(json.dumps(payload))
    invalid["steps"][1]["adapter_grad_norms"]["frame_residual"] = 1e-7
    valid, reason = validate_smoke_artifact(
        invalid,
        expected_contract_hash="contract-a",
        smoke_started_at=100.0,
        now=102.0,
    )
    assert not valid and "step2" in reason
    valid, reason = validate_smoke_artifact(
        payload,
        expected_contract_hash="contract-b",
        smoke_started_at=100.0,
        now=102.0,
    )
    assert not valid and "contract hash" in reason


def test_contract_hash_changes_when_an_input_changes():
    with TemporaryDirectory() as directory:
        repo = Path(directory)
        relative_files = [
            "scripts/run_object_motion_adapter_v4.sh",
            "scripts/run_object_motion_adapter_v3.sh",
            "scripts/run_object_motion_smoke_after_epoch2.sh",
            "football_object_motion/smoke.py",
            "football_object_motion/ball_backbone.py",
            "football_object_motion/model.py",
            "football_object_motion/train.py",
            "football_object_motion/losses.py",
            "football_object_motion/teacher.py",
            *RESOLVED_CRITICAL_PARAMETERS["model.object_motion.reviewed_negative_manifests"],
        ]
        for relative in relative_files:
            target = repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(relative)
        config = repo / "config.yaml"
        checkpoint = repo / "anchor.pt"
        config.write_text("config-v1")
        checkpoint.write_bytes(b"checkpoint")
        first = build_contract(repo, config, checkpoint)["contract_hash"]
        (repo / "football_object_motion/model.py").write_text("changed-model")
        second = build_contract(repo, config, checkpoint)["contract_hash"]
        assert first != second


def test_only_balllora_fullwindow33_can_launch():
    launcher = (REPO / "scripts/run_object_motion_adapter_v3.sh").read_text()
    v4_launcher = (REPO / "scripts/run_object_motion_adapter_v4.sh").read_text()
    assert "v3_balllora_fullwindow33" in launcher
    assert "v3_fullwindow51" not in launcher
    assert 'GPU_LIST="${GPU_LIST:-0,1,4,6}"' in v4_launcher
    assert 'GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-12}"' in v4_launcher
    assert 'MOTION_FRAMES_PER_SEGMENT="${MOTION_FRAMES_PER_SEGMENT:-11}"' in v4_launcher
    assert 'motion_frames_per_segment="${MOTION_FRAMES_PER_SEGMENT:-11}"' in launcher
    assert '--nproc-per-node="${gpu_count}"' in launcher
    assert 'train.per_gpu_batch_size=${per_gpu_batch_size}' in launcher


def test_smoke_and_launcher_share_production_pair_and_ball_parameters():
    launcher = (REPO / "scripts/run_object_motion_adapter_v3.sh").read_text()
    v4_launcher = (REPO / "scripts/run_object_motion_adapter_v4.sh").read_text()
    smoke = (REPO / "football_object_motion/smoke.py").read_text()
    shared = (
        "model.object_motion.ball_lora_rank=4",
        "model.object_motion.ball_lora_alpha=4.0",
        "model.object_motion.ball_lora_last_blocks=8",
        "model.object_motion.ball_feature_layers=[11,17,20,23]",
        "model.object_motion.ball_topk=4",
        "model.object_motion.ball_temperature=0.25",
        "model.object_motion.require_true_pairs=true",
        "model.object_motion.pair_min_gap_sec=5.0",
        "data.long_video.negative_ratio=5.0",
        "data.long_video.online_simulation.event_sampling_mode=cover_all_once",
        "data.long_video.online_simulation.clean_background_windows_per_epoch=2400",
        "train.object_motion_frame_loss_weight=0.10",
        "train.ball_lora_lr=1.0e-6",
        "train.object_motion_head_lr=2.0e-5",
    )
    assert 'MOTION_FRAMES_PER_SEGMENT="${MOTION_FRAMES_PER_SEGMENT:-11}"' in v4_launcher
    assert 'model.object_motion.frames_per_segment=${motion_frames_per_segment}' in launcher
    for token in shared:
        assert token in launcher, token
        assert token in smoke, token

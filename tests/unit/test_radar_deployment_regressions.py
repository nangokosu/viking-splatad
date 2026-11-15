"""Radar opacity stability and train/render wiring with synthetic inputs and tools."""

import math
import shlex

import pytest
import torch

from tests.unit.test_render_memory_and_resume import (
    FAILED_IN_RENDER_JOB_INFO_TEXT,
    HEAD_SHA,
    RUN_ID,
    RUN_ITERATIONS,
    run_sbatch,
)
from viking_splatad.splatad.radar_lidar_loss import compute_radar_lidar_loss


def test_transparent_supported_gaussian_has_finite_loss_and_opacity_gradient():
    """Extreme transparency must retain the analytic gradient sigma(o) minus one."""
    # Twice log(float32 tiny) makes sigmoid underflow, although the log-loss is finite.
    transparent_logit = math.log(torch.finfo(torch.float32).tiny) * 2
    opacity_logits = torch.tensor([transparent_logit], requires_grad=True)
    result = compute_radar_lidar_loss(
        gaussian_means=torch.zeros((1, 3)),
        opacity_logits=opacity_logits,
        static_gaussian_mask=torch.tensor([True]),
        radar_seed_means=torch.zeros((1, 3)),
        radar_seed_covariances=torch.eye(3).unsqueeze(0),
        loss_weight=1.0,
        gate_quantile=0.5,
        chunk_size=1,
    )
    result.loss.backward()

    # For negative o, -log(sigmoid(o)) = -o + log1p(exp(o)); compute in Python doubles.
    expected_loss = -transparent_logit + math.log1p(math.exp(transparent_logit))
    expected_gradient = math.exp(transparent_logit) / (1 + math.exp(transparent_logit)) - 1
    assert result.loss.item() == pytest.approx(expected_loss)
    assert opacity_logits.grad.item() == pytest.approx(expected_gradient)
    assert result.supported_gaussian_count == 1


@pytest.mark.parametrize("resuming", [False, True])
def test_radar_job_uses_registered_method_directory_and_parser_local_seed_option(tmp_path, resuming):
    """Mocked jobs train or resume and pass the same radar checkpoint to every output step."""
    training_method = "splatad-radar-lidar"
    train_dir = tmp_path / "stage" / "runs" / RUN_ID / "train" / "may_short_grass" / training_method / RUN_ID
    checkpoint_dir = train_dir / "nerfstudio_models"
    checkpoint_dir.mkdir(parents=True)
    (train_dir / "config.yml").write_text("synthetic radar config")
    (checkpoint_dir / f"step-{RUN_ITERATIONS - 1:09d}.ckpt").write_text("")
    extra_environment = {
        "TRAIN_METHOD": training_method,
        "RADAR_INPUTS_DIR": "radar_inputs",
        "RADAR_LIDAR_LOSS_WEIGHT": "1",
        "RADAR_LIDAR_GATE_QUANTILE": "0.5",
        "RADAR_LIDAR_GATE_CHUNK_SIZE": "1",
    }
    if resuming:
        extra_environment.update(RESUME_AFTER_TRAINING="True", RESUME_GIT_SHA=HEAD_SHA)
    completed, calls, job_info = run_sbatch(
        tmp_path, extra_environment, FAILED_IN_RENDER_JOB_INFO_TEXT if resuming else "", True
    )
    assert completed.returncode == 0, completed.stderr
    assert "end_utc:" in job_info
    tokenized_calls = [shlex.split(call) for call in calls]
    train_calls = [arguments for arguments in tokenized_calls if arguments[0] == "ns-train"]
    if resuming:
        assert train_calls == []
    else:
        assert len(train_calls) == 1
        train_arguments = train_calls[0]
        assert train_arguments[1] == training_method
        assert train_arguments.index("--radar-inputs-dir") > train_arguments.index("viking-data")
        seed_option_index = train_arguments.index("--radar-inputs-dir")
        assert train_arguments[seed_option_index + 1] == str(tmp_path / "stage/data/may_short_grass/radar_inputs")
        assert "--pipeline.datamanager.dataparser.radar-inputs-dir" not in train_arguments
    # Both the render-config writer and lidar renderer must load this method's checkpoint config.
    for script_name, option_name in (
        ("12_prepare_rerender.py", "--train-config"),
        ("08_render_lidar_pointcloud.py", "--load-config"),
    ):
        arguments = next(arguments for arguments in tokenized_calls if any(script_name in token for token in arguments))
        assert arguments[arguments.index(option_name) + 1] == str(train_dir / "config.yml")

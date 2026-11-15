"""Method provenance must survive rendering of radar training runs."""

from tests.unit.test_slurm_submit import load_submit_module


def test_render_environment_retains_radar_training_method():
    """A new render job locates the checkpoint under the original method name."""
    submit_module = load_submit_module()
    training_method = "splatad-radar-lidar"
    run_info = {
        "SESSION": "synthetic_session",
        "GIT_SHA": "training_revision",
        "RENDER_DOWNSCALE_FACTOR": "1",
        "TRAIN_METHOD": training_method,
    }
    environment = submit_module.build_render_job_environment(
        "synthetic_run", "synthetic_render", "/stage", run_info,
        "render_revision", {"env_prefix": "/stage/env"},
    )

    assert environment["TRAIN_METHOD"] == training_method
    assert environment["TRAIN_GIT_SHA"] == run_info["GIT_SHA"]
    assert environment["GIT_SHA"] == "render_revision"

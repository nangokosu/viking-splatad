"""Tests of the radar-variant method selection in the Slurm batch scripts.

`scripts/slurm/splatad_run.sbatch` picks, per TRAIN_METHOD, the model flags, the dataparser flags and
the NERFSTUDIO_METHOD_CONFIGS registration. The selection block (from `RADAR_MODEL_ARGS=()` up to the
`ns-train` line) is executed here in a bash subprocess with a controlled environment, so the
expected values come from the interface contract and not from grepping the script text.
`render_only.sbatch` and `install_into_neurad.sh` must carry the same registration string.
"""

import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_SBATCH_PATH = PROJECT_ROOT / "scripts" / "slurm" / "splatad_run.sbatch"
RENDER_SBATCH_PATH = PROJECT_ROOT / "scripts" / "slurm" / "render_only.sbatch"
INSTALL_SCRIPT_PATH = PROJECT_ROOT / "src" / "viking_splatad" / "splatad" / "install_into_neurad.sh"
SPLATAD_PACKAGE_DIR = PROJECT_ROOT / "src" / "viking_splatad" / "splatad"

# Interface contract: method name -> (spec module, spec attribute). The registration string is the
# comma join of these three, in this order.
RADAR_METHOD_SPECS = {
    "splatad-radar-init": ("viking_splatad.splatad.radar_init_method", "RADAR_INIT_METHOD_SPEC"),
    "splatad-radar-lidar": ("viking_splatad.splatad.radar_lidar_method", "RADAR_LIDAR_METHOD_SPEC"),
    "splatad-radar-init-lidar": (
        "viking_splatad.splatad.radar_init_lidar_method",
        "RADAR_INIT_LIDAR_METHOD_SPEC",
    ),
}
# Contract: which methods use the seeds and which use the opacity loss.
SEED_METHODS = ("splatad-radar-init", "splatad-radar-init-lidar")
LOSS_METHODS = ("splatad-radar-lidar", "splatad-radar-init-lidar")
ALL_METHODS = ("splatad",) + tuple(RADAR_METHOD_SPECS)

SEED_FLAG_NAMES = {"--pipeline.model.radar-seed-opacity"}
LOSS_FLAG_NAMES = {
    "--pipeline.model.radar-lidar-loss-weight",
    "--pipeline.model.radar-lidar-gate-quantile",
    "--pipeline.model.radar-lidar-gate-chunk-size",
    "--pipeline.model.radar-lidar-auto-weight",
    "--pipeline.model.radar-lidar-diagnostics-path",
}

STAGE_ROOT = "/stage/root"
SESSION = "example_session"
RUN_DIR = f"{STAGE_ROOT}/runs/example_run"
RELATIVE_INPUTS_DIR = "radar_inputs/example_profile"
SEED_OPACITY = "0.37"

# Field separator for the harness output (never appears in flag values).
OUTPUT_SEPARATOR = "\x1f"


def expected_registration_string() -> str:
    """Return the comma-joined NERFSTUDIO_METHOD_CONFIGS value the contract prescribes."""
    return ",".join(
        f"{method_name}={module_name}:{spec_attribute}"
        for method_name, (module_name, spec_attribute) in RADAR_METHOD_SPECS.items()
    )


def extract_selection_block() -> str:
    """Return splatad_run.sbatch text from `RADAR_MODEL_ARGS=()` up to (excluding) `ns-train`.

    Returns:
        The bash source of the method-selection logic.
    """
    script_lines = RUN_SBATCH_PATH.read_text().splitlines()
    start_index = script_lines.index("RADAR_MODEL_ARGS=()")
    stop_index = next(
        line_index
        for line_index, line_text in enumerate(script_lines)
        if line_index > start_index and line_text.startswith("ns-train")
    )
    return "\n".join(script_lines[start_index:stop_index])


def run_selection(environment: Dict[str, str]) -> subprocess.CompletedProcess:
    """Run the selection block in bash with `environment` and print the resulting state.

    Args:
        environment: variables for the bash process (TRAIN_METHOD, RADAR_*, ...).

    Returns:
        The completed process; stdout holds MODEL/PARSER/CONFIGS records on success.
    """
    harness = "\n".join(
        [
            "set -euo pipefail",
            # The block appends to JOB_INFO; send it nowhere.
            "JOB_INFO=/dev/null",
            "TRAIN_LIDAR_ONLY=False",
            extract_selection_block(),
            f'for item in ${{RADAR_MODEL_ARGS[@]+"${{RADAR_MODEL_ARGS[@]}}"}}; do '
            f'printf "MODEL{OUTPUT_SEPARATOR}%s\\n" "$item"; done',
            f'for item in ${{RADAR_DATAPARSER_ARGS[@]+"${{RADAR_DATAPARSER_ARGS[@]}}"}}; do '
            f'printf "PARSER{OUTPUT_SEPARATOR}%s\\n" "$item"; done',
            f'printf "CONFIGS{OUTPUT_SEPARATOR}%s\\n" "${{NERFSTUDIO_METHOD_CONFIGS-__unset__}}"',
        ]
    )
    base_environment = {
        "PATH": "/usr/bin:/bin",
        "STAGE_ROOT": STAGE_ROOT,
        "SESSION": SESSION,
        "RUN_DIR": RUN_DIR,
        # Read by the `export VIKING_DETERMINISTIC=1 PYTHONHASHSEED=...` line inside the block.
        "TRAIN_SEED": "7",
        "RADAR_LIDAR_LOSS_WEIGHT": "0.5",
        "RADAR_LIDAR_GATE_QUANTILE": "0.9",
        "RADAR_LIDAR_GATE_CHUNK_SIZE": "128",
    }
    base_environment.update(environment)
    return subprocess.run(
        ["bash", "-c", harness], env=base_environment, capture_output=True, text=True, check=False
    )


def parse_records(completed: subprocess.CompletedProcess) -> Dict[str, List[str]]:
    """Group the harness stdout by record kind (MODEL, PARSER, CONFIGS)."""
    assert completed.returncode == 0, completed.stderr
    grouped: Dict[str, List[str]] = {"MODEL": [], "PARSER": [], "CONFIGS": []}
    for output_line in completed.stdout.splitlines():
        record_kind, _, record_value = output_line.partition(OUTPUT_SEPARATOR)
        grouped[record_kind].append(record_value)
    return grouped


def method_environment(method_name: str, inputs_dir: Optional[str] = RELATIVE_INPUTS_DIR) -> Dict[str, str]:
    """Return the job environment for `method_name`, with seed variables only for seed methods."""
    environment = {"TRAIN_METHOD": method_name}
    if method_name != "splatad" and inputs_dir is not None:
        environment["RADAR_INPUTS_DIR"] = inputs_dir
    if method_name in SEED_METHODS:
        environment["RADAR_SEED_OPACITY"] = SEED_OPACITY
    return environment


def flag_names(argument_list: List[str]) -> set:
    """Return the `--flag` tokens of an argument list."""
    return {argument for argument in argument_list if argument.startswith("--")}


@pytest.mark.parametrize("method_name", ALL_METHODS)
def test_model_flags_follow_the_seed_and_loss_contract(method_name):
    """RADAR_MODEL_ARGS holds the seed flag for seed methods and the five loss flags for loss methods."""
    records = parse_records(run_selection(method_environment(method_name)))
    present_flags = flag_names(records["MODEL"])
    expected_flags = set()
    if method_name in SEED_METHODS:
        expected_flags |= SEED_FLAG_NAMES
    if method_name in LOSS_METHODS:
        expected_flags |= LOSS_FLAG_NAMES
    assert present_flags == expected_flags
    if method_name in SEED_METHODS:
        # The seed opacity value follows its flag.
        flag_position = records["MODEL"].index("--pipeline.model.radar-seed-opacity")
        assert records["MODEL"][flag_position + 1] == SEED_OPACITY


@pytest.mark.parametrize("method_name", ALL_METHODS)
def test_dataparser_flags_and_registration_follow_the_contract(method_name):
    """Radar methods pass --radar-inputs-dir and register all three specs; stock does neither."""
    records = parse_records(run_selection(method_environment(method_name)))
    if method_name == "splatad":
        assert records["PARSER"] == []
        assert records["CONFIGS"] == [""]
    else:
        assert records["PARSER"] == [
            "--radar-inputs-dir",
            f"{STAGE_ROOT}/data/{SESSION}/{RELATIVE_INPUTS_DIR}",
        ]
        assert records["CONFIGS"] == [expected_registration_string()]


@pytest.mark.parametrize("method_name", SEED_METHODS)
def test_absolute_inputs_dir_is_passed_unchanged(method_name):
    """An absolute RADAR_INPUTS_DIR is not re-rooted under the session directory."""
    absolute_inputs_dir = "/elsewhere/radar_inputs/absolute_profile"
    records = parse_records(run_selection(method_environment(method_name, absolute_inputs_dir)))
    assert records["PARSER"] == ["--radar-inputs-dir", absolute_inputs_dir]


def test_unknown_method_exits_non_zero():
    """A TRAIN_METHOD outside the four known methods stops the job."""
    completed = run_selection({"TRAIN_METHOD": "splatad-radar-unknown"})
    assert completed.returncode != 0


@pytest.mark.parametrize("method_name", SEED_METHODS)
def test_seed_method_without_seed_opacity_follows_stock_initial_opacity(method_name):
    """Without RADAR_SEED_OPACITY a seed method runs and passes no seed-opacity flag.

    The model then uses stock SplatAD's own init_opacities for the seeds (user's decision
    2026-11-14); RADAR_SEED_OPACITY is only an optional override (its presence is covered by
    test_model_flags_follow_the_seed_and_loss_contract).
    """
    environment = method_environment(method_name)
    del environment["RADAR_SEED_OPACITY"]
    completed = run_selection(environment)
    assert completed.returncode == 0
    assert not flag_names(parse_records(completed)["MODEL"]) & SEED_FLAG_NAMES


@pytest.mark.parametrize("method_name", SEED_METHODS)
def test_seed_method_without_inputs_dir_exits_non_zero(method_name):
    """Seed methods require RADAR_INPUTS_DIR."""
    assert run_selection(method_environment(method_name, inputs_dir=None)).returncode != 0


def test_run_script_has_exactly_one_ns_train_command():
    """One ns-train line serves every method (the method only changes its suffix and arguments)."""
    ns_train_lines = [
        line_text for line_text in RUN_SBATCH_PATH.read_text().splitlines() if line_text.startswith("ns-train")
    ]
    assert len(ns_train_lines) == 1


def test_registered_specs_point_at_existing_module_files():
    """Every spec in the registration string names a module file under src/viking_splatad/splatad/."""
    registration = parse_records(run_selection(method_environment("splatad-radar-init-lidar")))["CONFIGS"][0]
    entries = registration.split(",")
    assert len(entries) == len(RADAR_METHOD_SPECS)
    for entry in entries:
        method_name, _, module_and_attribute = entry.partition("=")
        module_name, _, spec_attribute = module_and_attribute.partition(":")
        assert method_name and spec_attribute, entry
        module_prefix = "viking_splatad.splatad."
        assert module_name.startswith(module_prefix), entry
        module_path = SPLATAD_PACKAGE_DIR / (module_name[len(module_prefix):] + ".py")
        assert module_path.is_file(), f"{entry}: missing {module_path}"


def registration_values_in(text: str) -> List[str]:
    """Return every NERFSTUDIO_METHOD_CONFIGS= assignment value in `text` (quotes stripped)."""
    return [
        quoted or bare
        for quoted, bare in re.findall(r'NERFSTUDIO_METHOD_CONFIGS=(?:"([^"]*)"|(\S+))', text)
    ]


def test_render_only_script_carries_the_run_script_registration():
    """render_only.sbatch exports the same comma-joined value as splatad_run.sbatch."""
    run_value = parse_records(run_selection(method_environment("splatad-radar-init-lidar")))["CONFIGS"][0]
    assert run_value in registration_values_in(RENDER_SBATCH_PATH.read_text())


def test_install_script_prints_the_run_script_registration():
    """install_into_neurad.sh prints the same comma-joined value as splatad_run.sbatch."""
    run_value = parse_records(run_selection(method_environment("splatad-radar-init-lidar")))["CONFIGS"][0]
    assert run_value in registration_values_in(INSTALL_SCRIPT_PATH.read_text())

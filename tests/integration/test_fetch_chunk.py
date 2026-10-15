"""Tests of the selective multi-volume 7z fetcher (library + scripts/00_fetch_chunk.py).

A synthetic multi-volume archive (one LZMA2 folder per chunk, like the real Zenodo archive) is
served by a local HTTP server implementing Range, so no real dataset bytes are ever downloaded.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import py7zr
import pytest
import requests
import yaml

from viking_splatad.io.zenodo_archive import (
    ANONYMOUS_USER_AGENT,
    SEVENZIP_MAGIC,
    SEVENZIP_START_HEADER_BYTES,
    ArchiveExtractionError,
    VolumeInfo,
    ensure_archive_header,
    ensure_sparse_file,
    extract_entry,
    fetch_virtual_range,
    folder_packed_range,
    list_archive_entries,
    list_volumes,
    locate_volume,
    split_virtual_range,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FETCH_SCRIPT = PROJECT_ROOT / "scripts" / "00_fetch_chunk.py"
MAY_CONFIG = PROJECT_ROOT / "configs" / "sessions" / "may_short_grass.yaml"

# Synthetic archive layout (mirrors the real one: folder 0 = metadata + calibration + labels + chunk 0).
RECORD_ID = "424242"
ARCHIVE_STEM = "ros2_jazzy.7z"
BAG_NAME = "short_grass__ros2"
ARCHIVE_BAG_DIRECTORY = f"ros2_jazzy/data/2024_05_short_grass_run/bag/{BAG_NAME}"
CHUNK_COUNT = 4
# Incompressible random chunk payload size: large enough that folders span volume boundaries.
CHUNK_PAYLOAD_BYTES = 30_000
# Fixed seed so the synthetic payloads are reproducible.
PAYLOAD_SEED = 20240501
# Number of synthetic volumes, per the task (3 volumes of unequal size).
VOLUME_COUNT = 3


def chunk_name(chunk_index: int) -> str:
    """Archive entry name of a synthetic chunk."""
    return f"{ARCHIVE_BAG_DIRECTORY}/{BAG_NAME}_{chunk_index}.mcap"


@pytest.fixture(scope="module")
def synthetic_archive(tmp_path_factory) -> Dict[str, object]:
    """Build a one-folder-per-chunk 7z, split into unequal volumes cut inside folders 1 and 2.

    Returns:
        Dict with ``volumes_bytes`` (list of bytes), ``payloads`` (entry name -> bytes) and
        ``whole_bytes`` (the complete archive).
    """
    work_dir = tmp_path_factory.mktemp("synthetic_archive")
    random_generator = np.random.default_rng(PAYLOAD_SEED)
    payloads = {}
    # Step 1: payload files. Random bytes do not compress, so packed size ~ payload size.
    for chunk_index in range(CHUNK_COUNT):
        payloads[chunk_name(chunk_index)] = random_generator.bytes(CHUNK_PAYLOAD_BYTES + chunk_index * 1000)
    payloads[f"{ARCHIVE_BAG_DIRECTORY}/metadata.yaml"] = b"rosbag2_bagfile_information: {}\n"
    payloads["ros2_jazzy/calibration/extrinsics/extrinsics.txt"] = b"frames: demo\n" * 20
    payloads["ros2_jazzy/cuboid_labels/short_and_tall_grass_labels.json"] = b'{"labels": []}'
    for entry_name, payload in payloads.items():
        source_path = work_dir / "source" / entry_name
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(payload)
    archive_path = work_dir / "whole.7z"
    # Step 2: folder 0 = small files + chunk 0 in one session (solid block).
    folder_zero_names = [name for name in payloads if name != chunk_name(0) and "mcap" not in name]
    with py7zr.SevenZipFile(archive_path, "w") as archive:
        for entry_name in folder_zero_names + [chunk_name(0)]:
            archive.write(work_dir / "source" / entry_name, entry_name)
    # Step 3: every later chunk is appended in its own session => its own folder.
    for chunk_index in range(1, CHUNK_COUNT):
        with py7zr.SevenZipFile(archive_path, "a") as archive:
            archive.write(work_dir / "source" / chunk_name(chunk_index), chunk_name(chunk_index))
    whole_bytes = archive_path.read_bytes()
    # Step 4: cut points at the middle of folder 1 and folder 2 so those folders straddle boundaries.
    folder_one_start, folder_one_end, _ = folder_packed_range(archive_path, f"_{1}.mcap")
    folder_two_start, folder_two_end, _ = folder_packed_range(archive_path, f"_{2}.mcap")
    cut_points = [(folder_one_start + folder_one_end) // 2, (folder_two_start + folder_two_end) // 2]
    volumes_bytes = [whole_bytes[:cut_points[0]], whole_bytes[cut_points[0]:cut_points[1]],
                     whole_bytes[cut_points[1]:]]
    assert len(volumes_bytes) == VOLUME_COUNT and len({len(volume) for volume in volumes_bytes}) == VOLUME_COUNT
    return {"volumes_bytes": volumes_bytes, "payloads": payloads, "whole_bytes": whole_bytes}


class RangeServer:
    """Local HTTP server mimicking Zenodo: records API JSON plus Range-capable volume downloads."""

    def __init__(self, volumes_bytes: List[bytes], honour_range: bool = True):
        """Start serving on an ephemeral localhost port.

        Args:
            volumes_bytes: Content of each volume in order.
            honour_range: When False, ignore Range and answer 200 (to test 206 verification).
        """
        self.volume_content = {f"{ARCHIVE_STEM}.{number + 1:03d}": content
                               for number, content in enumerate(volumes_bytes)}
        self.honour_range = honour_range
        self.bytes_served = 0
        self.range_requests = 0
        self.user_agents: List[str] = []
        # (volume name, offset) of a byte to flip in served data, or None.
        self.corrupt_position: Optional[tuple] = None
        self.lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            """Handles the records API and volume downloads."""

            def log_message(self, *_args):
                """Silence default request logging."""

            def do_GET(self):  # noqa: N802 - http.server naming
                """Serve API JSON or a (partial) volume."""
                with outer.lock:
                    outer.user_agents.append(self.headers.get("User-Agent", ""))
                if self.path == f"/api/records/{RECORD_ID}":
                    body = json.dumps({"files": [{"key": name, "size": len(content)}
                                                 for name, content in outer.volume_content.items()]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                match = re.match(rf"/records/{RECORD_ID}/files/([^?]+)\?download=1$", self.path)
                if not match or match.group(1) not in outer.volume_content:
                    self.send_error(404)
                    return
                volume_name = match.group(1)
                content = outer.volume_content[volume_name]
                range_header = self.headers.get("Range")
                if not range_header or not outer.honour_range:
                    body = content
                    self.send_response(200)
                else:
                    first_text, last_text = re.match(r"bytes=(\d+)-(\d+)", range_header).groups()
                    first_byte, last_byte = int(first_text), int(last_text)
                    body = content[first_byte:last_byte + 1]
                    if outer.corrupt_position and outer.corrupt_position[0] == volume_name \
                            and first_byte <= outer.corrupt_position[1] <= last_byte:
                        corrupt_index = outer.corrupt_position[1] - first_byte
                        body = bytearray(body)
                        body[corrupt_index] ^= 0xFF  # flip all bits of one byte
                        body = bytes(body)
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {first_byte}-{last_byte}/{len(content)}")
                    with outer.lock:
                        outer.range_requests += 1
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                with outer.lock:
                    outer.bytes_served += len(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        """Stop the server."""
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server(synthetic_archive):
    """A fresh RangeServer per test (so byte counters start at zero)."""
    range_server = RangeServer(synthetic_archive["volumes_bytes"])
    yield range_server
    range_server.close()


def volumes_of(server: RangeServer) -> List[VolumeInfo]:
    """Volume list as the library would obtain it from the API."""
    return list_volumes(RECORD_ID, ARCHIVE_STEM, base_url=server.base_url)


def test_locate_volume_and_split_range_across_boundaries():
    """Virtual range math: a range crossing two boundaries yields three pieces summing to its length."""
    volume_sizes = [100, 50, 200]
    assert locate_volume(0, volume_sizes) == (0, 0)
    # Offset 100 is the first byte of volume 1; 149 the last of volume 1; 150 first of volume 2.
    assert locate_volume(100, volume_sizes) == (1, 0)
    assert locate_volume(149, volume_sizes) == (1, 49)
    assert locate_volume(150, volume_sizes) == (2, 0)
    with pytest.raises(ValueError):
        locate_volume(350, volume_sizes)
    pieces = split_virtual_range(volume_sizes, 90, 170)
    assert [(piece.volume_index, piece.offset_in_volume, piece.length) for piece in pieces] == [
        (0, 90, 10), (1, 0, 50), (2, 0, 20)]
    assert sum(piece.length for piece in pieces) == 170 - 90
    with pytest.raises(ValueError):
        split_virtual_range(volume_sizes, 10, 400)


def test_fixture_has_one_folder_per_chunk(synthetic_archive, tmp_path):
    """Sanity: chunks 1..N each own a folder; folder 0 shares small files with chunk 0."""
    archive_path = tmp_path / "whole.7z"
    archive_path.write_bytes(synthetic_archive["whole_bytes"])
    entries = {entry.name: entry for entry in list_archive_entries(archive_path)}
    chunk_folders = [entries[chunk_name(chunk_index)].folder_index for chunk_index in range(CHUNK_COUNT)]
    assert chunk_folders == list(range(CHUNK_COUNT))
    assert entries["ros2_jazzy/calibration/extrinsics/extrinsics.txt"].folder_index == 0


def test_list_volumes_uses_anonymous_user_agent(server):
    """API listing returns ordered volumes with sizes and sends only the anonymous User-Agent."""
    volumes = volumes_of(server)
    assert [volume.name for volume in volumes] == sorted(server.volume_content)
    assert [volume.size for volume in volumes] == [len(content) for content in server.volume_content.values()]
    assert set(server.user_agents) == {ANONYMOUS_USER_AGENT}


def test_folder_range_matches_header_and_fetch_extracts_identical_bytes(synthetic_archive, server, tmp_path):
    """Fetching only chunk 2's folder range extracts bytes identical to the original; chunk 3 stays locked."""
    volumes = volumes_of(server)
    sparse_path = tmp_path / "sparse.7z"
    ensure_archive_header(sparse_path, volumes, connections=2)
    header_bytes_served = server.bytes_served
    # Header must be tiny compared to the archive (only 32 B + end header + encoded header).
    assert header_bytes_served < len(synthetic_archive["whole_bytes"]) // 2
    start, end, folder_index = folder_packed_range(sparse_path, f"{BAG_NAME}_2.mcap")
    # Independent expectation from the complete archive header.
    with py7zr.SevenZipFile(py7zr_bytes(synthetic_archive, tmp_path)) as full_archive:
        packinfo = full_archive.header.main_streams.packinfo
        expected_start = SEVENZIP_START_HEADER_BYTES + packinfo.packpos + sum(packinfo.packsizes[:folder_index])
        expected_length = packinfo.packsizes[folder_index]
    assert (folder_index, start, end - start) == (2, expected_start, expected_length)
    # Folder 2 straddles a volume boundary by construction.
    assert len(split_virtual_range([len(v) for v in synthetic_archive["volumes_bytes"]], start, end)) == 2
    # Not yet fetched: extraction must fail instead of returning wrong data.
    with pytest.raises(ArchiveExtractionError):
        extract_entry(sparse_path, f"{BAG_NAME}_2.mcap", tmp_path / "out")
    fetched = fetch_virtual_range(volumes, start, end, sparse_path, connections=3)
    assert fetched == end - start
    assert server.bytes_served == header_bytes_served + (end - start)
    extract_entry(sparse_path, f"{BAG_NAME}_2.mcap", tmp_path / "out")
    extracted = (tmp_path / "out" / chunk_name(2)).read_bytes()
    assert hashlib.sha256(extracted).digest() == hashlib.sha256(synthetic_archive["payloads"][chunk_name(2)]).digest()
    # A different, unfetched chunk is not extractable and leaves no partial file.
    with pytest.raises(ArchiveExtractionError):
        extract_entry(sparse_path, f"{BAG_NAME}_3.mcap", tmp_path / "out")
    assert not (tmp_path / "out" / chunk_name(3)).exists()


def py7zr_bytes(synthetic_archive, tmp_path) -> Path:
    """Write the complete synthetic archive to disk (independent reference for header values)."""
    reference_path = tmp_path / "reference.7z"
    reference_path.write_bytes(synthetic_archive["whole_bytes"])
    return reference_path


def test_crc_mismatch_raises(synthetic_archive, tmp_path):
    """A flipped byte inside the served packed stream surfaces as ArchiveExtractionError."""
    volumes_bytes = synthetic_archive["volumes_bytes"]
    corrupting_server = RangeServer(volumes_bytes)
    try:
        volumes = volumes_of(corrupting_server)
        sparse_path = tmp_path / "sparse.7z"
        ensure_archive_header(sparse_path, volumes)
        start, end, _ = folder_packed_range(sparse_path, f"{BAG_NAME}_1.mcap")
        # Corrupt the byte in the middle of the folder's packed stream (random data => stored/LZMA2 chunk).
        corrupt_virtual_offset = (start + end) // 2
        volume_index, offset_in_volume = locate_volume(corrupt_virtual_offset, [len(v) for v in volumes_bytes])
        corrupting_server.corrupt_position = (volumes[volume_index].name, offset_in_volume)
        fetch_virtual_range(volumes, start, end, sparse_path, connections=2)
        with pytest.raises(ArchiveExtractionError):
            extract_entry(sparse_path, f"{BAG_NAME}_1.mcap", tmp_path / "out")
        assert not (tmp_path / "out" / chunk_name(1)).exists()
    finally:
        corrupting_server.close()


def test_non_206_response_is_rejected(synthetic_archive, tmp_path):
    """A server that ignores Range (200) must raise rather than write a whole volume."""
    ignoring_server = RangeServer(synthetic_archive["volumes_bytes"], honour_range=False)
    try:
        volumes = volumes_of(ignoring_server)
        sparse_path = tmp_path / "sparse.7z"
        with pytest.raises(RuntimeError, match="Range"):
            ensure_archive_header(sparse_path, volumes)
    finally:
        ignoring_server.close()


def write_session_config(tmp_path: Path) -> Path:
    """Copy the May config with raw paths pointing into tmp_path and synthetic record overrides.

    Args:
        tmp_path: Test directory.

    Returns:
        Path of the written YAML.
    """
    config = yaml.safe_load(MAY_CONFIG.read_text())
    session_root = tmp_path / "raw" / "synthetic_session" / "extracted" / "ros2_jazzy"
    config["session"]["bag_dir"] = str(session_root / "data" / "2024_05_short_grass_run" / "bag" / BAG_NAME)
    config["session"]["calibration_dir"] = str(session_root / "calibration")
    config["session"]["labels_json"] = str(session_root / "cuboid_labels" / "short_and_tall_grass_labels.json")
    config["session"]["zenodo_record"] = RECORD_ID
    config["session"]["archive_stem"] = ARCHIVE_STEM
    config_path = tmp_path / "session.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def run_cli(server: RangeServer, config_path: Path, *extra_arguments: str) -> subprocess.CompletedProcess:
    """Run scripts/00_fetch_chunk.py against the local server.

    Args:
        server: Local server whose base URL is injected through the environment.
        config_path: Session YAML.
        extra_arguments: Additional CLI arguments.
    """
    environment = dict(os.environ, VIKING_ZENODO_BASE_URL=server.base_url)
    return subprocess.run([sys.executable, str(FETCH_SCRIPT), "--config", str(config_path), *extra_arguments],
                          capture_output=True, text=True, env=environment, check=False)


def test_cli_dry_run_end_to_end_skip_and_force(synthetic_archive, server, tmp_path):
    """CLI: dry-run fetches no dataset bytes; real run extracts; re-run skips; --force refetches."""
    config_path = write_session_config(tmp_path)
    extracted_chunk = (tmp_path / "raw" / "synthetic_session" / "extracted" / chunk_name(1))
    # Dry run: only header requests (small), nothing extracted.
    dry_run = run_cli(server, config_path, "--chunk", "1", "--dry-run")
    assert dry_run.returncode == 0, dry_run.stderr
    assert "folder 1" in dry_run.stdout and "pieces:" in dry_run.stdout
    assert not extracted_chunk.exists()
    header_bytes = server.bytes_served
    assert header_bytes < len(synthetic_archive["whole_bytes"]) // 2
    # Real run: fetch exactly chunk 1's folder, extract identical bytes.
    real_run = run_cli(server, config_path, "--chunk", "1", "--connections", "2")
    assert real_run.returncode == 0, real_run.stderr
    sparse_path = tmp_path / "raw" / "synthetic_session" / "ros2_jazzy.sparse.7z"
    start, end, _ = folder_packed_range(sparse_path, f"{BAG_NAME}_1.mcap")
    assert server.bytes_served == header_bytes + (end - start)
    assert extracted_chunk.read_bytes() == synthetic_archive["payloads"][chunk_name(1)]
    assert "fetched" in real_run.stdout
    # Re-run: skipped, no new bytes.
    served_before_rerun = server.bytes_served
    rerun = run_cli(server, config_path, "--chunk", "1")
    assert rerun.returncode == 0 and "skipped" in rerun.stdout
    assert server.bytes_served == served_before_rerun
    # Force: fetches again.
    forced = run_cli(server, config_path, "--chunk", "1", "--force")
    assert forced.returncode == 0 and server.bytes_served == served_before_rerun + (end - start)


def test_cli_fetch_calibration_and_multiple_chunks(synthetic_archive, server, tmp_path):
    """CLI: --fetch-calibration extracts folder-0 small files; --chunks handles several chunks; --all lists all."""
    config_path = write_session_config(tmp_path)
    extracted_root = tmp_path / "raw" / "synthetic_session" / "extracted"
    result = run_cli(server, config_path, "--fetch-calibration", "--chunks", "2", "3")
    assert result.returncode == 0, result.stderr
    for small_name in ("ros2_jazzy/calibration/extrinsics/extrinsics.txt",
                       "ros2_jazzy/cuboid_labels/short_and_tall_grass_labels.json",
                       f"{ARCHIVE_BAG_DIRECTORY}/metadata.yaml"):
        assert (extracted_root / small_name).read_bytes() == synthetic_archive["payloads"][small_name]
    for chunk_index in (2, 3):
        assert (extracted_root / chunk_name(chunk_index)).read_bytes() == \
            synthetic_archive["payloads"][chunk_name(chunk_index)]
    # Unknown chunk index fails cleanly.
    missing = run_cli(server, config_path, "--chunk", str(CHUNK_COUNT + 5), "--dry-run")
    assert missing.returncode != 0 and "not in archive" in missing.stderr
    # --all --dry-run mentions every chunk.
    everything = run_cli(server, config_path, "--all", "--dry-run")
    assert everything.returncode == 0
    for chunk_index in range(CHUNK_COUNT):
        assert f"chunk {chunk_index}:" in everything.stdout


def test_cli_corrupted_chunk_fails_with_nonzero_exit(synthetic_archive, tmp_path):
    """CLI: corrupted served bytes make the script exit non-zero and leave no extracted chunk."""
    volumes_bytes = synthetic_archive["volumes_bytes"]
    corrupting_server = RangeServer(volumes_bytes)
    try:
        config_path = write_session_config(tmp_path)
        volumes = volumes_of(corrupting_server)
        sparse_path = tmp_path / "raw" / "synthetic_session" / "ros2_jazzy.sparse.7z"
        ensure_archive_header(sparse_path, volumes)
        start, end, _ = folder_packed_range(sparse_path, f"{BAG_NAME}_1.mcap")
        volume_index, offset_in_volume = locate_volume((start + end) // 2, [len(v) for v in volumes_bytes])
        corrupting_server.corrupt_position = (volumes[volume_index].name, offset_in_volume)
        result = run_cli(corrupting_server, config_path, "--chunk", "1")
        assert result.returncode != 0 and "FAILED" in result.stderr
        assert not (tmp_path / "raw" / "synthetic_session" / "extracted" / chunk_name(1)).exists()
    finally:
        corrupting_server.close()


@pytest.mark.real_data
def test_real_zenodo_listing_and_header():
    """Real Zenodo: 15 volumes; chunk 1's folder range equals its pack size (JSON + 32 B header only)."""
    try:
        volumes = list_volumes("20795213", "ros2_jazzy.7z", base_url="https://zenodo.org")
    except requests.RequestException:
        pytest.skip("Zenodo not reachable (offline)")
    assert len(volumes) == 15
    # Fetch just the 32-byte start header into a temporary sparse file and check the 7z signature.
    scratch_path = Path(tempfile.mkdtemp()) / "header_probe.7z"
    ensure_sparse_file(scratch_path, sum(volume.size for volume in volumes))
    assert fetch_virtual_range(volumes, 0, SEVENZIP_START_HEADER_BYTES, scratch_path, connections=1) \
        == SEVENZIP_START_HEADER_BYTES
    with open(scratch_path, "rb") as scratch_handle:
        assert scratch_handle.read(len(SEVENZIP_MAGIC)) == SEVENZIP_MAGIC
    scratch_path.unlink()
    # The remaining header regions come from the sparse archive of an earlier run, when present
    # (otherwise the test stops after the listing/signature checks).
    session_sparse = PROJECT_ROOT / "data" / "raw" / "may_short_grass" / "ros2_jazzy.sparse.7z"
    if not session_sparse.is_file():
        pytest.skip("no local sparse archive with header available")
    start, end, folder_index = folder_packed_range(session_sparse, "short_grass__ros2_1.mcap")
    with py7zr.SevenZipFile(session_sparse) as archive:
        packsize = archive.header.main_streams.packinfo.packsizes[folder_index]
    assert end - start == packsize
    assert folder_index >= 1

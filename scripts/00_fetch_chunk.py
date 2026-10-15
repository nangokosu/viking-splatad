"""Fetch and extract single bag chunks of the multi-volume Zenodo archive (no full download).

The archive holds one independent LZMA2 folder per bag chunk, so only that folder's packed bytes
(~1.6 GB per chunk) are downloaded with parallel HTTP Range requests into a sparse local copy of
the archive; py7zr then extracts just that chunk and verifies its CRC32.

Usage:
    uv run python scripts/00_fetch_chunk.py --config configs/sessions/may_short_grass.yaml \
        (--chunk N | --chunks N M ... | --all) [--fetch-calibration] [--dry-run] \
        [--connections K] [--force]

Optional session YAML keys (read with .get, so the strict loader is unaffected):
    session.zenodo_record   Zenodo record id (default below)
    session.archive_stem    archive name without volume number (default below)
Environment override (tests): VIKING_ZENODO_BASE_URL points API + downloads at another origin.

Not implemented: freeing already-extracted bytes from the sparse file (hole punching). A plain
truncate would invalidate other chunks' bytes, and portable hole punching is not available in the
Python standard library on macOS. Delete the sparse file when all wanted chunks are extracted.

Only anonymous requests are made (User-Agent viking-splatad/0.1).
"""

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from viking_splatad.config import load_session_config
from viking_splatad.io.zenodo_archive import (
    ArchiveEntry,
    ArchiveExtractionError,
    VolumeInfo,
    ensure_archive_header,
    extract_matching,
    fetch_virtual_range,
    folder_packed_range_by_index,
    list_archive_entries,
    list_volumes,
    split_virtual_range,
)

# Zenodo record of the Viking Hill dataset (DOI 10.5281/zenodo.20795213); overridable per session YAML.
DEFAULT_ZENODO_RECORD = "20795213"
# Archive file name before the volume number (.001, .002, ...); overridable per session YAML.
DEFAULT_ARCHIVE_STEM = "ros2_jazzy.7z"
# Default parallel connections: one connection is throttled well below the link speed (measured
# earlier with the head fetch, which used 8); kept modest to stay polite to Zenodo.
DEFAULT_CONNECTIONS = 8
# Directory name that marks the extraction root inside data/raw/<session>/.
EXTRACT_ROOT_DIRECTORY_NAME = "extracted"
# Archive sub-directories fetched by --fetch-calibration (they share folder 0 with chunk 0).
CALIBRATION_DIRECTORY_NAMES = ("calibration", "cuboid_labels")
# Name of the bag metadata file extracted by --fetch-calibration.
BAG_METADATA_FILE_NAME = "metadata.yaml"


def parse_arguments(argument_list: Optional[List[str]]) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argument_list: Arguments (default: sys.argv[1:]).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="session YAML")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--chunk", type=int, help="single chunk index N")
    selection.add_argument("--chunks", type=int, nargs="+", help="several chunk indices")
    selection.add_argument("--all", action="store_true", help="every chunk of this session's bag")
    parser.add_argument("--fetch-calibration", action="store_true",
                        help="also extract calibration/, cuboid_labels/ and the bag metadata.yaml (folder 0)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, fetch no dataset bytes")
    parser.add_argument("--connections", type=int, default=DEFAULT_CONNECTIONS, help="parallel connections")
    parser.add_argument("--force", action="store_true", help="re-fetch and re-extract existing files")
    return parser.parse_args(argument_list)


def resolve_session_settings(config: dict) -> Dict[str, object]:
    """Derive record, stem, bag name and local paths from the session config.

    Args:
        config: Output of load_session_config (paths already absolute).

    Returns:
        Dict with record, stem, bag_dir, bag_name, extract_root, sparse_path.

    Raises:
        ValueError: If bag_dir has no ``extracted`` ancestor.
    """
    session = config["session"]
    bag_dir = Path(session["bag_dir"])
    # Step 1: extraction root = the 'extracted' ancestor of bag_dir (archive paths are rooted there).
    ancestor_names = [parent.name for parent in bag_dir.parents]
    if EXTRACT_ROOT_DIRECTORY_NAME not in ancestor_names:
        raise ValueError(f"bag_dir {bag_dir} has no '{EXTRACT_ROOT_DIRECTORY_NAME}' ancestor")
    extract_root = list(bag_dir.parents)[ancestor_names.index(EXTRACT_ROOT_DIRECTORY_NAME)]
    # Step 2: optional overrides, else documented defaults.
    archive_stem = session.get("archive_stem", DEFAULT_ARCHIVE_STEM)
    # Step 3: sparse copy sits next to the extraction root, named after the stem without '.7z'.
    sparse_name = archive_stem[:-len(".7z")] + ".sparse.7z" if archive_stem.endswith(".7z") else archive_stem + ".sparse"
    return {
        "record": str(session.get("zenodo_record", DEFAULT_ZENODO_RECORD)),
        "stem": archive_stem,
        "bag_dir": bag_dir,
        "bag_name": bag_dir.name,
        "extract_root": extract_root,
        "sparse_path": extract_root.parent / sparse_name,
    }


def find_chunk_entries(entries: List[ArchiveEntry], bag_name: str) -> Dict[int, ArchiveEntry]:
    """Map chunk index to its archive entry (``<bag_name>_<N>.mcap``).

    Args:
        entries: All archive entries.
        bag_name: Bag directory name, e.g. ``short_grass__ros2``.

    Returns:
        Dict chunk index -> entry, sorted by index.
    """
    chunk_pattern = re.compile(rf"/{re.escape(bag_name)}/{re.escape(bag_name)}_(\d+)\.mcap$")
    chunk_entries = {}
    for entry in entries:
        # Prefix '/' so top-level names match the same pattern as nested ones.
        match = chunk_pattern.search("/" + entry.name)
        if match:
            chunk_entries[int(match.group(1))] = entry
    return dict(sorted(chunk_entries.items()))


def describe_plan(volumes: List[VolumeInfo], range_start: int, range_end: int, folder_index: int) -> str:
    """Format a dry-run description of a folder's byte range and its volume pieces.

    Args:
        volumes: Archive volumes.
        range_start: Virtual start (inclusive).
        range_end: Virtual end (exclusive).
        folder_index: Folder number.

    Returns:
        Multi-line text.
    """
    pieces = split_virtual_range([volume.size for volume in volumes], range_start, range_end)
    piece_text = ", ".join(
        f"{volumes[piece.volume_index].name}[{piece.offset_in_volume}+{piece.length}]" for piece in pieces)
    return (f"folder {folder_index}: virtual bytes [{range_start}, {range_end}) "
            f"= {range_end - range_start} bytes; pieces: {piece_text}")


def fetch_and_extract(sparse_path: Path, volumes: List[VolumeInfo], entries_to_extract: List[ArchiveEntry],
                      extract_root: Path, connections: int) -> int:
    """Fetch the packed bytes of the entries' folders and extract exactly those entries.

    Args:
        sparse_path: Sparse archive with header present.
        volumes: Archive volumes.
        entries_to_extract: Entries to extract (their folders are fetched).
        extract_root: Extraction root directory.
        connections: Parallel connections.

    Returns:
        Bytes fetched.
    """
    fetched_bytes = 0
    # Step 1: fetch each distinct folder once (folder 0 serves several entries).
    for folder_index in sorted({entry.folder_index for entry in entries_to_extract}):
        range_start, range_end, _ = folder_packed_range_by_index(sparse_path, folder_index)
        fetched_bytes += fetch_virtual_range(volumes, range_start, range_end, sparse_path, connections)
    # Step 2: extract only the requested names (py7zr verifies CRC32).
    wanted_names = {entry.name for entry in entries_to_extract}
    extract_matching(sparse_path, lambda entry_name: entry_name in wanted_names, extract_root)
    return fetched_bytes


def is_up_to_date(entry: ArchiveEntry, extract_root: Path) -> bool:
    """Return True when the extracted file exists with the header's size.

    Args:
        entry: Archive entry.
        extract_root: Extraction root.
    """
    target_path = extract_root / entry.name
    return target_path.is_file() and target_path.stat().st_size == entry.size


def process_entries(label: str, entries_to_extract: List[ArchiveEntry], settings: Dict[str, object],
                    volumes: List[VolumeInfo], arguments: argparse.Namespace) -> bool:
    """Handle one unit of work (a chunk or the calibration set): skip, dry-run or fetch+extract.

    Args:
        label: Text for the summary line.
        entries_to_extract: Entries making up this unit.
        settings: Output of resolve_session_settings.
        volumes: Archive volumes.
        arguments: Parsed CLI arguments.

    Returns:
        True on success (including skip/dry-run), False on an extraction error.
    """
    extract_root = settings["extract_root"]
    sparse_path = settings["sparse_path"]
    extracted_bytes = sum(entry.size for entry in entries_to_extract)
    # Step 1: skip when everything is already extracted at the right size.
    if not arguments.force and all(is_up_to_date(entry, extract_root) for entry in entries_to_extract):
        print(f"{label}: skipped (already extracted, {extracted_bytes} bytes)")
        return True
    # Step 2: dry run prints the plan only.
    if arguments.dry_run:
        for folder_index in sorted({entry.folder_index for entry in entries_to_extract}):
            range_start, range_end, _ = folder_packed_range_by_index(sparse_path, folder_index)
            print(f"{label}: {describe_plan(volumes, range_start, range_end, folder_index)}; "
                  f"extracts {extracted_bytes} bytes")
        return True
    # Step 3: real work with timing.
    start_time = time.monotonic()
    try:
        fetched_bytes = fetch_and_extract(sparse_path, volumes, entries_to_extract, extract_root,
                                          arguments.connections)
    except ArchiveExtractionError as error:
        print(f"{label}: FAILED {error}", file=sys.stderr)
        return False
    print(f"{label}: fetched {fetched_bytes} bytes in {time.monotonic() - start_time:.1f}s, "
          f"extracted {extracted_bytes} bytes")
    return True


def main(argument_list: Optional[List[str]] = None) -> int:
    """Run the fetch/extract workflow.

    Args:
        argument_list: Command-line arguments (default: sys.argv[1:]).

    Returns:
        0 on success, 1 on any failure.
    """
    arguments = parse_arguments(argument_list)
    if not (arguments.chunk is not None or arguments.chunks or arguments.all or arguments.fetch_calibration):
        print("Nothing to do: pass --chunk, --chunks, --all and/or --fetch-calibration", file=sys.stderr)
        return 1
    settings = resolve_session_settings(load_session_config(arguments.config))
    # Step 1: volumes from the Zenodo API (JSON only) and header regions of the sparse archive.
    volumes = list_volumes(settings["record"], settings["stem"])
    ensure_archive_header(settings["sparse_path"], volumes, arguments.connections)
    entries = list_archive_entries(settings["sparse_path"])
    chunk_entries = find_chunk_entries(entries, settings["bag_name"])
    # Step 2: resolve requested chunk indices.
    if arguments.all:
        requested_indices = list(chunk_entries)
    elif arguments.chunks:
        requested_indices = list(arguments.chunks)
    elif arguments.chunk is not None:
        requested_indices = [arguments.chunk]
    else:
        requested_indices = []
    all_succeeded = True
    # Step 3: calibration / labels / metadata (folder 0), when requested.
    if arguments.fetch_calibration:
        bag_prefix = "/" + settings["bag_name"] + "/"
        calibration_entries = [
            entry for entry in entries
            if any(f"/{directory}/" in "/" + entry.name for directory in CALIBRATION_DIRECTORY_NAMES)
            or ("/" + entry.name).endswith(bag_prefix + BAG_METADATA_FILE_NAME)]
        if not calibration_entries:
            print("No calibration entries found in archive", file=sys.stderr)
            all_succeeded = False
        else:
            all_succeeded &= process_entries("calibration", calibration_entries, settings, volumes, arguments)
    # Step 4: each requested chunk.
    for chunk_index in requested_indices:
        if chunk_index not in chunk_entries:
            print(f"chunk {chunk_index}: not in archive (available: {sorted(chunk_entries)})", file=sys.stderr)
            all_succeeded = False
            continue
        chunk_entry = chunk_entries[chunk_index]
        # The extracted file must land exactly where the config's bag_dir expects it.
        expected_path = settings["bag_dir"] / Path(chunk_entry.name).name
        if settings["extract_root"] / chunk_entry.name != expected_path:
            print(f"chunk {chunk_index}: archive path {chunk_entry.name} does not match bag_dir", file=sys.stderr)
            all_succeeded = False
            continue
        all_succeeded &= process_entries(f"chunk {chunk_index}", [chunk_entry], settings, volumes, arguments)
    return 0 if all_succeeded else 1


if __name__ == "__main__":
    sys.exit(main())

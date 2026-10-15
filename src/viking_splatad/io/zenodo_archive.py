"""Selective access to a multi-volume Zenodo 7z archive through HTTP range requests.

The Viking Hill archive (ros2_jazzy.7z.001 ... .015, ~160 GB) is one 7z whose volumes are a plain
byte-split of a single file (the "virtual concatenation"). It holds one independent LZMA2 folder
(solid block) per bag chunk, so a single chunk can be fetched and extracted alone:

  1. a local SPARSE file of the full virtual size is created (holes read as zeros);
  2. the 32-byte start header, the end header and the encoded-header stream are fetched, which
     lets py7zr open the sparse file and list every entry;
  3. the packed bytes of ONE folder are fetched into the sparse file at their virtual offsets;
  4. py7zr extracts only the entries of that folder (it verifies the CRC32 of every file).

Virtual layout of folder k (7z format): packed streams follow the 32-byte start header, so
    start_k = 32 + packpos + sum(packsizes[:first_stream_k]),   end_k = start_k + packed size.

All requests use the anonymous User-Agent below; no personal information is ever sent.
"""

import io
import lzma
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import py7zr
import requests
from py7zr.archiveinfo import HeaderStreamsInfo
from py7zr.exceptions import ArchiveError

# Generic anonymous identifier, per project rule: no personal info in any request.
ANONYMOUS_USER_AGENT = "viking-splatad/0.1"
# Environment variable overriding the Zenodo origin (used by tests to target a local server).
BASE_URL_ENVIRONMENT_VARIABLE = "VIKING_ZENODO_BASE_URL"
# Production origin of Zenodo.
DEFAULT_ZENODO_BASE_URL = "https://zenodo.org"
# Size of the fixed 7z signature header: 6 signature + 2 version + 4 CRC + 20 start-header bytes.
SEVENZIP_START_HEADER_BYTES = 32
# First six bytes of every 7z file (format specification).
SEVENZIP_MAGIC = b"7z\xbc\xaf\x27\x1c"
# Bytes between the magic and the next-header pointer: 2 version + 4 start-header CRC.
SEVENZIP_VERSION_AND_CRC_BYTES = 6
# Property id marking an encoded (compressed) next header (7z format specification).
ENCODED_HEADER_PROPERTY_ID = 0x17
# Streaming block size for range responses: only a memory/throughput trade-off.
STREAM_CHUNK_BYTES = 8 * 1024 * 1024
# HTTP timeout per request in seconds: generous because Zenodo can stall for tens of seconds.
REQUEST_TIMEOUT_SECONDS = 60
# Attempts per range segment: transient network errors are common on multi-GB transfers.
FETCH_ATTEMPTS = 3
# HTTP status for a satisfied range request.
HTTP_PARTIAL_CONTENT = 206


class ArchiveExtractionError(RuntimeError):
    """Raised when entries cannot be extracted or fail their CRC/size verification."""


@dataclass(frozen=True)
class VolumeInfo:
    """One archive volume: file name, size in bytes and direct download URL."""

    name: str
    size: int
    url: str


@dataclass(frozen=True)
class VolumePiece:
    """A contiguous slice of one volume covering part of a virtual range."""

    volume_index: int
    offset_in_volume: int
    length: int
    virtual_offset: int


@dataclass(frozen=True)
class ArchiveEntry:
    """One non-directory archive entry: path inside the archive, size and holding folder."""

    name: str
    size: int
    folder_index: int


def resolve_base_url() -> str:
    """Return the Zenodo origin, honouring the test override environment variable.

    Returns:
        Origin URL without trailing slash.
    """
    # Environment override first, else production.
    return os.environ.get(BASE_URL_ENVIRONMENT_VARIABLE, DEFAULT_ZENODO_BASE_URL).rstrip("/")


def list_volumes(record_id: str, archive_stem: str, base_url: Optional[str] = None) -> List[VolumeInfo]:
    """List archive volumes of a Zenodo record from the records API (JSON only, tiny request).

    Args:
        record_id: Zenodo record id.
        archive_stem: Archive file name without the volume number, e.g. ``ros2_jazzy.7z``.
        base_url: Zenodo origin (default: environment/production origin).

    Returns:
        Volumes sorted by name (``.001``, ``.002``, ... sort lexicographically).

    Raises:
        RuntimeError: If the record has no file named ``<stem>.<number>``.
    """
    origin = (base_url or resolve_base_url()).rstrip("/")
    # Step 1: query the record metadata.
    response = requests.get(f"{origin}/api/records/{record_id}",
                            headers={"User-Agent": ANONYMOUS_USER_AGENT}, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    record_files = response.json()["files"]
    # Step 2: keep only volumes of this archive and sort by name to get volume order.
    prefix = archive_stem + "."
    matching_files = sorted((entry for entry in record_files if entry["key"].startswith(prefix)),
                            key=lambda entry: entry["key"])
    if not matching_files:
        raise RuntimeError(f"Record {record_id} has no files named {prefix}*")
    return [VolumeInfo(entry["key"], int(entry["size"]),
                       f"{origin}/records/{record_id}/files/{entry['key']}?download=1")
            for entry in matching_files]


def locate_volume(virtual_offset: int, volume_sizes: List[int]) -> Tuple[int, int]:
    """Map an offset in the virtual concatenation to (volume_index, offset_within_volume).

    Args:
        virtual_offset: Byte offset in the concatenation of all volumes.
        volume_sizes: Size in bytes of each volume, in order.

    Returns:
        Tuple (volume_index, offset_within_volume).

    Raises:
        ValueError: If the offset is negative or beyond the archive end.
    """
    if virtual_offset < 0:
        raise ValueError("Negative offset")
    remaining_offset = virtual_offset
    # Walk volumes, subtracting sizes until the offset falls inside one (O(volume count)).
    for volume_index, volume_size in enumerate(volume_sizes):
        if remaining_offset < volume_size:
            return volume_index, remaining_offset
        remaining_offset -= volume_size
    raise ValueError("Offset beyond end of archive")


def split_virtual_range(volume_sizes: List[int], virtual_start: int, virtual_end: int) -> List[VolumePiece]:
    """Split the virtual byte range [virtual_start, virtual_end) at volume boundaries.

    Args:
        volume_sizes: Size in bytes of each volume, in order.
        virtual_start: First virtual byte (inclusive).
        virtual_end: One past the last virtual byte (exclusive).

    Returns:
        Pieces in order; their lengths sum to ``virtual_end - virtual_start``.

    Raises:
        ValueError: If the range is empty/inverted or exceeds the archive.
    """
    if virtual_end <= virtual_start:
        raise ValueError("Empty or inverted range")
    if virtual_end > sum(volume_sizes):
        raise ValueError("Range beyond end of archive")
    pieces: List[VolumePiece] = []
    cursor = virtual_start
    # Each iteration consumes the rest of the current volume or the rest of the range,
    # whichever ends first, so a range crossing V boundaries yields V+1 pieces.
    while cursor < virtual_end:
        volume_index, offset_in_volume = locate_volume(cursor, volume_sizes)
        piece_length = min(virtual_end - cursor, volume_sizes[volume_index] - offset_in_volume)
        pieces.append(VolumePiece(volume_index, offset_in_volume, piece_length, cursor))
        cursor += piece_length
    return pieces


def fetch_range_into(download_url: str, first_byte: int, last_byte: int,
                     output_path: Path, write_offset: int) -> int:
    """Stream bytes [first_byte, last_byte] of a URL into a file at write_offset.

    Verifies the response is 206 (range honoured) and that exactly the requested number of
    bytes arrived; retries a few times on transient failures.

    Args:
        download_url: Direct URL honouring HTTP Range.
        first_byte: First remote byte (inclusive).
        last_byte: Last remote byte (inclusive).
        output_path: Existing file opened ``r+b`` (own handle, so threads do not share state).
        write_offset: Position in the output file where first_byte is stored.

    Returns:
        Number of bytes written (== last_byte - first_byte + 1).

    Raises:
        RuntimeError: If the range is not honoured or the byte count is wrong after all attempts.
    """
    expected_bytes = last_byte - first_byte + 1
    request_headers = {"Range": f"bytes={first_byte}-{last_byte}", "User-Agent": ANONYMOUS_USER_AGENT}
    last_error: Optional[Exception] = None
    for _attempt in range(FETCH_ATTEMPTS):
        try:
            bytes_written = 0
            # stream=True so a multi-GB range never sits fully in memory.
            with requests.get(download_url, headers=request_headers, stream=True,
                              timeout=REQUEST_TIMEOUT_SECONDS) as response:
                # A 200 would mean the server ignored Range and is sending the whole volume.
                if response.status_code != HTTP_PARTIAL_CONTENT:
                    raise RuntimeError(f"Range not honoured ({response.status_code}) for {download_url}")
                with open(output_path, "r+b") as output_handle:
                    output_handle.seek(write_offset)
                    for data_block in response.iter_content(STREAM_CHUNK_BYTES):
                        output_handle.write(data_block)
                        bytes_written += len(data_block)
            if bytes_written != expected_bytes:
                raise RuntimeError(f"Short range response: got {bytes_written} of {expected_bytes} bytes")
            return bytes_written
        except (requests.RequestException, RuntimeError) as error:
            last_error = error
    raise RuntimeError(f"Range fetch failed after {FETCH_ATTEMPTS} attempts: {last_error}")


def fetch_virtual_range(volumes: List[VolumeInfo], virtual_start: int, virtual_end: int,
                        sparse_path: Path, connections: int) -> int:
    """Fetch the virtual byte range [virtual_start, virtual_end) into a sparse archive copy.

    The range is split at volume boundaries, and each piece is further split into segments
    of about ``ceil(total / connections)`` bytes, so every connection gets comparable work
    (one connection is throttled well below the link speed). The thread pool bounds the
    number of simultaneous connections to ``connections``.

    Args:
        volumes: Archive volumes in order (name, size, url).
        virtual_start: First virtual byte (inclusive).
        virtual_end: One past the last virtual byte (exclusive).
        sparse_path: Existing sparse file of the full virtual size.
        connections: Maximum parallel connections (>= 1).

    Returns:
        Total bytes fetched.
    """
    if connections < 1:
        raise ValueError("connections must be >= 1")
    volume_sizes = [volume.size for volume in volumes]
    pieces = split_virtual_range(volume_sizes, virtual_start, virtual_end)
    total_bytes = virtual_end - virtual_start
    # Ceil division so segments of this size cover the range with at most
    # `connections` segments per piece (plus one leftover segment per extra piece).
    segment_bytes = -(-total_bytes // connections)
    segment_jobs: List[Tuple[str, int, int, int]] = []
    for piece in pieces:
        piece_cursor = 0
        while piece_cursor < piece.length:
            segment_length = min(segment_bytes, piece.length - piece_cursor)
            first_remote_byte = piece.offset_in_volume + piece_cursor
            segment_jobs.append((volumes[piece.volume_index].url, first_remote_byte,
                                 first_remote_byte + segment_length - 1,
                                 piece.virtual_offset + piece_cursor))
            piece_cursor += segment_length

    def run_segment(job: Tuple[str, int, int, int]) -> int:
        """Fetch one (url, first_byte, last_byte, write_offset) segment."""
        job_url, job_first, job_last, job_write_offset = job
        return fetch_range_into(job_url, job_first, job_last, sparse_path, job_write_offset)

    with ThreadPoolExecutor(max_workers=connections) as executor:
        return sum(executor.map(run_segment, segment_jobs))


def ensure_sparse_file(sparse_path: Path, total_size: int) -> None:
    """Create the sparse file (or resize it) to the full virtual archive size.

    Existing content is preserved so previously fetched chunks stay valid.

    Args:
        sparse_path: Sparse archive path.
        total_size: Sum of all volume sizes.
    """
    sparse_path.parent.mkdir(parents=True, exist_ok=True)
    # Step 1: create an empty file when absent ("ab" never truncates).
    with open(sparse_path, "ab"):
        pass
    # Step 2: set length; truncate() to a larger size creates a hole that reads as zeros.
    if sparse_path.stat().st_size != total_size:
        with open(sparse_path, "r+b") as sparse_handle:
            sparse_handle.truncate(total_size)


def archive_is_openable(archive_path: Path) -> bool:
    """Return True when py7zr can read the archive index (header regions present).

    Args:
        archive_path: Sparse or complete archive.
    """
    try:
        with py7zr.SevenZipFile(archive_path) as archive:
            archive.getnames()
        return True
    except Exception:  # noqa: BLE001 - any parse failure just means "header not fetched yet"
        return False


def ensure_archive_header(sparse_path: Path, volumes: List[VolumeInfo], connections: int = 1) -> int:
    """Fetch the header regions that let py7zr open the sparse archive.

    The start header (32 B) gives the next-header offset/size; the next header is normally an
    EncodedHeader (property 0x17) whose pack info points at the compressed real header stream.
    That location is computed from the pointer, so no blind margin is needed:
        packed_header_start = 32 + packpos,  packed_header_size = sum(packsizes).

    Args:
        sparse_path: Sparse archive (created at full size when missing).
        volumes: Archive volumes in order.
        connections: Parallel connections for each fetch (headers are small; 1 is fine).

    Returns:
        Bytes fetched (0 when the archive already opens).
    """
    ensure_sparse_file(sparse_path, sum(volume.size for volume in volumes))
    # Step 1: nothing to do when the index is already readable (re-run / reuse).
    if archive_is_openable(sparse_path):
        return 0
    fetched_bytes = fetch_virtual_range(volumes, 0, SEVENZIP_START_HEADER_BYTES, sparse_path, connections)
    with open(sparse_path, "rb") as sparse_handle:
        start_header = sparse_handle.read(SEVENZIP_START_HEADER_BYTES)
    # Step 2: validate the signature and decode the next-header pointer.
    if start_header[:len(SEVENZIP_MAGIC)] != SEVENZIP_MAGIC:
        raise RuntimeError("First volume does not start with the 7z signature")
    # Layout after the magic, version and CRC: u64 offset, u64 size, u32 CRC (little-endian).
    pointer_start = len(SEVENZIP_MAGIC) + SEVENZIP_VERSION_AND_CRC_BYTES
    next_header_offset, next_header_size, _next_header_crc = struct.unpack(
        "<QQI", start_header[pointer_start:SEVENZIP_START_HEADER_BYTES])
    # NextHeaderOffset is measured from the end of the 32-byte start header.
    next_header_start = SEVENZIP_START_HEADER_BYTES + next_header_offset
    # Step 3: fetch the next header itself.
    fetched_bytes += fetch_virtual_range(volumes, next_header_start, next_header_start + next_header_size,
                                         sparse_path, connections)
    with open(sparse_path, "rb") as sparse_handle:
        sparse_handle.seek(next_header_start)
        next_header_bytes = sparse_handle.read(next_header_size)
    # Step 4: an encoded header points at the packed real header; fetch exactly that region.
    if next_header_bytes and next_header_bytes[0] == ENCODED_HEADER_PROPERTY_ID:
        header_buffer = io.BytesIO(next_header_bytes)
        header_buffer.read(1)  # consume the property id; the streams info follows
        streams_info = HeaderStreamsInfo.retrieve(header_buffer)
        packed_start = SEVENZIP_START_HEADER_BYTES + streams_info.packinfo.packpos
        packed_end = packed_start + sum(streams_info.packinfo.packsizes)
        fetched_bytes += fetch_virtual_range(volumes, packed_start, packed_end, sparse_path, connections)
    if not archive_is_openable(sparse_path):
        raise RuntimeError("Archive still not readable after fetching header regions")
    return fetched_bytes


def _folder_packed_ranges(header) -> List[Tuple[int, int]]:
    """Compute (virtual_start, virtual_end_exclusive) of every folder's packed streams.

    Args:
        header: py7zr Header with main_streams.

    Returns:
        One tuple per folder, in folder order.
    """
    packinfo = header.main_streams.packinfo
    folder_ranges: List[Tuple[int, int]] = []
    first_stream_index = 0
    for folder in header.main_streams.unpackinfo.folders:
        # A folder consumes len(packed_indices) consecutive pack streams; a simple single-coder
        # LZMA2 folder has exactly one (the list can read empty on writer-built headers, hence `or 1`).
        stream_count = len(folder.packed_indices) or 1
        packed_bytes_before = sum(packinfo.packsizes[:first_stream_index])
        packed_bytes_of_folder = sum(packinfo.packsizes[first_stream_index:first_stream_index + stream_count])
        range_start = SEVENZIP_START_HEADER_BYTES + packinfo.packpos + packed_bytes_before
        folder_ranges.append((range_start, range_start + packed_bytes_of_folder))
        first_stream_index += stream_count
    return folder_ranges


def list_archive_entries(archive_path: Path) -> List[ArchiveEntry]:
    """List non-directory entries with their folder index (reads only the header).

    A file's folder is found by counting data streams: folder j holds
    ``num_unpackstreams_folders[j]`` consecutive non-empty files.

    Args:
        archive_path: Sparse or complete archive whose header regions are present.

    Returns:
        Entries in archive order.
    """
    with py7zr.SevenZipFile(archive_path) as archive:
        substreams = archive.header.main_streams.substreamsinfo
        folder_count = len(archive.header.main_streams.unpackinfo.folders)
        # Without substreams info every folder holds exactly one file.
        streams_per_folder = list(substreams.num_unpackstreams_folders) if substreams else [1] * folder_count
        entries: List[ArchiveEntry] = []
        folder_index = 0
        streams_used_in_folder = 0
        for archive_file in archive.files:
            if archive_file.emptystream:
                continue
            # Advance past folders whose streams are used up (skips zero-stream folders too).
            while streams_used_in_folder >= streams_per_folder[folder_index]:
                folder_index += 1
                streams_used_in_folder = 0
            entries.append(ArchiveEntry(archive_file.filename, int(archive_file.uncompressed), folder_index))
            streams_used_in_folder += 1
        return entries


def folder_packed_range_by_index(archive_path: Path, folder_index: int) -> Tuple[int, int, int]:
    """Virtual packed-byte range of a folder.

    Args:
        archive_path: Archive whose header is readable.
        folder_index: Folder number.

    Returns:
        (virtual_start, virtual_end_exclusive, folder_index).
    """
    with py7zr.SevenZipFile(archive_path) as archive:
        range_start, range_end = _folder_packed_ranges(archive.header)[folder_index]
    return range_start, range_end, folder_index


def folder_packed_range(archive_path: Path, entry_name_suffix: str) -> Tuple[int, int, int]:
    """Locate the folder holding the entry whose name ends with a suffix.

    Args:
        archive_path: Archive whose header is readable.
        entry_name_suffix: Suffix identifying exactly one entry, e.g. ``short_grass__ros2_1.mcap``.

    Returns:
        (virtual_start, virtual_end_exclusive, folder_index); the end is exclusive, so
        ``end - start`` equals the folder's packed size.

    Raises:
        KeyError: If zero or several entries match.
    """
    matches = [entry for entry in list_archive_entries(archive_path) if entry.name.endswith(entry_name_suffix)]
    if len(matches) != 1:
        raise KeyError(f"{len(matches)} entries end with {entry_name_suffix!r} (expected exactly 1)")
    return folder_packed_range_by_index(archive_path, matches[0].folder_index)


def extract_matching(archive_path: Path, name_predicate: Callable[[str], bool], out_dir: Path) -> List[ArchiveEntry]:
    """Extract only entries whose name satisfies a predicate; py7zr verifies CRC32.

    After extraction each file size is re-checked against the header, which catches a
    decoder that stopped early on zero-filled (unfetched) data. Partial outputs are removed
    on failure.

    Args:
        archive_path: Sparse or complete archive with the needed folders' bytes present.
        name_predicate: Called with each archive entry name.
        out_dir: Extraction root (entry paths are recreated below it).

    Returns:
        The extracted entries.

    Raises:
        KeyError: If nothing matches.
        ArchiveExtractionError: On CRC mismatch, decode failure, or size mismatch.
    """
    wanted_entries = [entry for entry in list_archive_entries(archive_path) if name_predicate(entry.name)]
    if not wanted_entries:
        raise KeyError("No archive entry matches the request")
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted_names = [entry.name for entry in wanted_entries]

    def remove_partial_outputs() -> None:
        """Delete files left behind by a failed extraction."""
        for entry_name in wanted_names:
            (out_dir / entry_name).unlink(missing_ok=True)

    # Step 1: extract only the wanted names; py7zr checks each file's CRC32.
    try:
        with py7zr.SevenZipFile(archive_path) as archive:
            archive.extract(path=out_dir, targets=wanted_names)
    except (ArchiveError, lzma.LZMAError, EOFError, ValueError, OSError) as error:
        remove_partial_outputs()
        raise ArchiveExtractionError(f"Extraction failed (CRC/decode error): {error!r}") from error
    # Step 2: size cross-check against the header.
    for entry in wanted_entries:
        extracted_size = (out_dir / entry.name).stat().st_size
        if extracted_size != entry.size:
            remove_partial_outputs()
            raise ArchiveExtractionError(
                f"{entry.name}: extracted {extracted_size} bytes, header says {entry.size}")
    return wanted_entries


def extract_entry(archive_path: Path, entry_suffix: str, out_dir: Path) -> List[ArchiveEntry]:
    """Extract the entries whose name ends with a suffix (see extract_matching).

    Args:
        archive_path: Archive with the needed bytes present.
        entry_suffix: Name suffix, e.g. ``short_grass__ros2_1.mcap``.
        out_dir: Extraction root.

    Returns:
        The extracted entries.
    """
    return extract_matching(archive_path, lambda entry_name: entry_name.endswith(entry_suffix), out_dir)

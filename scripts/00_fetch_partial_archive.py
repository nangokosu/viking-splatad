"""Build a sparse copy of the multi-part Viking Hill 7z archive from HTTP range requests.

A 7z archive keeps its file index (the "next header") at the END of the archive, which
here lives in the last volume. To extract only the first bag chunk without pulling all
~160 GB we:
  1. read the 32-byte start header from the first volume (gives index offset + size),
  2. range-fetch just the index region from wherever it falls in the virtual
     concatenation of all volumes,
  3. range-fetch the first `--head-bytes` of the archive (the data of the earliest files),
  4. write both into one sparse file of the full virtual size (holes stay zero).
7-Zip/py7zr can then list the archive and extract any file whose packed bytes lie
entirely inside the fetched head region.

Only plain, anonymous GET requests are made (generic User-Agent, no personal info).
"""

import argparse
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

# Generic anonymous identifier, per project rule: no personal info in any request.
ANONYMOUS_USER_AGENT = "viking-splatad/0.1"
# Size of the fixed 7z signature header: 6 signature + 2 version + 4 CRC + 20 start-header.
SEVENZIP_START_HEADER_BYTES = 32
# Download chunk size for streaming range responses; only a memory/throughput trade-off.
STREAM_CHUNK_BYTES = 8 * 1024 * 1024


def fetch_range(download_url: str, first_byte: int, last_byte: int, output_handle, write_offset: int) -> int:
    """Stream bytes [first_byte, last_byte] of a URL into a file at write_offset.

    Args:
        download_url: Direct download URL that honours HTTP Range.
        first_byte: First byte offset (inclusive) within the remote file.
        last_byte: Last byte offset (inclusive) within the remote file.
        output_handle: Writable, seekable binary file object.
        write_offset: Position in the output file where first_byte is stored.

    Returns:
        Number of bytes written.
    """
    request_headers = {"Range": f"bytes={first_byte}-{last_byte}", "User-Agent": ANONYMOUS_USER_AGENT}
    bytes_written = 0
    # stream=True so a multi-GB range never sits fully in memory.
    with requests.get(download_url, headers=request_headers, stream=True, timeout=60) as response:
        # 206 proves the server honoured the range; a 200 would silently send the whole file.
        if response.status_code != 206:
            raise RuntimeError(f"Range not honoured ({response.status_code}) for {download_url}")
        output_handle.seek(write_offset)
        for data_chunk in response.iter_content(STREAM_CHUNK_BYTES):
            output_handle.write(data_chunk)
            bytes_written += len(data_chunk)
    return bytes_written


def fetch_head_parallel(download_url: str, head_last_byte: int, output_path: Path, connection_count: int) -> int:
    """Fetch bytes [0, head_last_byte] using several concurrent range requests.

    One connection to the server is throttled well below the link speed, so the head is
    split into `connection_count` equal contiguous segments fetched in parallel; each
    worker writes its segment at the matching offset through its own file handle.

    Args:
        download_url: Direct download URL that honours HTTP Range.
        head_last_byte: Last byte (inclusive) of the head region.
        output_path: Existing sparse output file (opened r+b per worker).
        connection_count: Number of parallel connections (kept modest to be polite).

    Returns:
        Total bytes written.
    """
    head_size = head_last_byte + 1
    # Ceil division so the segments cover the whole head with no gap.
    segment_size = -(-head_size // connection_count)

    def fetch_segment(segment_index: int) -> int:
        """Fetch one segment into its offset of the output file."""
        first_byte = segment_index * segment_size
        last_byte = min(first_byte + segment_size, head_size) - 1
        if first_byte > last_byte:
            return 0
        with open(output_path, "r+b") as segment_handle:
            return fetch_range(download_url, first_byte, last_byte, segment_handle, first_byte)

    with ThreadPoolExecutor(max_workers=connection_count) as executor:
        return sum(executor.map(fetch_segment, range(connection_count)))


def locate_volume(virtual_offset: int, volume_sizes: list) -> tuple:
    """Map an offset in the virtual concatenated archive to (volume_index, offset_in_volume).

    Args:
        virtual_offset: Byte offset in the concatenation of all volumes.
        volume_sizes: Size in bytes of each volume, in order.

    Returns:
        Tuple (volume_index, offset_within_volume).
    """
    remaining_offset = virtual_offset
    for volume_index, volume_size in enumerate(volume_sizes):
        # Offset falls inside this volume when it is smaller than the volume size.
        if remaining_offset < volume_size:
            return volume_index, remaining_offset
        remaining_offset -= volume_size
    raise ValueError("Offset beyond end of archive")


def main() -> None:
    """Parse arguments, fetch header + head regions and write the sparse archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, help="Zenodo record id")
    parser.add_argument("--stem", required=True, help="archive stem, e.g. ros2_jazzy.7z")
    parser.add_argument("--head-bytes", type=int, required=True, help="bytes to fetch from archive start")
    parser.add_argument("--output", type=Path, required=True, help="sparse output file")
    parser.add_argument("--connections", type=int, default=8,
                        help="parallel range connections for the head fetch")
    parser.add_argument("--tail-margin-bytes", type=int, default=0,
                        help="extra bytes fetched before the index to cover the encoded header stream")
    arguments = parser.parse_args()

    # Step 1: list volumes and sizes from the Zenodo API (JSON metadata only).
    record = requests.get(
        f"https://zenodo.org/api/records/{arguments.record}",
        headers={"User-Agent": ANONYMOUS_USER_AGENT}, timeout=60,
    ).json()
    volume_sizes_by_name = {
        file_entry["key"]: file_entry["size"]
        for file_entry in record["files"] if file_entry["key"].startswith(arguments.stem + ".")
    }
    volume_names = sorted(volume_sizes_by_name)
    volume_sizes = [volume_sizes_by_name[name] for name in volume_names]
    total_size = sum(volume_sizes)

    def url_of(volume_index: int) -> str:
        """Direct download URL of one volume."""
        return f"https://zenodo.org/records/{arguments.record}/files/{volume_names[volume_index]}?download=1"

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with open(arguments.output, "wb") as output_handle:
        # Step 2: sparse file of the full virtual size; unfetched regions stay holes.
        output_handle.truncate(total_size)

        # Step 3: read start header (first 32 bytes) to learn where the index lives.
        output_handle.seek(0)
        fetch_range(url_of(0), 0, SEVENZIP_START_HEADER_BYTES - 1, output_handle, 0)
        output_handle.flush()
        with open(arguments.output, "rb") as read_handle:
            start_header = read_handle.read(SEVENZIP_START_HEADER_BYTES)
        # Layout: 6B signature, 2B version, 4B start-header CRC, then
        # NextHeaderOffset(u64), NextHeaderSize(u64), NextHeaderCRC(u32), all little-endian.
        next_header_offset, next_header_size, _ = struct.unpack("<QQI", start_header[12:32])
        # NextHeaderOffset is measured from the end of the 32-byte start header.
        index_virtual_start = SEVENZIP_START_HEADER_BYTES + next_header_offset
        index_virtual_end = index_virtual_start + next_header_size - 1
        print(f"index at virtual [{index_virtual_start}, {index_virtual_end}] size {next_header_size}", flush=True)

        # Step 4: fetch the index region PLUS a tail margin, splitting across volumes if it
        # straddles a boundary. The 40-byte index is normally an "encoded header" pointing
        # at a compressed header stream placed just before it, so the margin must cover
        # that stream (its exact position is derived afterwards from the pointer).
        cursor = max(0, index_virtual_start - arguments.tail_margin_bytes)
        while cursor <= index_virtual_end:
            volume_index, offset_in_volume = locate_volume(cursor, volume_sizes)
            last_in_volume = min(index_virtual_end - cursor, volume_sizes[volume_index] - offset_in_volume - 1)
            fetch_range(url_of(volume_index), offset_in_volume, offset_in_volume + last_in_volume,
                        output_handle, cursor)
            cursor += last_in_volume + 1

        # Step 5: fetch the head region (data of the earliest files) from volume 0.
        head_last_byte = min(arguments.head_bytes, volume_sizes[0]) - 1
        start_time = time.time()
        # Flush index bytes first; workers reopen the file with their own handles.
        output_handle.flush()
        written = fetch_head_parallel(url_of(0), head_last_byte, arguments.output, arguments.connections)
        print(f"head: {written} bytes in {time.time() - start_time:.0f}s", flush=True)
    print(f"sparse archive: {arguments.output} (virtual size {total_size})")


if __name__ == "__main__":
    sys.exit(main())

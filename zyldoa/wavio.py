"""Streaming reader for the 19-channel, 24-bit Zylia WAV.

The recordings are ~2.4 GB, so we never load the whole file. We parse the RIFF chunks to
find the PCM data offset, then seek to arbitrary sample positions to pull short windows, and
provide a chunked iterator for whole-file envelope passes.

Long sessions (the packed 24-azimuth VBAP recordings, ~3 h) are split by the recorder at the
WAV 4 GB limit into sequential parts named `<base>.wav, <base>_1.wav, ... <base>_N.wav` with
no inter-part gap. `MultiZyliaWav` presents those parts as one continuous recording with the
same interface as `ZyliaWav`; `open_recording(folder)` picks the right reader.
"""
from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np


class ZyliaWav:
    """Random-access reader for an interleaved 24-bit PCM WAV."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._f = open(self.path, "rb")
        self._parse_header()

    def _parse_header(self) -> None:
        f = self._f
        riff = f.read(12)
        if riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise ValueError(f"Not a RIFF/WAVE file: {self.path}")
        self.n_channels = self.sample_rate = self.bits_per_sample = None
        self.data_offset = self.data_bytes = None
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            cid, size = hdr[:4], struct.unpack("<I", hdr[4:8])[0]
            if cid == b"fmt ":
                fmt = f.read(size)
                (_, self.n_channels, self.sample_rate, _, self.block_align,
                 self.bits_per_sample) = struct.unpack("<HHIIHH", fmt[:16])
            elif cid == b"data":
                self.data_offset = f.tell()
                self.data_bytes = size
                f.seek(size + (size & 1), 1)
            else:
                f.seek(size + (size & 1), 1)
        if self.data_offset is None:
            raise ValueError("No data chunk found")
        self.bytes_per_sample = self.bits_per_sample // 8
        self.n_frames = self.data_bytes // self.block_align
        self.duration_s = self.n_frames / self.sample_rate

    @staticmethod
    def _int24_to_float(raw: bytes) -> np.ndarray:
        """Little-endian signed 24-bit bytes -> float32 in [-1, 1)."""
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
        return (ints.astype(np.float32) / 8388608.0)

    def read(self, start_frame: int, n_frames: int) -> np.ndarray:
        """Read a window -> float32 array of shape (n_frames, n_channels)."""
        start_frame = max(0, int(start_frame))
        n_frames = int(min(n_frames, self.n_frames - start_frame))
        if n_frames <= 0:
            return np.zeros((0, self.n_channels), dtype=np.float32)
        self._f.seek(self.data_offset + start_frame * self.block_align)
        raw = self._f.read(n_frames * self.block_align)
        flat = self._int24_to_float(raw)
        return flat.reshape(-1, self.n_channels)

    def read_seconds(self, start_s: float, dur_s: float) -> np.ndarray:
        return self.read(int(round(start_s * self.sample_rate)),
                         int(round(dur_s * self.sample_rate)))

    def iter_chunks(self, chunk_frames: int = 1 << 20):
        """Yield (start_frame, block) covering the whole file for streaming passes."""
        pos = 0
        while pos < self.n_frames:
            n = min(chunk_frames, self.n_frames - pos)
            yield pos, self.read(pos, n)
            pos += n

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class MultiZyliaWav:
    """Sequential multi-part recording presented as one continuous `ZyliaWav`-like stream.

    Parts must share the same format (channels / rate / bit depth); frame indices are global
    across the concatenation, and `read()` stitches across part boundaries."""

    def __init__(self, paths: list[Path]):
        if not paths:
            raise ValueError("MultiZyliaWav needs at least one part")
        self.paths = [Path(p) for p in paths]
        self.parts = [ZyliaWav(p) for p in self.paths]
        p0 = self.parts[0]
        for p in self.parts[1:]:
            same = (p.n_channels == p0.n_channels and p.sample_rate == p0.sample_rate
                    and p.bits_per_sample == p0.bits_per_sample)
            if not same:
                raise ValueError(f"WAV part format mismatch: {p.path.name} vs {p0.path.name}")
        self.path = self.paths[0]
        self.n_channels = p0.n_channels
        self.sample_rate = p0.sample_rate
        self.bits_per_sample = p0.bits_per_sample
        self.block_align = p0.block_align
        self._starts = np.concatenate([[0], np.cumsum([p.n_frames for p in self.parts])])
        self.n_frames = int(self._starts[-1])
        self.duration_s = self.n_frames / self.sample_rate

    def read(self, start_frame: int, n_frames: int) -> np.ndarray:
        start_frame = max(0, int(start_frame))
        n_frames = int(min(n_frames, self.n_frames - start_frame))
        if n_frames <= 0:
            return np.zeros((0, self.n_channels), dtype=np.float32)
        chunks = []
        i = int(np.searchsorted(self._starts, start_frame, side="right")) - 1
        pos, remaining = start_frame, n_frames
        while remaining > 0 and i < len(self.parts):
            local = pos - int(self._starts[i])
            take = int(min(remaining, self.parts[i].n_frames - local))
            chunks.append(self.parts[i].read(local, take))
            pos += take
            remaining -= take
            i += 1
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)

    def read_seconds(self, start_s: float, dur_s: float) -> np.ndarray:
        return self.read(int(round(start_s * self.sample_rate)),
                         int(round(dur_s * self.sample_rate)))

    def iter_chunks(self, chunk_frames: int = 1 << 20):
        pos = 0
        while pos < self.n_frames:
            n = min(chunk_frames, self.n_frames - pos)
            yield pos, self.read(pos, n)
            pos += n

    def close(self) -> None:
        for p in self.parts:
            p.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def list_wavs(folder: str | Path) -> list[Path]:
    """Real Zylia WAVs in a session folder, ignoring hidden files and macOS AppleDouble
    resource-fork stubs ('._Take_*.wav') that network shares scatter around."""
    return sorted(p for p in Path(folder).glob("*.wav") if not p.name.startswith("."))


def open_recording(folder: str | Path) -> ZyliaWav | MultiZyliaWav:
    """Open the Zylia recording in `folder`: single WAV, or numbered multi-part session.

    Multi-part naming is `<base>.wav` + `<base>_1.wav ... <base>_N.wav`; the base is the file
    whose stem prefixes all the others. Any other multi-WAV arrangement is an error rather
    than a silent guess."""
    wavs = list_wavs(folder)
    if not wavs:
        raise FileNotFoundError(f"No .wav in {folder}")
    if len(wavs) == 1:
        return ZyliaWav(wavs[0])
    base = min(wavs, key=lambda p: len(p.stem))
    pat = re.compile(rf"^{re.escape(base.stem)}_(\d+)$")
    numbered = []
    for p in wavs:
        if p == base:
            continue
        m = pat.match(p.stem)
        if not m:
            raise ValueError(f"Unrecognized multi-part WAV naming in {folder}: "
                             f"{p.name} does not follow {base.stem}_<n>.wav")
        numbered.append((int(m.group(1)), p))
    ordered = [base] + [p for _, p in sorted(numbered)]
    return MultiZyliaWav(ordered)

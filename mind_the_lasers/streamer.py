"""
XDF replay — stream a recorded XDF file as real LSL outlets.

Replays the EEG stream in real-time so any online pipeline (classifiers,
visualisers, recorders) can connect and receive data exactly as if a live
amplifier were running.  Markers are optional.

Usage
-----
    # Basic replay (EEG only, real-time)
    python streamer.py path/to/file.xdf

    # With markers at 2× speed, looping
    python streamer.py file.xdf --markers --speed 2.0 --loop

    # Use a specific EEG stream name when the XDF has multiple streams
    python streamer.py file.xdf --eeg-name gNautilus

Arguments
---------
    xdf_path            XDF file to replay
    --eeg-name          Stream name to look for in XDF  (default: first EEG-type stream)
    --out-name          LSL outlet name for EEG         (default: same as XDF stream name)
    --chunk-size        Samples pushed per LSL call     (default: 32 ≈ 128 ms @ 250 Hz)
    --speed             Replay speed multiplier         (default: 1.0 = real-time)
    --markers           Also create a marker LSL outlet
    --marker-name       XDF stream name for markers     (default: first Markers-type stream)
    --marker-out-name   LSL outlet name for markers     (default: same as XDF stream name)
    --loop              Restart from beginning when file ends
    --list-streams      Print stream names/types in the XDF and exit
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import pyxdf
except ImportError:
    sys.exit("pyxdf not found — pip install pyxdf")

try:
    from pylsl import StreamInfo, StreamOutlet, cf_double64, cf_string
except ImportError:
    sys.exit("pylsl not found — pip install pylsl")


# ── XDF helpers ───────────────────────────────────────────────────────────────

def _str(v) -> str:
    """Unwrap single-element lists that pyxdf nests."""
    return v[0] if isinstance(v, list) else str(v)


def find_stream(streams: list, *, name: str | None = None,
                stype: str | None = None) -> dict | None:
    for s in streams:
        if stype and _str(s["info"]["type"]).upper() != stype.upper():
            continue
        if name and _str(s["info"]["name"]).lower() != name.lower():
            continue
        return s
    return None


def channel_labels(stream: dict) -> list[str]:
    n = int(_str(stream["info"]["channel_count"]))
    try:
        chans = stream["info"]["desc"][0]["channels"][0]["channel"]
        labels = [_str(c["label"]) for c in chans]
        if len(labels) == n:
            return labels
    except (KeyError, IndexError, TypeError):
        pass
    return [f"CH{i + 1}" for i in range(n)]


def list_streams(streams: list) -> None:
    print(f"{'#':<4} {'Name':<24} {'Type':<12} {'Channels':<10} {'srate':<8} Samples")
    print("-" * 70)
    for i, s in enumerate(streams):
        name   = _str(s["info"]["name"])
        stype  = _str(s["info"]["type"])
        n_ch   = int(_str(s["info"]["channel_count"]))
        srate  = float(_str(s["info"]["nominal_srate"]))
        n_samp = len(s["time_stamps"])
        print(f"{i:<4} {name:<24} {stype:<12} {n_ch:<10} {srate:<8.1f} {n_samp}")


# ── LSL outlet creation ───────────────────────────────────────────────────────

def make_eeg_outlet(stream: dict, out_name: str | None) -> StreamOutlet:
    name   = out_name or _str(stream["info"]["name"])
    n_ch   = int(_str(stream["info"]["channel_count"]))
    srate  = float(_str(stream["info"]["nominal_srate"]))
    labels = channel_labels(stream)

    info = StreamInfo(
        name=name,
        type="EEG",
        channel_count=n_ch,
        nominal_srate=srate,
        channel_format=cf_double64,
        source_id=f"xdf_replay_{name}",
    )
    # Embed channel labels so downstream tools can resolve montage
    desc  = info.desc().append_child("channels")
    for lbl in labels:
        ch = desc.append_child("channel")
        ch.append_child_value("label", lbl)
        ch.append_child_value("type", "EEG")
        ch.append_child_value("unit", "microvolts")

    outlet = StreamOutlet(info, chunk_size=0)
    print(f"[EEG outlet]  name={name!r}  channels={n_ch}  srate={srate} Hz")
    print(f"  labels: {labels}")
    return outlet


def make_marker_outlet(stream: dict, out_name: str | None) -> StreamOutlet:
    name = out_name or _str(stream["info"]["name"])
    info = StreamInfo(
        name=name,
        type="Markers",
        channel_count=1,
        nominal_srate=0,          # irregular
        channel_format=cf_string,
        source_id=f"xdf_replay_{name}",
    )
    outlet = StreamOutlet(info)
    print(f"[Markers outlet]  name={name!r}")
    return outlet


# ── replay loop ───────────────────────────────────────────────────────────────

def replay_once(eeg_stream: dict, eeg_outlet: StreamOutlet,
                marker_stream: dict | None, marker_outlet: StreamOutlet | None,
                chunk_size: int, speed: float) -> None:
    data       = eeg_stream["time_series"]        # (n_samples, n_ch) float64
    ts         = eeg_stream["time_stamps"]         # (n_samples,)
    n_samples  = len(ts)
    sfreq      = float(_str(eeg_stream["info"]["nominal_srate"]))
    chunk_dur  = chunk_size / sfreq / speed        # wall-clock seconds per chunk

    # Pre-compute marker schedule relative to EEG start
    markers: list[tuple[float, str]] = []
    if marker_stream is not None and marker_outlet is not None:
        t0_eeg = ts[0]
        for mt, mv in zip(marker_stream["time_stamps"],
                          marker_stream["time_series"]):
            rel = float(mt) - t0_eeg            # seconds after EEG start
            val = str(mv[0]).strip() if len(mv) > 0 else ""
            if val:
                markers.append((rel, val))
        markers.sort()
        print(f"  {len(markers)} markers loaded")

    print(f"Streaming {n_samples} samples ({n_samples / sfreq:.1f} s) "
          f"in chunks of {chunk_size} @ {speed}× speed …  Ctrl-C to stop")

    mk_idx        = 0
    wall_start    = time.perf_counter()
    sample_cursor = 0

    while sample_cursor < n_samples:
        chunk_end = min(sample_cursor + chunk_size, n_samples)
        chunk     = data[sample_cursor:chunk_end]          # (k, n_ch)

        # Push EEG chunk (pylsl expects list-of-samples or numpy array)
        eeg_outlet.push_chunk(chunk.tolist())

        sample_cursor = chunk_end
        elapsed_eeg   = sample_cursor / sfreq              # simulated EEG time

        # Emit markers whose time has arrived
        while mk_idx < len(markers) and markers[mk_idx][0] <= elapsed_eeg:
            marker_outlet.push_sample([markers[mk_idx][1]])
            print(f"  [marker] {markers[mk_idx][1]!r}  @ {markers[mk_idx][0]:.3f}s")
            mk_idx += 1

        # Pace to real-time (or scaled) wall clock
        target_wall = wall_start + elapsed_eeg / speed
        sleep_dur   = target_wall - time.perf_counter()
        if sleep_dur > 0:
            time.sleep(sleep_dur)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay an XDF file as live LSL streams",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("xdf_path", help="Path to XDF file")
    parser.add_argument("--eeg-name",        default=None,
                        help="XDF stream name to use as EEG source "
                             "(default: first stream with type EEG)")
    parser.add_argument("--out-name",        default=None,
                        help="LSL outlet name (default: same as XDF stream name)")
    parser.add_argument("--chunk-size",      type=int,   default=32,
                        help="Samples per LSL push (~128 ms at 250 Hz)")
    parser.add_argument("--speed",           type=float, default=1.0,
                        help="Replay speed multiplier (1.0 = real-time)")
    parser.add_argument("--markers",         action="store_true",
                        help="Also stream the marker stream")
    parser.add_argument("--marker-name",     default=None,
                        help="XDF stream name for markers "
                             "(default: first Markers-type stream)")
    parser.add_argument("--marker-out-name", default=None,
                        help="LSL outlet name for markers")
    parser.add_argument("--loop",            action="store_true",
                        help="Restart replay when file ends")
    parser.add_argument("--list-streams",    action="store_true",
                        help="Print streams in XDF and exit")
    args = parser.parse_args()

    xdf_path = Path(args.xdf_path)
    if not xdf_path.exists():
        sys.exit(f"File not found: {xdf_path}")

    print(f"Loading {xdf_path} …")
    import logging
    logging.getLogger("pyxdf").setLevel(logging.CRITICAL)
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    print(f"  {len(streams)} stream(s) found")

    if args.list_streams:
        list_streams(streams)
        return

    # ── find EEG stream ───────────────────────────────────────────────────────
    eeg_stream = find_stream(streams, name=args.eeg_name, stype="EEG")
    if eeg_stream is None:
        print("No EEG-type stream found. Available streams:")
        list_streams(streams)
        sys.exit(1)

    # ── find marker stream (optional) ─────────────────────────────────────────
    marker_stream = marker_outlet = None
    if args.markers:
        marker_stream = find_stream(streams, name=args.marker_name, stype="Markers")
        if marker_stream is None:
            # Try common alternatives
            for stype in ("Marker", "marker", "markers"):
                marker_stream = find_stream(streams, stype=stype)
                if marker_stream:
                    break
        if marker_stream is None:
            print("WARNING: --markers requested but no Markers stream found in XDF")
        else:
            marker_outlet = make_marker_outlet(marker_stream, args.marker_out_name)

    # ── create EEG outlet ─────────────────────────────────────────────────────
    eeg_outlet = make_eeg_outlet(eeg_stream, args.out_name)

    print("\nWaiting 2 s for consumers to discover the stream …")
    time.sleep(2.0)

    # ── replay loop ───────────────────────────────────────────────────────────
    run = 0
    try:
        while True:
            run += 1
            if run > 1:
                print(f"\n--- Loop {run} ---")
            replay_once(eeg_stream, eeg_outlet,
                        marker_stream, marker_outlet,
                        args.chunk_size, args.speed)
            if not args.loop:
                break
            print("Replay complete — looping …")
    except KeyboardInterrupt:
        print("\nStopped by user.")

    print("Done.")


if __name__ == "__main__":
    main()

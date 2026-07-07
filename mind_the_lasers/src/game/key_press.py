"""
Manual UDP prediction sender — keyboard-driven simulation of classifier output.

Arrow keys trigger predictions; hold duration sets confidence:
  Left  arrow  →  "left"  (idx 1)
  Right arrow  →  "right" (idx 2)
  Down  arrow  →  "rest"  (idx 0)

Confidence = min(hold_time / --conf-scale, 1.0)
             OR random [0, 1] with --random-confidence

Packet format (JSON dict, compatible with helpers.py and prediction_widget.py):
    {"prediction": 1, "label": "left", "confidence": 0.74, "extra": ""}

Esc or Ctrl-C to quit.

Usage
-----
    python send_prediction_udp.py                       # hold-time confidence
    python send_prediction_udp.py --random-confidence   # random confidence
    python send_prediction_udp.py --extra session_2     # add extra field
    python send_prediction_udp.py --ip 192.168.1.5 --port 5005
    python send_prediction_udp.py --conf-scale 0.5      # 0.5 s = 100%
"""
from __future__ import annotations
import argparse
import json
import random
import socket
import sys
import time

try:
    from pynput import keyboard
except ImportError:
    sys.exit("pynput not found — pip install pynput")

# ── key → (label, prediction_index) ──────────────────────────────────────────
# Indices match helpers.py LABEL2IDX: circle/rest=0, left=1, right=2
KEY_MAP = {
    keyboard.Key.left:  ("left",  1),
    keyboard.Key.right: ("right", 2),
    keyboard.Key.down:  ("rest",  0),
}

KEY_NAME = {
    keyboard.Key.left:  "LEFT  arrow",
    keyboard.Key.right: "RIGHT arrow",
    keyboard.Key.down:  "DOWN  arrow",
}

LABEL_COLOR = {
    "left":  "\033[94m",   # blue
    "right": "\033[92m",   # green
    "rest":  "\033[93m",   # yellow
}
RESET = "\033[0m"
BOLD  = "\033[1m"


def _parse_extra(raw: str):
    """Return raw as int → float → str, or None if empty."""
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _conf_bar(conf: float, width: int = 20) -> str:
    filled = int(round(conf * width))
    return "[" + "#" * filled + "." * (width - filled) + f"] {conf:.0%}"


def make_listener(sock: socket.socket, ip: str, port: int,
                  conf_scale: float, use_random: bool, extra) -> keyboard.Listener:
    press_times: dict = {}
    total_sent = [0]

    def on_press(key):
        if key in KEY_MAP and key not in press_times:
            press_times[key] = time.perf_counter()
            label, idx = KEY_MAP[key]
            print(f"  {KEY_NAME[key]} pressed  [{label}]", end="\r", flush=True)

    def on_release(key):
        if key == keyboard.Key.esc:
            print(f"\n{BOLD}Stopped.{RESET}  Total sent: {total_sent[0]}")
            return False                          # stops listener

        if key not in KEY_MAP:
            return

        label, pred_idx = KEY_MAP[key]
        t0 = press_times.pop(key, None)

        if use_random:
            confidence = random.random()
            hold = None
        elif t0 is not None:
            hold = time.perf_counter() - t0
            confidence = min(hold / conf_scale, 1.0)
        else:
            confidence = 0.5
            hold = None

        msg: dict = {
            "prediction": pred_idx,
            "label":      label,
            "confidence": round(confidence, 4),
        }
        if extra is not None:
            msg["extra"] = extra

        payload = json.dumps(msg).encode("utf-8")
        sock.sendto(payload, (ip, port))
        total_sent[0] += 1

        col   = LABEL_COLOR.get(label, "")
        bar   = _conf_bar(confidence)
        hold_str = f"  hold={hold:.2f}s" if hold is not None else "  (random)"
        print(
            f"  {col}{BOLD}{label.upper():6s}{RESET}  "
            f"pred={pred_idx}  conf={bar}{hold_str}  "
            f"→ {ip}:{port}  #{total_sent[0]}"
        )

    return keyboard.Listener(on_press=on_press, on_release=on_release)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send manual UDP classifier predictions via arrow keys",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ip",   default="127.0.0.1", help="UDP destination IP")
    parser.add_argument("--port", default=5005, type=int, help="UDP destination port")
    parser.add_argument("--conf-scale", default=1.0, type=float,
                        help="Hold time (seconds) that gives 100%% confidence")
    parser.add_argument("--random-confidence", action="store_true",
                        help="Use random confidence instead of hold-time")
    parser.add_argument("--extra", default="", metavar="VALUE",
                        help="Extra field to include in every packet (string/number/empty)")
    args = parser.parse_args()

    extra = _parse_extra(args.extra)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"""
{BOLD}UDP Prediction Sender{RESET}
  Target      : {args.ip}:{args.port}
  Confidence  : {"random [0,1]" if args.random_confidence else f"hold-time / {args.conf_scale:.1f}s"}
  Extra field : {repr(extra) if extra is not None else "(none)"}

{BOLD}Controls:{RESET}
  LEFT  arrow  →  left  (idx 1)
  RIGHT arrow  →  right (idx 2)
  DOWN  arrow  →  rest  (idx 0)
  ESC          →  quit

  Tap = low confidence, hold longer = higher confidence
  (hold {args.conf_scale:.1f}s for 100%)
""")

    listener = make_listener(sock, args.ip, args.port,
                              args.conf_scale, args.random_confidence, extra)
    try:
        with listener:
            listener.join()
    except KeyboardInterrupt:
        print(f"\n{BOLD}Stopped by Ctrl-C.{RESET}")
    finally:
        sock.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import socket
import sys


def send_command(host: str, port: int, payload: dict, timeout: float = 2.0) -> dict:
    message = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.sendall(message)
            sock.shutdown(socket.SHUT_WR)
            data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
    except Exception as exc:
        raise RuntimeError(f"failed to send command: {exc}") from exc
    if not data:
        raise RuntimeError("empty response from server")
    try:
        return json.loads(data.decode("utf-8").strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid response: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description="Set a track's color label via the realtime fusion server command port.")
    parser.add_argument("--host", default="192.168.0.173", help="command server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=18100, help="command server port (default: 18100)")
    parser.add_argument("--track-id", type=int, required=True, help="target track id")
    parser.add_argument("--color", required=True, help="color label to set (use 'none' to clear)")
    parser.add_argument("--timeout", type=float, default=2.0, help="command response timeout (seconds)")
    args = parser.parse_args()

    color_str = str(args.color).strip()
    color_payload = None if not color_str or color_str.lower() == "none" else color_str

    payload = {
        "cmd": "set_color",
        "track_id": args.track_id,
        "color": color_payload,
    }

    try:
        resp = send_command(args.host, args.port, payload, timeout=args.timeout)
    except RuntimeError as exc:
        print(f"[set_track_color] {exc}", file=sys.stderr)
        sys.exit(1)

    status = resp.get("status")
    if status == "ok":
        track_id = resp.get("track_id")
        color = resp.get("color")
        print(f"[set_track_color] set track {track_id} color -> {color}")
    else:
        message = resp.get("message", "unknown error")
        print(f"[set_track_color] command failed: {message}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
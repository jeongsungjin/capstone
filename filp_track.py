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
    parser = argparse.ArgumentParser(description="Flip a track's yaw via the realtime fusion server command port.")
    parser.add_argument("--host", default="192.168.0.165", help="command server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=18100, help="command server port (default: 18100)")
    parser.add_argument("--track-id", type=int, required=True, help="target track id")
    parser.add_argument("--delta", type=float, default=180.0, help="yaw delta degrees (default: 180)")
    parser.add_argument("--timeout", type=float, default=2.0, help="command response timeout (seconds)")
    args = parser.parse_args()

    payload = {
        "cmd": "flip_yaw",
        "track_id": args.track_id,
        "delta": args.delta,
    }

    try:
        resp = send_command(args.host, args.port, payload, timeout=args.timeout)
    except RuntimeError as exc:
        print(f"[flip_track] {exc}", file=sys.stderr)
        sys.exit(1)

    status = resp.get("status")
    if status == "ok":
        track_id = resp.get("track_id")
        delta = resp.get("delta")
        print(f"[flip_track] flipped track {track_id} by {delta} degrees")
    else:
        message = resp.get("message", "unknown error")
        print(f"[flip_track] command failed: {message}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
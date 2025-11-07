#!/usr/bin/env python3

import argparse
import math
import socket
import struct
import sys
import time

FMT_INT = "!iiI"  # int angle, int speed, uint32 seq
FMT_FLOAT = "!fiI"  # float angle(rad), int speed, uint32 seq


def main() -> int:
    parser = argparse.ArgumentParser(description="UDP Ackermann receiver (standalone, no ROS)")
    parser.add_argument("--bind-ip", default="0.0.0.0", help="Local bind IP")
    parser.add_argument("--bind-port", type=int, default=5555, help="Local bind port")
    parser.add_argument("--float-angle", action="store_true", help="Decode angle as float radians (!fiI)")
    parser.add_argument("--angle-scale", type=float, default=1.0, help="Scale to convert int angle to rad (INT mode)")
    parser.add_argument("--angle-invert", action="store_true", help="Invert steering sign")
    parser.add_argument("--speed-scale", type=float, default=1.0, help="Multiply received speed int by this factor")
    parser.add_argument("--throttle-sec", type=float, default=0.2, help="Print throttle (seconds)")
    parser.add_argument("--raw", action="store_true", help="Print raw integers instead of converted values")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)
    except OSError:
        pass
    sock.bind((args.bind_ip, args.bind_port))
    sock.setblocking(True)

    print(f"[RC-UDP-RECV] listening on {args.bind_ip}:{args.bind_port}")

    last_print = 0.0
    pkt_size_int = struct.calcsize(FMT_INT)
    pkt_size_float = struct.calcsize(FMT_FLOAT)
    try:
        while True:
            data, addr = sock.recvfrom(64)
            now = time.time()
            if args.float-angle if False else False:
                pass
            if args.float_angle and len(data) >= pkt_size_float:
                angle_f, speed_i, seq = struct.unpack(FMT_FLOAT, data[:pkt_size_float])
                steer = -angle_f if args.angle_invert else angle_f
                speed = float(speed_i) * args.speed_scale
                if (now - last_print) >= args.throttle_sec:
                    if args.raw:
                        print(f"from={addr[0]}:{addr[1]} seq={seq} angle_f={angle_f:.4f} speed_i={speed_i}")
                    else:
                        print(
                            f"from={addr[0]}:{addr[1]} seq={seq} steer={steer:.4f} rad ({math.degrees(steer):.1f} deg) speed={speed:.2f}"
                        )
                    last_print = now
                continue
            if (not args.float_angle) and len(data) >= pkt_size_int:
                angle_i, speed_i, seq = struct.unpack(FMT_INT, data[:pkt_size_int])
                if args.raw:
                    if (now - last_print) >= args.throttle_sec:
                        print(f"from={addr[0]}:{addr[1]} seq={seq} angle_i={angle_i} speed_i={speed_i}")
                        last_print = now
                    continue
                steer = float(angle_i) / max(1e-6, args.angle_scale)
                if args.angle_invert:
                    steer = -steer
                speed = float(speed_i) * args.speed_scale
                if (now - last_print) >= args.throttle_sec:
                    print(
                        f"from={addr[0]}:{addr[1]} seq={seq} "
                        f"steer={steer:.3f} rad ({math.degrees(steer):.1f} deg) speed={speed:.2f}"
                    )
                    last_print = now
    except KeyboardInterrupt:
        print("\n[RC-UDP-RECV] interrupted, exiting", file=sys.stderr)
    finally:
        try:
            sock.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



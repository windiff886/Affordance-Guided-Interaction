from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="rollout 演示入口占位脚本")
    parser.add_argument("--steps", type=int, default=100, help="演示步数")
    args = parser.parse_args()
    print(f"[rollout_demo] 占位入口，待接入 rollout，可视化步数={args.steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


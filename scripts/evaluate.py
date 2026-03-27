from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="评估入口占位脚本")
    parser.add_argument(
        "--config",
        default="configs/training/default.yaml",
        help="评估配置文件路径",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    print(f"[evaluate] 框架入口已创建，待实现评估流程。config={config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


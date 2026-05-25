#!/usr/bin/env python3
"""
下载基线模型 Qwen2.5-1.5B-Instruct（模型 A/B 用）。

直连 huggingface.co 超时时，按顺序尝试：
  1) HF 镜像 hf-mirror.com
  2) ModelScope（国内通常更稳）

用法:
  python scripts/download_baseline_model.py
  python scripts/download_baseline_model.py --method modelscope
  python scripts/download_baseline_model.py --method hf_mirror
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct"
HF_MIRROR = "https://hf-mirror.com"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def check_files(out_dir: Path) -> bool:
    required = ["config.json", "model.safetensors", "tokenizer.json"]
    ok = all((out_dir / f).is_file() for f in required)
    if ok:
        print(f"校验通过: {out_dir}")
        for f in required:
            size = (out_dir / f).stat().st_size
            print(f"  {f}: {size / 1024 / 1024:.1f} MB")
    return ok


def download_hf_mirror(out_dir: Path) -> None:
    os.environ["HF_ENDPOINT"] = HF_MIRROR
    print(f"[HF 镜像] HF_ENDPOINT={HF_MIRROR}")
    print(f"          保存到 {out_dir}")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )


def download_hf_cli(out_dir: Path) -> None:
    import subprocess

    env = os.environ.copy()
    env["HF_ENDPOINT"] = HF_MIRROR
    cmd = ["hf", "download", REPO_ID, "--local-dir", str(out_dir)]
    print(f"[hf CLI] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def download_modelscope(out_dir: Path) -> None:
    print(f"[ModelScope] {REPO_ID} -> {out_dir}")
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("未安装 modelscope，请执行: pip install modelscope")
        raise

    snapshot_download(
        model_id=REPO_ID,
        local_dir=str(out_dir),
        revision="master",
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out_dir",
        type=Path,
        default=project_root() / "models" / "qwen2.5-1.5B-Instruct",
    )
    p.add_argument(
        "--method",
        choices=["auto", "hf_mirror", "hf_cli", "modelscope"],
        default="auto",
        help="auto=先镜像再 ModelScope",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if check_files(out_dir):
        print("模型已存在，跳过下载。")
        return

    methods: list[tuple[str, callable]] = []
    if args.method == "auto":
        methods = [
            ("hf_mirror", download_hf_mirror),
            ("modelscope", download_modelscope),
        ]
    elif args.method == "hf_mirror":
        methods = [("hf_mirror", download_hf_mirror)]
    elif args.method == "hf_cli":
        methods = [("hf_cli", download_hf_cli)]
    else:
        methods = [("modelscope", download_modelscope)]

    last_err: Exception | None = None
    for name, fn in methods:
        try:
            print(f"\n>>> 尝试: {name}")
            fn(out_dir)
            if check_files(out_dir):
                print(f"\n下载成功。训练时使用:\n  --model_path {out_dir}")
                return
            print("下载结束但文件不完整，继续下一种方式...")
        except Exception as e:
            last_err = e
            print(f"失败 ({name}): {e}")

    print("\n全部方式均失败。可选手动方案:")
    print("  1) 浏览器打开 https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct")
    print("     下载 model.safetensors / tokenizer.json / config.json 等到:")
    print(f"     {out_dir}")
    print("  2) ModelScope: https://www.modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct")
    print("     下载后解压到同上目录")
    print("  3) 设置代理后重试: export https_proxy=http://127.0.0.1:7890")
    if last_err:
        raise SystemExit(1) from last_err
    raise SystemExit(1)


if __name__ == "__main__":
    main()

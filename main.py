# -*- coding: utf-8 -*-
"""Spaceship Titanic 项目的命令行薄入口（《统一接口定义》§3.7）。"""
from __future__ import annotations

import argparse
from collections.abc import Sequence

from spaceship.pipeline import run_full_pipeline
from spaceship.train import MODEL_REGISTRY


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析模型选择、模型列表与仅交叉验证命令行参数。"""
    parser = argparse.ArgumentParser(description="运行 Spaceship Titanic 训练与提交流程。")
    parser.add_argument(
        "--model",
        help="注册表中的模型名；未指定时使用 config.models.default_model。",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="打印全部可用模型名后退出。",
    )
    parser.add_argument(
        "--cv-only",
        action="store_true",
        help="仅执行特征构建、训练和交叉验证，不保存模型或生成提交文件。",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """执行 CLI 请求并返回进程状态码。"""
    args = _parse_args(argv)
    if args.list_models:
        print("可用模型：" + ", ".join(sorted(MODEL_REGISTRY)))
        return 0

    report = run_full_pipeline(
        model_name=args.model,
        save_model_file=not args.cv_only,
        cv_only=args.cv_only,
    )
    print(f"模型：{report.model_name}")
    print(f"5 折 CV Accuracy：{report.cv_accuracy:.6f}")
    if report.submission_path is not None:
        print(f"提交文件：{report.submission_path}")
    print(f"耗时：{report.elapsed_seconds:.2f} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

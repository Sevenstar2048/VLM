from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.analysis import analyze_failure_patterns, compare_auto_vs_human, create_human_label_template
from src.llm_eval import LLMSafetyEvaluator
from src.rule_eval import RuleBasedSafetyEvaluator
from src.video_io import MultiViewGridParser, ensure_dir, list_videos, write_keyframes


def cmd_prepare_human(args: argparse.Namespace) -> None:
    videos = list_videos(args.data_dir)
    if not videos:
        raise RuntimeError(f"在 {args.data_dir} 下没有找到视频文件")

    create_human_label_template(videos, args.out_csv)
    print(f"[OK] 已生成人工标注模板: {args.out_csv}")
    print("请将 semantic_human/logical_human/decision_human/unsafe_human 填写为 0/1")


def cmd_auto_eval(args: argparse.Namespace) -> None:
    videos = list_videos(args.data_dir)
    if not videos:
        raise RuntimeError(f"在 {args.data_dir} 下没有找到视频文件")

    if args.llm_only and not args.use_llm:
        raise RuntimeError("启用 --llm_only 时必须同时设置 --use_llm")

    ensure_dir(args.output_dir)
    keyframe_dir = str(Path(args.output_dir) / "keyframes")
    ensure_dir(keyframe_dir)

    parser = MultiViewGridParser(rows=3, cols=6)
    evaluator = None if args.llm_only else RuleBasedSafetyEvaluator(front_camera_ids=args.front_cameras)

    llm = None
    if args.use_llm:
        llm = LLMSafetyEvaluator(args.prompt_template)
        if not llm.is_enabled():
            print("[WARN] 已启用 --use_llm，但未检测到 LLM_API_KEY，跳过LLM评估")
            llm = None

    all_rows = []
    llm_rows = []

    for video in tqdm(videos, desc="Auto evaluating"):
        sampled = parser.sample_video(video, max_frames=args.max_frames)
        rule_result = evaluator.evaluate(sampled) if evaluator is not None else None
        keyframes = write_keyframes(sampled, keyframe_dir, max_export=args.export_keyframes)

        if rule_result is not None:
            row = asdict(rule_result)
            row.update(rule_result.details)
            all_rows.append(row)

        if llm is not None:
            try:
                rule_context: dict[str, object] | None = None
                if rule_result is not None:
                    rule_context = {
                        "semantic_score": rule_result.semantic_score,
                        "logical_score": rule_result.logical_score,
                        "decision_score": rule_result.decision_score,
                        "total_risk": rule_result.total_risk,
                    }

                llm_result = llm.evaluate(video_path=video, keyframes=keyframes, rule_context=rule_context)
                if llm_result is not None:
                    llm_rows.append(
                        {
                            "video_path": video,
                            **llm_result,
                        }
                    )
            except Exception as e:
                llm_rows.append({"video_path": video, "error": str(e)})

    if all_rows:
        auto_csv = str(Path(args.output_dir) / "auto_eval_rule.csv")
        pd.DataFrame(all_rows).to_csv(auto_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] 规则评估结果: {auto_csv}")

    if llm_rows:
        llm_csv = str(Path(args.output_dir) / "auto_eval_llm.csv")
        pd.DataFrame(llm_rows).to_csv(llm_csv, index=False, encoding="utf-8-sig")
        print(f"[OK] LLM评估结果: {llm_csv}")


def cmd_analyze(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)

    pattern_stats = analyze_failure_patterns(args.auto_csv, args.output_dir)
    print("[OK] 失效模式统计完成")
    print(json.dumps(pattern_stats, ensure_ascii=False, indent=2))

    if args.human_csv:
        metrics = compare_auto_vs_human(args.auto_csv, args.human_csv, args.output_dir)
        print("[OK] 自动评估 vs 人工评估 对比完成")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成式自动驾驶视频安全评测管线")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_human = sub.add_parser("prepare-human", help="生成人工标注模板")
    p_human.add_argument("--data_dir", type=str, required=True, help="视频数据集目录")
    p_human.add_argument("--out_csv", type=str, default="outputs/human_labels_template.csv")
    p_human.set_defaults(func=cmd_prepare_human)

    p_auto = sub.add_parser("auto-eval", help="运行自动评估（规则 + 可选LLM）")
    p_auto.add_argument("--data_dir", type=str, required=True, help="视频数据集目录")
    p_auto.add_argument("--output_dir", type=str, default="outputs")
    p_auto.add_argument("--max_frames", type=int, default=64, help="每条视频最大采样帧数")
    p_auto.add_argument("--export_keyframes", type=int, default=8, help="每条视频导出的关键帧数量")
    p_auto.add_argument("--front_cameras", type=int, nargs="+", default=[2, 3], help="前向摄像头ID")
    p_auto.add_argument("--use_llm", action="store_true", help="启用LLM二次评估")
    p_auto.add_argument("--llm_only", action="store_true", help="仅使用LLM评估，跳过规则法")
    p_auto.add_argument("--prompt_template", type=str, default="prompts/llm_safety_prompt.md")
    p_auto.set_defaults(func=cmd_auto_eval)

    p_ana = sub.add_parser("analyze", help="进行统计分析和人机一致性分析")
    p_ana.add_argument("--auto_csv", type=str, required=True, help="自动评估结果csv")
    p_ana.add_argument("--human_csv", type=str, default="", help="人工评估结果csv")
    p_ana.add_argument("--output_dir", type=str, default="outputs/analysis")
    p_ana.set_defaults(func=cmd_analyze)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

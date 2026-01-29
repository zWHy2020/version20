#!/usr/bin/env python3
# coding: utf-8
"""
ChatScene-v1 数据集划分脚本。

输出:
  splits/train.txt
  splits/val.txt
  splits/test.txt
  splits/split_summary.json
"""

import argparse
import csv
import json
import logging
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def detect_video_dir(data_dir: str, video_dir: Optional[str]) -> str:
    if video_dir:
        return video_dir
    candidate_video = os.path.join(data_dir, "video")
    candidate_videos = os.path.join(data_dir, "videos")
    if os.path.isdir(candidate_video):
        return candidate_video
    if os.path.isdir(candidate_videos):
        return candidate_videos
    return candidate_video


def detect_csv_path(data_dir: str, csv_path: Optional[str]) -> str:
    if csv_path:
        return csv_path
    return os.path.join(data_dir, "scenario_descriptions.csv")


def _choose_video_id_col(headers: Sequence[str], video_id_col: Optional[str]) -> str:
    if video_id_col:
        if video_id_col not in headers:
            raise ValueError(f"--video_id_col={video_id_col} 不在 CSV 列中: {headers}")
        return video_id_col
    lowered = [h.lower() for h in headers]
    for idx, name in enumerate(lowered):
        if "video" in name and "id" in name:
            return headers[idx]
    if "video_id" in lowered:
        return headers[lowered.index("video_id")]
    raise ValueError("无法识别 video_id 列，请使用 --video_id_col 指定")


def _read_csv_rows(csv_path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames or []
    return rows, headers


def _normalize_video_id(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        return ""
    value = value.replace("\\", os.sep).replace("/", os.sep)
    value = os.path.normpath(value)
    dirname, basename = os.path.split(value)
    stem, _ = os.path.splitext(basename)
    if not dirname or dirname == ".":
        return stem
    group_id = dirname.replace(os.sep, "_")
    return f"{group_id}_{stem}"


def _scan_video_ids(video_dir: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"未找到视频目录: {video_dir}")
    video_ids: Dict[str, str] = {}
    group_map: Dict[str, List[str]] = {}
    strip_prefixes = {"video", "videos"}
    for root, _, files in os.walk(video_dir):
        rel_dir = os.path.relpath(root, video_dir)
        rel_dir = "" if rel_dir == "." else rel_dir
        rel_parts = rel_dir.split(os.sep) if rel_dir else []
        if rel_parts and rel_parts[0] in strip_prefixes:
            rel_parts = rel_parts[1:]
        cleaned_rel_dir = os.sep.join(rel_parts)
        group_id = cleaned_rel_dir.replace(os.sep, "_") if cleaned_rel_dir else ""
        for fname in files:
            lower = fname.lower()
            if lower.endswith(".mp4") or lower.endswith(".avi"):
                stem = os.path.splitext(fname)[0]
                video_id = f"{group_id}_{stem}" if group_id else stem
                rel_path = os.path.join(rel_dir, fname) if rel_dir else fname
                video_ids[video_id] = rel_path
                group_key = group_id or stem
                group_map.setdefault(group_key, []).append(video_id)
    return video_ids, group_map


def _assign_groups(
    groups: List[Tuple[str, List[str]]],
    ratios: Tuple[float, float, float],
    rng: random.Random,
) -> Tuple[List[str], List[str], List[str]]:
    rng.shuffle(groups)
    total = sum(len(ids) for _, ids in groups)
    train_target = int(total * ratios[0])
    val_target = int(total * ratios[1])
    counts = {"train": 0, "val": 0, "test": 0}
    splits = {"train": [], "val": [], "test": []}

    for _, ids in groups:
        placed = False
        for split_name, target in (("train", train_target), ("val", val_target)):
            if counts[split_name] + len(ids) <= target:
                splits[split_name].extend(ids)
                counts[split_name] += len(ids)
                placed = True
                break
        if not placed:
            splits["test"].extend(ids)
            counts["test"] += len(ids)
    return splits["train"], splits["val"], splits["test"]


def _assign_random(
    video_ids: List[str],
    ratios: Tuple[float, float, float],
    rng: random.Random,
) -> Tuple[List[str], List[str], List[str]]:
    rng.shuffle(video_ids)
    total = len(video_ids)
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])
    return video_ids[:train_end], video_ids[train_end:val_end], video_ids[val_end:]


def main() -> None:
    parser = argparse.ArgumentParser(description="ChatScene-v1 Train/Val/Test 划分")
    parser.add_argument("--data_dir", type=str, required=True, help="ChatScene-v1 根目录")
    parser.add_argument("--video_dir", type=str, default=None, help="视频目录（默认 data_dir/video 或 data_dir/videos）")
    parser.add_argument("--csv_path", type=str, default=None, help="CSV 路径（默认 data_dir/scenario_descriptions.csv）")
    parser.add_argument("--out_dir", type=str, default=None, help="输出目录（默认 data_dir/splits）")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group_by_cols", type=str, default=None, help="逗号分隔列名")
    parser.add_argument("--video_id_col", type=str, default=None, help="CSV 中 video_id 列名")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.data_dir, "splits")
    video_dir = detect_video_dir(args.data_dir, args.video_dir)
    csv_path = detect_csv_path(args.data_dir, args.csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到 CSV 文件: {csv_path}")

    rows, headers = _read_csv_rows(csv_path)
    video_id_col = _choose_video_id_col(headers, args.video_id_col)
    available_videos, video_groups = _scan_video_ids(video_dir)
    available_ids = set(available_videos.keys())

    csv_video_ids = []
    for row in rows:
        vid = _normalize_video_id(str(row.get(video_id_col, "")))
        if vid:
            csv_video_ids.append(vid)

    matched_set = set()
    missing_videos = set()
    for vid in csv_video_ids:
        if vid in video_groups:
            matched_set.update(video_groups[vid])
        elif vid in available_ids:
            matched_set.add(vid)
        else:
            missing_videos.add(vid)
    matched_ids = sorted(matched_set)
    missing_videos = sorted(missing_videos)
    extra_videos = sorted(available_ids - matched_set)

    rng = random.Random(args.seed)
    group_cols = []
    group_active = False
    if args.group_by_cols:
        group_cols = [c.strip() for c in args.group_by_cols.split(",") if c.strip()]
        missing_cols = [c for c in group_cols if c not in headers]
        if missing_cols:
            logger.warning("group_by_cols 缺失列 %s，将退化为普通随机划分", missing_cols)
        else:
            group_active = True

    if group_active:
        group_map: Dict[str, List[str]] = {}
        for row in rows:
            vid = _normalize_video_id(str(row.get(video_id_col, "")))
            if vid in video_groups:
                ids = video_groups[vid]
            elif vid in available_ids:
                ids = [vid]
            else:
                continue
            key = "|".join(str(row.get(col, "")).strip() for col in group_cols)
            group_map.setdefault(key, []).extend(ids)
        groups = [(key, sorted(set(ids))) for key, ids in group_map.items()]
        train_ids, val_ids, test_ids = _assign_groups(
            groups,
            (args.train_ratio, args.val_ratio, args.test_ratio),
            rng,
        )
    else:
        train_ids, val_ids, test_ids = _assign_random(
            matched_ids,
            (args.train_ratio, args.val_ratio, args.test_ratio),
            rng,
        )

    os.makedirs(out_dir, exist_ok=True)
    for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        path = os.path.join(out_dir, f"{name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for vid in ids:
                f.write(f"{vid}\n")
        logger.info("已写入 %s (样本数=%d)", path, len(ids))

    summary = {
        "total_csv_rows": len(rows),
        "total_video_files": len(available_ids),
        "matched_video_ids": len(matched_ids),
        "missing_videos_in_fs": len(missing_videos),
        "extra_videos_without_csv": len(extra_videos),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "group_by_cols_requested": group_cols,
        "group_by_cols_applied": group_active,
        "split_counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "missing_video_ids": missing_videos,
    }
    summary_path = os.path.join(out_dir, "split_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("已写入划分摘要: %s", summary_path)


if __name__ == "__main__":
    main()
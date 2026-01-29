#!/usr/bin/env python3
# coding: utf-8
"""
多模态 JSCC 数据准备脚本（Manifest v2）

核心变化：
- 以“视频”为主键：每个 video_id 仅出现一次，保留 captions 与 keyframes 列表
- 支持一次性提取 K 张关键帧，命名：keyframes/{video_id}_{i:02d}.jpg
- 生成 train_manifest_v2.json / val_manifest_v2.json（可选保留旧版）
- 提供统计：unique video 数、caption 分布、keyframe 分布、缺失文件计数
"""

import argparse
import collections
import csv
import hashlib
import json
import logging
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def detect_video_root(data_dir: str) -> str:
    """自动检测视频目录，优先 data_dir/videos/all，其次 data_dir/videos。"""
    candidate_all = os.path.join(data_dir, "videos", "all")
    candidate_default = os.path.join(data_dir, "videos")
    if os.path.isdir(candidate_all):
        return candidate_all
    return candidate_default


class MSRVTTDatasetDownloader:
    """MSR-VTT 数据集下载器（保持向后兼容，便于补全缺失视频）。"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.video_dir = os.path.join(data_dir, "videos")
        self.annotation_file = os.path.join(data_dir, "train_val_videodatainfo.json")
        os.makedirs(self.video_dir, exist_ok=True)
        self.dataset_info = {
            "annotation_url": "https://raw.githubusercontent.com/ms-multimedia-challenge/2017-msr-vtt-contest/master/data/train_val_videodatainfo.json",
            "video_base_url": "https://www.robots.ox.ac.uk/~maxbain/frozen-thoughts/msrvtt/videos/",
            "video_extension": ".mp4",
        }

    def download_annotation(self) -> bool:
        if os.path.exists(self.annotation_file):
            logger.info("找到已存在的注释文件: %s", self.annotation_file)
            return True
        try:
            logger.info("本地未找到注释文件，正在下载...")
            response = requests.get(self.dataset_info["annotation_url"], stream=True, timeout=30)
            response.raise_for_status()
            with open(self.annotation_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("注释文件已保存到: %s", self.annotation_file)
            return True
        except Exception as exc:  # pragma: no cover - 下载失败时提示用户
            logger.error("下载注释文件失败: %s", exc)
            return False

    def load_annotations(self) -> Dict[str, Any]:
        with open(self.annotation_file, "r", encoding="utf-8") as f:
            annotations = json.load(f)
        logger.info("成功加载注释文件，包含 %d 个句子", len(annotations.get("sentences", [])))
        return annotations

    def download_video(self, video_id: str) -> Optional[str]:
        video_path = os.path.join(self.video_dir, f"{video_id}{self.dataset_info['video_extension']}")
        if os.path.exists(video_path):
            return video_path
        video_url = f"{self.dataset_info['video_base_url']}{video_id}{self.dataset_info['video_extension']}"
        try:
            response = requests.get(video_url, stream=True, timeout=60)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            with open(video_path, "wb") as f:
                with tqdm(total=total_size, unit="B", unit_scale=True, desc=f"下载 {video_id}") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            logger.info("视频已下载: %s", video_path)
            return video_path
        except Exception as exc:  # pragma: no cover - 仅用于诊断
            logger.error("下载视频 %s 失败: %s", video_id, exc)
            return None


def compute_keyframe_indices(total_frames: int, num_keyframes: int) -> List[int]:
    if total_frames <= 0:
        return []
    if num_keyframes == 1:
        return [total_frames // 2]
    return [min(total_frames - 1, int(round(i * (total_frames - 1) / float(num_keyframes - 1)))) for i in range(num_keyframes)]


def extract_keyframes(video_path: str, save_dir: str, video_id: str, num_keyframes: int) -> Tuple[List[str], int]:
    """按需提取关键帧，返回保存的相对路径列表与失败计数。"""
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("无法打开视频文件: %s", video_path)
        return [], 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # 兼容非标准视频，顺序计数
        total_frames = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            total_frames += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    indices = compute_keyframe_indices(total_frames, num_keyframes)
    extracted: List[str] = []
    failure = 0
    for idx, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.error("读取视频 %s 第 %d 帧失败", video_id, frame_idx)
            failure += 1
            continue
        filename = f"{video_id}_{idx:02d}.jpg"
        save_path = os.path.join(save_dir, filename)
        if not cv2.imwrite(save_path, frame):
            logger.error("保存关键帧失败: %s", save_path)
            failure += 1
            continue
        extracted.append(save_path)
    cap.release()
    return extracted, failure


def aggregate_annotations(annotations: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    videos = annotations.get("videos", [])
    sentences = annotations.get("sentences", [])
    video_info = {video["video_id"]: video for video in videos}
    captions_by_video: Dict[str, List[str]] = collections.defaultdict(list)
    for s in sentences:
        captions_by_video[s["video_id"]].append(s["caption"])
    return video_info, captions_by_video


def detect_chatscene_video_dir(data_dir: str, video_dir: Optional[str]) -> str:
    if video_dir:
        return video_dir
    candidate_video = os.path.join(data_dir, "video")
    candidate_videos = os.path.join(data_dir, "videos")
    if os.path.isdir(candidate_video):
        return candidate_video
    if os.path.isdir(candidate_videos):
        return candidate_videos
    return candidate_video


def detect_chatscene_keyframes_dir(data_dir: str, keyframes_dir: Optional[str]) -> str:
    if keyframes_dir:
        return keyframes_dir
    candidate_key_frames = os.path.join(data_dir, "key_frames")
    candidate_keyframes = os.path.join(data_dir, "keyframes")
    if os.path.isdir(candidate_key_frames):
        return candidate_key_frames
    if os.path.isdir(candidate_keyframes):
        return candidate_keyframes
    return candidate_key_frames


def detect_chatscene_csv(data_dir: str, csv_path: Optional[str]) -> str:
    if csv_path:
        return csv_path
    candidate = os.path.join(data_dir, "scenario_descriptions.csv")
    return candidate


def _read_split_file(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split 文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _choose_text_col(headers: List[str], text_col: Optional[str]) -> str:
    if text_col:
        if text_col not in headers:
            raise ValueError(f"--text_col={text_col} 不在 CSV 列中: {headers}")
        return text_col
    lowered = [h.lower() for h in headers]
    for idx, name in enumerate(lowered):
        if "description" in name:
            return headers[idx]
    for idx, name in enumerate(lowered):
        if "text" in name:
            return headers[idx]
    raise ValueError("未找到可用的文本列，请使用 --text_col 指定")


def _choose_video_id_col(headers: List[str], video_id_col: Optional[str]) -> str:
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


def _collect_keyframes(keyframes_dir: str, video_id: str) -> List[str]:
    candidates: List[str] = []
    parts = video_id.split("_")
    if len(parts) >= 5 and parts[0] == "scenario" and parts[2] == "behavior":
        scenario = f"{parts[0]}_{parts[1]}"
        behavior = f"{parts[2]}_{parts[3]}"
        clip = "_".join(parts[4:])
        nested_dir = os.path.join(keyframes_dir, scenario, behavior, clip)
        for candidate_dir in (nested_dir, os.path.join(nested_dir, "frames"), os.path.join(nested_dir, "keyframes")):
            if not os.path.isdir(candidate_dir):
                continue
            for fname in sorted(os.listdir(candidate_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    candidates.append(os.path.join(candidate_dir, fname))
            if candidates:
                return candidates
    subdir = os.path.join(keyframes_dir, video_id)
    if os.path.isdir(subdir):
        for fname in sorted(os.listdir(subdir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                candidates.append(os.path.join(subdir, fname))
        if candidates:
            return candidates
    for fname in sorted(os.listdir(keyframes_dir)):
        lower = fname.lower()
        if not lower.endswith((".jpg", ".jpeg", ".png")):
            continue
        if fname.startswith(f"{video_id}_") or fname.startswith(video_id):
            candidates.append(os.path.join(keyframes_dir, fname))
    return candidates


def _dedupe_texts(texts: Iterable[str]) -> List[str]:
    seen = set()
    cleaned: List[str] = []
    for text in texts:
        if text is None:
            continue
        stripped = str(text).strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        cleaned.append(stripped)
    return cleaned


def _scan_chatscene_video_ids(video_dir: str) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    if not os.path.isdir(video_dir):
        return {}, {}
    video_ids: Dict[str, str] = {}
    group_map: Dict[str, List[str]] = {}
    for root, dirs, files in os.walk(video_dir):
        dirs.sort()
        files.sort()
        rel_dir = os.path.relpath(root, video_dir)
        rel_dir = "" if rel_dir == "." else rel_dir
        group_id = rel_dir.replace(os.sep, "_") if rel_dir else ""
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


def build_chatscene_manifests(
    data_dir: str,
    video_dir: str,
    keyframes_dir: str,
    csv_path: str,
    splits_dir: str,
    manifest_suffix: str,
    max_samples: Optional[int],
    num_keyframes: Optional[int],
    strict: bool,
    text_col: Optional[str],
    video_id_col: Optional[str],
    seed: int = 42,
) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到 CSV 文件: {csv_path}")
    train_ids = _read_split_file(os.path.join(splits_dir, "train.txt"))
    val_ids = _read_split_file(os.path.join(splits_dir, "val.txt"))
    test_ids = _read_split_file(os.path.join(splits_dir, "test.txt"))

    video_id_map, group_map = _scan_chatscene_video_ids(video_dir)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        video_col = _choose_video_id_col(headers, video_id_col)
        text_column = _choose_text_col(headers, text_col)
        text_by_video: Dict[str, List[str]] = collections.defaultdict(list)
        unmatched_ids = set()
        for row in reader:
            vid = str(row.get(video_col, "")).strip()
            if not vid:
                continue
            text = row.get(text_column, "")
            if vid in video_id_map:
                text_by_video[vid].append(text)
            elif vid in group_map:
                for child_id in group_map[vid]:
                    text_by_video[child_id].append(text)
            else:
                unmatched_ids.add(vid)
    if unmatched_ids:
        logger.warning(
            "CSV 中有 %d 个 video_id 未匹配到视频或分组（示例: %s）",
            len(unmatched_ids),
            ", ".join(sorted(unmatched_ids)[:5]),
        )

    rng = random.Random(seed)
    stats = {
        "missing_video": 0,
        "missing_keyframes": 0,
        "missing_text": 0,
        "total_samples": 0,
        "keyframe_dist": [],
        "text_dist": [],
    }

    def build_entries(video_ids: List[str], split_name: str) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for vid in video_ids:
            if max_samples and stats["total_samples"] >= max_samples:
                break
            video_rel_path = video_id_map.get(vid)
            if video_rel_path:
                video_path = os.path.join(video_dir, video_rel_path)
            else:
                video_path = os.path.join(video_dir, f"{vid}.mp4")
            if not os.path.exists(video_path):
                stats["missing_video"] += 1
                if strict:
                    continue
                video_rel = ""
            else:
                video_rel = os.path.relpath(video_path, data_dir)
            keyframes = _collect_keyframes(keyframes_dir, vid)
            if num_keyframes and keyframes:
                if split_name == "train":
                    if len(keyframes) > num_keyframes:
                        keyframes = rng.sample(keyframes, num_keyframes)
                else:
                    keyframes = keyframes[:num_keyframes]
            if not keyframes:
                stats["missing_keyframes"] += 1
                if strict:
                    continue
            texts = _dedupe_texts(text_by_video.get(vid, []))
            if not texts:
                stats["missing_text"] += 1
                if strict:
                    continue
            entry = {
                "video": {"file": video_rel},
                "image": {"files": [os.path.relpath(p, data_dir) for p in keyframes]},
                "text": {"texts": texts},
                "meta": {"video_id": vid},
            }
            entries.append(entry)
            stats["total_samples"] += 1
            stats["keyframe_dist"].append(len(keyframes))
            stats["text_dist"].append(len(texts))
        return entries

    train_entries = build_entries(train_ids, "train")
    val_entries = build_entries(val_ids, "val")
    test_entries = build_entries(test_ids, "test")

    suffix = manifest_suffix.strip()
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    train_manifest_path = os.path.join(data_dir, f"train_manifest{suffix}.json")
    val_manifest_path = os.path.join(data_dir, f"val_manifest{suffix}.json")
    test_manifest_path = os.path.join(data_dir, f"test_manifest{suffix}.json")

    save_manifest(train_entries, train_manifest_path)
    save_manifest(val_entries, val_manifest_path)
    save_manifest(test_entries, test_manifest_path)

    summarize_stats(
        {
            "missing_video": stats["missing_video"],
            "missing_keyframes": stats["missing_keyframes"],
            "failed_keyframe": 0,
            "total_videos": stats["total_samples"],
            "unique_video_ids": set(),
            "caption_dist": stats["text_dist"],
            "keyframe_dist": stats["keyframe_dist"],
        }
    )


def build_manifest_entries(
    data_dir: str,
    keyframe_dir: str,
    video_root: str,
    video_info: Dict[str, Dict[str, Any]],
    captions_by_video: Dict[str, List[str]],
    num_keyframes: int,
    downloader: MSRVTTDatasetDownloader,
    force_download: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    train_entries: List[Dict[str, Any]] = []
    val_entries: List[Dict[str, Any]] = []
    stats = {
        "missing_video": 0,
        "missing_keyframes": 0,
        "failed_keyframe": 0,
        "total_videos": 0,
        "unique_video_ids": set(),
        "caption_dist": [],
        "keyframe_dist": [],
    }
    for video_id, info in tqdm(video_info.items(), desc="构建manifest(v2)"):
        split = info.get("split")
        if split not in {"train", "val"}:
            logger.warning("视频 %s split=%s 未知，跳过", video_id, split)
            continue
        captions = captions_by_video.get(video_id, [])
        if len(captions) == 0:
            logger.warning("视频 %s 未找到 caption，跳过", video_id)
            continue
        video_path = os.path.join(video_root, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            if force_download:
                video_path = downloader.download_video(video_id) or ""
            if not video_path or not os.path.exists(video_path):
                logger.error("缺失视频文件: %s", video_id)
                stats["missing_video"] += 1
                continue
        keyframes, fail_count = extract_keyframes(video_path, keyframe_dir, video_id, num_keyframes)
        stats["failed_keyframe"] += fail_count
        if not keyframes:
            stats["missing_keyframes"] += 1
            logger.error("未能为视频 %s 提取关键帧，跳过该视频", video_id)
            continue
        entry = {
            "video": {"file": os.path.relpath(video_path, data_dir)},
            "image": {"files": [os.path.relpath(p, data_dir) for p in keyframes]},
            "text": {"texts": captions},
            "meta": {"video_id": video_id},
        }
        if split == "train":
            train_entries.append(entry)
        else:
            val_entries.append(entry)
        stats["total_videos"] += 1
        stats["unique_video_ids"].add(video_id)
        stats["caption_dist"].append(len(captions))
        stats["keyframe_dist"].append(len(keyframes))
    return train_entries, val_entries, stats


def summarize_stats(stats: Dict[str, Any]) -> None:
    import numpy as np

    def describe(values: List[int]) -> str:
        if not values:
            return "n=0"
        arr = np.array(values)
        return f"n={len(values)}, min={arr.min()}, max={arr.max()}, mean={arr.mean():.2f}, median={np.median(arr):.2f}"

    logger.info("=" * 60)
    logger.info("构建完成，统计信息：")
    logger.info("总视频数: %d (unique ids)", stats["total_videos"])
    logger.info("缺失视频: %d, 关键帧失败: %d, 无关键帧样本: %d", stats["missing_video"], stats["failed_keyframe"], stats["missing_keyframes"])
    logger.info("caption 分布: %s", describe(stats["caption_dist"]))
    logger.info("keyframe 分布: %s", describe(stats["keyframe_dist"]))
    logger.info("=" * 60)


def save_manifest(entries: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    logger.info("已保存 manifest: %s (条目数=%d)", path, len(entries))


def main() -> None:
    parser = argparse.ArgumentParser(description="生成多模态JSCC manifest v2（每视频一条记录）")
    parser.add_argument("--dataset", type=str, choices=["msrvtt", "chatscene_v1"], default="msrvtt")
    parser.add_argument("--data_dir", type=str, default="./data", help="数据集根目录")
    parser.add_argument("--keyframe_dir", type=str, default="keyframes", help="关键帧相对/绝对目录（默认 data_dir/keyframes）")
    parser.add_argument("--num_keyframes", type=int, default=4, help="每个视频提取的关键帧数量")
    parser.add_argument("--max_samples", type=int, default=None, help="（可选）限制处理的视频数量")
    parser.add_argument("--force_download", action="store_true", help="缺失视频时尝试下载 MSR-VTT 源视频")
    parser.add_argument("--manifest_suffix", type=str, default="_v2", help="输出文件后缀，例如 _v2 -> train_manifest_v2.json")
    parser.add_argument("--keep_legacy", action="store_true", help="同时保留旧版 train_manifest.json/val_manifest.json（不覆盖）")
    parser.add_argument("--verbose", action="store_true", help="启用 DEBUG 日志")
    parser.add_argument("--video_dir", type=str, default=None, help="ChatScene-v1 视频目录（默认 data_dir/video 或 data_dir/videos）")
    parser.add_argument("--keyframes_dir", type=str, default=None, help="ChatScene-v1 关键帧目录（默认 data_dir/key_frames 或 data_dir/keyframes）")
    parser.add_argument("--csv_path", type=str, default=None, help="ChatScene-v1 CSV 路径（默认 data_dir/scenario_descriptions.csv）")
    parser.add_argument("--splits_dir", type=str, default=None, help="ChatScene-v1 划分目录（默认 data_dir/splits）")
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="严格模式：缺失视频/关键帧/文本时跳过",
    )
    parser.add_argument("--text_col", type=str, default=None, help="ChatScene-v1 CSV 文本列名")
    parser.add_argument("--video_id_col", type=str, default=None, help="ChatScene-v1 CSV video_id 列名")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    os.makedirs(args.data_dir, exist_ok=True)
    if args.dataset == "msrvtt":
        keyframe_dir = args.keyframe_dir
        if not os.path.isabs(keyframe_dir):
            keyframe_dir = os.path.join(args.data_dir, keyframe_dir)

        downloader = MSRVTTDatasetDownloader(args.data_dir)
        if not downloader.download_annotation():
            raise RuntimeError("无法获取 train_val_videodatainfo.json")
        annotations = downloader.load_annotations()
        video_info, captions_by_video = aggregate_annotations(annotations)
        if args.max_samples:
            # 仅保留前 max_samples 个 video_id
            limited_keys = list(video_info.keys())[: args.max_samples]
            video_info = {vid: video_info[vid] for vid in limited_keys}
        video_root = detect_video_root(args.data_dir)
        train_entries, val_entries, stats = build_manifest_entries(
            data_dir=args.data_dir,
            keyframe_dir=keyframe_dir,
            video_root=video_root,
            video_info=video_info,
            captions_by_video=captions_by_video,
            num_keyframes=args.num_keyframes,
            downloader=downloader,
            force_download=args.force_download,
        )

        suffix = args.manifest_suffix.strip()
        if suffix and not suffix.startswith("_"):
            suffix = "_" + suffix
        train_manifest_path = os.path.join(args.data_dir, f"train_manifest{suffix}.json")
        val_manifest_path = os.path.join(args.data_dir, f"val_manifest{suffix}.json")
        save_manifest(train_entries, train_manifest_path)
        save_manifest(val_entries, val_manifest_path)
        summarize_stats(stats)

        if args.keep_legacy:
            legacy_train = os.path.join(args.data_dir, "train_manifest.json")
            legacy_val = os.path.join(args.data_dir, "val_manifest.json")
            if not os.path.exists(legacy_train):
                save_manifest(train_entries, legacy_train)
            if not os.path.exists(legacy_val):
                save_manifest(val_entries, legacy_val)
    else:
        video_dir = detect_chatscene_video_dir(args.data_dir, args.video_dir)
        keyframes_dir = detect_chatscene_keyframes_dir(args.data_dir, args.keyframes_dir)
        csv_path = detect_chatscene_csv(args.data_dir, args.csv_path)
        splits_dir = args.splits_dir or os.path.join(args.data_dir, "splits")
        build_chatscene_manifests(
            data_dir=args.data_dir,
            video_dir=video_dir,
            keyframes_dir=keyframes_dir,
            csv_path=csv_path,
            splits_dir=splits_dir,
            manifest_suffix=args.manifest_suffix,
            max_samples=args.max_samples,
            num_keyframes=args.num_keyframes,
            strict=args.strict,
            text_col=args.text_col,
            video_id_col=args.video_id_col,
        )


if __name__ == "__main__":
    main()

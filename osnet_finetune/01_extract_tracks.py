"""01_extract_tracks.py

AVI 영상을 입력받아 YOLOv8 + ByteTrack 으로 사람을 검출/추적하고,
track_id 별 크롭 이미지와 메타데이터 CSV 를 생성한다.

출력 구조:
    <out>/cam01/<track_id>/<frame_index>.jpg
    <out>/cam02/...
    data/tracks_meta.csv

CSV 컬럼:
    camera, track_id, frame, t_sec, x1, y1, x2, y2, conf, file

사용 예:
    python 01_extract_tracks.py \
        --videos data/raw_videos/cam01.avi data/raw_videos/cam02.avi data/raw_videos/cam03.avi \
        --out data/crops \
        --meta data/tracks_meta.csv \
        --yolo-weights yolov8s.pt \
        --conf 0.4 --iou 0.5 \
        --sample-fps 2
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

CAM_RE = re.compile(r"cam(\d+)", re.IGNORECASE)


def parse_camera_id(video_path: Path) -> int:
    """파일명에서 camera id 추출 (cam01.avi -> 1)."""
    m = CAM_RE.search(video_path.stem)
    if not m:
        raise ValueError(
            f"파일명에서 camera id를 찾을 수 없습니다: {video_path.name}. "
            "'cam01.avi' 형식이어야 합니다."
        )
    return int(m.group(1))


def quality_ok(
    crop: np.ndarray,
    bbox: Tuple[int, int, int, int],
    min_w: int,
    min_h: int,
    blur_thresh: float,
) -> bool:
    """간단한 품질 필터: 최소 크기 + Laplacian variance(blur) 체크."""
    x1, y1, x2, y2 = bbox
    if (x2 - x1) < min_w or (y2 - y1) < min_h:
        return False
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < blur_thresh:
        return False
    return True


def process_video(
    video_path: Path,
    out_root: Path,
    meta_writer: csv.writer,
    model: YOLO,
    args: argparse.Namespace,
    camera_id: int,
) -> Dict[int, int]:
    """단일 비디오 처리. track_id -> saved count 반환."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_every = max(int(round(fps / args.sample_fps)), 1)
    print(
        f"[cam{camera_id:02d}] fps={fps:.2f}, frames={total_frames}, "
        f"sample every {sample_every} frames -> ~{args.sample_fps} fps"
    )
    cap.release()

    cam_dir = out_root / f"cam{camera_id:02d}"
    cam_dir.mkdir(parents=True, exist_ok=True)

    saved_per_track: Dict[int, int] = {}
    last_saved_frame: Dict[int, int] = {}

    # YOLO 의 track() 은 자체적으로 영상 전체를 stream 처리한다.
    results = model.track(
        source=str(video_path),
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",  # ultralytics 내장
        classes=[0],               # person only (COCO class 0)
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )

    pbar = tqdm(total=total_frames, desc=f"cam{camera_id:02d}")
    frame_idx = -1
    for r in results:
        frame_idx += 1
        pbar.update(1)

        if r.boxes is None or r.boxes.id is None:
            continue

        # 샘플링: track마다 sample_every 간격으로만 저장 (전역 frame 단위)
        if frame_idx % sample_every != 0:
            continue

        frame = r.orig_img  # BGR
        if frame is None:
            continue

        h, w = frame.shape[:2]
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        ids = r.boxes.id.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), tid, conf in zip(boxes, ids, confs):
            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(0, min(int(x2), w - 1))
            y2 = max(0, min(int(y2), h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if not quality_ok(
                crop, (x1, y1, x2, y2), args.min_size[0], args.min_size[1], args.blur
            ):
                continue

            # track 별 디렉터리
            tdir = cam_dir / f"{int(tid):06d}"
            tdir.mkdir(parents=True, exist_ok=True)

            fname = f"{frame_idx:08d}.jpg"
            fpath = tdir / fname
            cv2.imwrite(str(fpath), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])

            saved_per_track[int(tid)] = saved_per_track.get(int(tid), 0) + 1
            last_saved_frame[int(tid)] = frame_idx

            t_sec = frame_idx / fps
            meta_writer.writerow(
                [
                    camera_id,
                    int(tid),
                    frame_idx,
                    f"{t_sec:.3f}",
                    x1, y1, x2, y2,
                    f"{float(conf):.4f}",
                    str(fpath.relative_to(out_root.parent)).replace("\\", "/"),
                ]
            )

    pbar.close()
    return saved_per_track


def main() -> None:
    p = argparse.ArgumentParser(description="YOLO+ByteTrack 트랙별 크롭 추출")
    p.add_argument("--videos", nargs="+", required=True, help="AVI 파일 경로 목록")
    p.add_argument("--out", default="data/crops", help="크롭 출력 루트")
    p.add_argument("--meta", default="data/tracks_meta.csv", help="메타데이터 CSV 경로")
    p.add_argument("--yolo-weights", default="yolov8s.pt")
    p.add_argument("--device", default="0", help="CUDA device id 또는 'cpu'")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument(
        "--sample-fps", type=float, default=2.0, help="저장할 fps (track별)"
    )
    p.add_argument(
        "--min-size", nargs=2, type=int, default=[64, 128], metavar=("W", "H")
    )
    p.add_argument(
        "--blur", type=float, default=20.0, help="Laplacian variance 최소값 (낮을수록 흐림)"
    )
    args = p.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.meta)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"YOLO weights : {args.yolo_weights}")
    print(f"Output crops : {out_root.resolve()}")
    print(f"Meta CSV     : {meta_path.resolve()}")

    model = YOLO(args.yolo_weights)

    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["camera", "track_id", "frame", "t_sec",
             "x1", "y1", "x2", "y2", "conf", "file"]
        )

        for vpath in args.videos:
            vpath = Path(vpath)
            cam_id = parse_camera_id(vpath)
            saved = process_video(vpath, out_root, writer, model, args, cam_id)
            tracks = len(saved)
            total_imgs = sum(saved.values())
            print(
                f"[cam{cam_id:02d}] tracks={tracks}, crops={total_imgs}, "
                f"avg={total_imgs / max(tracks, 1):.1f} imgs/track"
            )

    print("done.")


if __name__ == "__main__":
    main()

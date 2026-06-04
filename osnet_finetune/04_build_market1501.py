"""04_build_market1501.py

persons.json (검증된 person↔track 매핑) + crops/ 를 입력으로
Market-1501 호환 디렉터리를 생성한다.

파일명 규칙: <PID>_c<CAM>s1_<FRAME>_<NN>.jpg
    PID   : 4자리 (0001, 0002, ...)
    CAM   : 1~6
    FRAME : 6자리 frame index (zero-padded)
    NN    : 같은 frame 내 bbox 인덱스 (00~99)

출력:
    <out>/bounding_box_train/
    <out>/bounding_box_test/
    <out>/query/

ID 단위 split:
    - test_ratio 비율의 ID 를 평가용으로 분리
    - 학습 ID 와 평가 ID 는 겹치지 않음 (Re-ID 표준)
    - 각 평가 ID 에서 카메라당 1장씩 query 로, 나머지는 gallery(bounding_box_test)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

FRAME_RE = re.compile(r"(\d+)\.jpg$", re.IGNORECASE)


def parse_track_key(key: str) -> Tuple[int, int]:
    """'cam01/000017' -> (1, 17)"""
    cam_s, tr_s = key.split("/")
    return int(cam_s.replace("cam", "")), int(tr_s)


def collect_files(crops_root: Path, track_key: str) -> List[Path]:
    folder = crops_root / track_key
    return sorted(folder.glob("*.jpg")) if folder.exists() else []


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--crops", required=True)
    p.add_argument("--persons", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--min-images-per-id", type=int, default=8)
    p.add_argument("--max-images-per-id", type=int, default=80,
                   help="이 값을 넘으면 균등 샘플링으로 잘라냄")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    crops_root = Path(args.crops).resolve()
    out = Path(args.out).resolve()
    train_dir = out / "bounding_box_train"
    test_dir  = out / "bounding_box_test"
    query_dir = out / "query"
    for d in (train_dir, test_dir, query_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    persons_doc = json.loads(Path(args.persons).read_text("utf-8"))
    persons: Dict[str, Dict] = persons_doc["persons"]

    # 각 PID 별로 (cam, file_path) 목록 수집
    pid_to_imgs: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
    for pid, info in persons.items():
        for tk in info["members"]:
            cam, _tid = parse_track_key(tk)
            files = collect_files(crops_root, tk)
            for fp in files:
                pid_to_imgs[pid].append((cam, fp))

    # 최소 이미지 수 필터
    pid_to_imgs = {
        pid: imgs for pid, imgs in pid_to_imgs.items()
        if len(imgs) >= args.min_images_per_id
    }
    pids = sorted(pid_to_imgs.keys())
    print(f"검증된 PID 수: {len(persons)}, 유효 PID: {len(pids)} "
          f"(min_images_per_id={args.min_images_per_id})")
    if not pids:
        raise RuntimeError("유효 PID가 없습니다. min_images_per_id 를 낮추거나 추가 라벨링 필요")

    # ID 단위 split (test_ratio)
    random.shuffle(pids)
    n_test = max(1, int(round(len(pids) * args.test_ratio)))
    test_pids  = sorted(pids[:n_test])
    train_pids = sorted(pids[n_test:])
    print(f"train PIDs: {len(train_pids)}, test PIDs: {len(test_pids)}")

    # PID 를 0001 부터 새로 부여 (Market-1501 호환)
    # 학습 ID는 1.., 테스트 ID는 그 다음 번호로
    new_pid_map: Dict[str, int] = {}
    for i, pid in enumerate(train_pids, start=1):
        new_pid_map[pid] = i
    for j, pid in enumerate(test_pids, start=len(train_pids) + 1):
        new_pid_map[pid] = j

    def save_image(src: Path, dst_dir: Path, pid_int: int, cam: int,
                   frame: int, bbox_idx: int) -> None:
        fname = f"{pid_int:04d}_c{cam}s1_{frame:06d}_{bbox_idx:02d}.jpg"
        shutil.copy2(src, dst_dir / fname)

    def sample_balanced(imgs: List[Tuple[int, Path]], k: int) -> List[Tuple[int, Path]]:
        """이미지가 너무 많으면 카메라별 라운드로빈으로 균등 샘플링."""
        if len(imgs) <= k:
            return imgs
        by_cam: Dict[int, List[Path]] = defaultdict(list)
        for c, p in imgs:
            by_cam[c].append(p)
        out_list: List[Tuple[int, Path]] = []
        # 각 카메라에서 균등하게
        cams = list(by_cam.keys())
        while len(out_list) < k:
            progressed = False
            for c in cams:
                if by_cam[c]:
                    out_list.append((c, by_cam[c].pop(0)))
                    progressed = True
                    if len(out_list) >= k:
                        break
            if not progressed:
                break
        return out_list

    # 학습 셋
    train_imgs_total = 0
    for pid in tqdm(train_pids, desc="train"):
        imgs = sample_balanced(pid_to_imgs[pid], args.max_images_per_id)
        pid_int = new_pid_map[pid]
        for idx, (cam, fp) in enumerate(imgs):
            m = FRAME_RE.search(fp.name)
            frame = int(m.group(1)) if m else idx
            save_image(fp, train_dir, pid_int, cam, frame, idx % 100)
            train_imgs_total += 1

    # 평가 셋: query 1장/카메라, 나머지 gallery
    test_imgs_total = 0
    query_imgs_total = 0
    for pid in tqdm(test_pids, desc="test"):
        imgs = sample_balanced(pid_to_imgs[pid], args.max_images_per_id)
        pid_int = new_pid_map[pid]

        # 카메라별로 분류
        by_cam: Dict[int, List[Path]] = defaultdict(list)
        for cam, fp in imgs:
            by_cam[cam].append(fp)

        # 카메라 2개 이상에 잡힌 경우만 평가 의미가 있음
        if len(by_cam) < 2:
            # 1대 카메라만 → 학습 셋으로 양도
            for idx, (cam, fp) in enumerate(imgs):
                m = FRAME_RE.search(fp.name)
                frame = int(m.group(1)) if m else idx
                save_image(fp, train_dir, pid_int, cam, frame, idx % 100)
                train_imgs_total += 1
            continue

        for cam, fps in by_cam.items():
            random.shuffle(fps)
            # 1장 query
            q = fps.pop(0)
            m = FRAME_RE.search(q.name)
            frame = int(m.group(1)) if m else 0
            save_image(q, query_dir, pid_int, cam, frame, 0)
            query_imgs_total += 1
            # 나머지는 gallery
            for idx, fp in enumerate(fps):
                m = FRAME_RE.search(fp.name)
                frame = int(m.group(1)) if m else idx
                save_image(fp, test_dir, pid_int, cam, frame, (idx + 1) % 100)
                test_imgs_total += 1

    summary = {
        "train_pids": len(train_pids),
        "test_pids": len(test_pids),
        "train_images": train_imgs_total,
        "gallery_images": test_imgs_total,
        "query_images": query_imgs_total,
        "pid_map": new_pid_map,
    }
    (out / "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "pid_map"},
                     indent=2))
    print(f"saved Market-1501 dataset → {out}")


if __name__ == "__main__":
    main()

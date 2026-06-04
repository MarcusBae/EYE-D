"""02_propose_matches.py

01단계가 생성한 카메라별 track 크롭들을 입력 받아,
사전학습 OSNet 으로 track 평균 임베딩을 만들고, 카메라 간
"동일인 후보 클러스터" 를 제안한다.

전략 (카메라가 부분 중첩):
    1. track별 평균 임베딩 e_t = mean(OSNet(crops))  (L2 normalized)
    2. 두 track (cam_a, t_a), (cam_b, t_b) 의 score:
           sim     = cosine(e_a, e_b)              [-1, 1]
           overlap = 시간상 동시 발생 길이(초) / min(track 길이)
           score   = sim + lambda_t * tanh(overlap / tau)
       서로 다른 카메라일 때만 후보로 사용 (같은 카메라 내 track 은 그대로 두면
       Step 4 에서 자연스럽게 동일 ID로 들어감)
    3. score 가 threshold 이상인 쌍만 Union-Find 로 묶음
    4. proposals.json 생성

출력 포맷:
    {
        "tracks": [
            {"key": "cam01/000017", "camera": 1, "track_id": 17,
             "n_imgs": 84, "t_start": 12.3, "t_end": 41.7,
             "thumb": "cam01/000017/00000300.jpg"},
            ...
        ],
        "edges": [{"a": "cam01/000017", "b": "cam02/000031",
                   "sim": 0.78, "overlap": 4.2, "score": 0.91}, ...],
        "clusters": {
            "0": ["cam01/000017", "cam02/000031", "cam03/000044"],
            "1": ["cam01/000022"],
            ...
        }
    }
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# ---------- Union-Find ----------
class DSU:
    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


# ---------- OSNet 로더 ----------
def load_osnet(device: torch.device):
    """torchreid 의 osnet_x0_25 + Market-1501 pretrained 가중치 로드."""
    try:
        from torchreid import models
        from torchreid.utils import load_pretrained_weights
    except ImportError as e:
        raise ImportError(
            "torchreid 가 필요합니다. README 의 설치 안내를 참고하세요."
        ) from e

    model = models.build_model(
        name="osnet_x0_25", num_classes=1000, pretrained=True
    )
    model.eval().to(device)
    return model


# ---------- 임베딩 추출 ----------
PREPROC = transforms.Compose(
    [
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
)


def embed_images(model, paths: List[Path], device: torch.device, batch: int = 64) -> np.ndarray:
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            imgs = []
            for p in paths[i : i + batch]:
                try:
                    img = Image.open(p).convert("RGB")
                except Exception:
                    continue
                imgs.append(PREPROC(img))
            if not imgs:
                continue
            x = torch.stack(imgs).to(device)
            f = model(x)
            f = F.normalize(f, p=2, dim=1)
            feats.append(f.cpu().numpy())
    if not feats:
        return np.zeros((0, 512), dtype=np.float32)
    return np.concatenate(feats, axis=0)


# ---------- track 단위 임베딩 ----------
def per_track_embedding(
    model, crops_root: Path, meta: pd.DataFrame, device: torch.device,
    max_per_track: int = 32,
) -> Dict[str, Dict]:
    """key = 'camNN/track' -> {'feat': vec, 'n': N, 't_start': s, 't_end': s, 'thumb': path}"""
    out: Dict[str, Dict] = {}
    groups = meta.groupby(["camera", "track_id"])

    for (cam, tid), g in tqdm(groups, desc="embed tracks"):
        files = [crops_root.parent / f for f in g["file"].tolist()]
        files = [p for p in files if p.exists()]
        if len(files) < 3:
            continue  # 너무 짧은 track 제외

        # 임베딩용 샘플 (균등 샘플링)
        if len(files) > max_per_track:
            idx = np.linspace(0, len(files) - 1, max_per_track).astype(int)
            sample = [files[i] for i in idx]
        else:
            sample = files

        feats = embed_images(model, sample, device)
        if feats.shape[0] == 0:
            continue
        mean = feats.mean(axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-9)

        key = f"cam{int(cam):02d}/{int(tid):06d}"
        # 가운데 프레임을 썸네일로
        mid = files[len(files) // 2]
        out[key] = {
            "feat": mean.astype(np.float32),
            "camera": int(cam),
            "track_id": int(tid),
            "n_imgs": int(len(files)),
            "t_start": float(g["t_sec"].min()),
            "t_end": float(g["t_sec"].max()),
            "thumb": str(mid.relative_to(crops_root.parent)).replace("\\", "/"),
        }
    return out


# ---------- 시간 동시성 ----------
def temporal_overlap(a: Dict, b: Dict, offsets: Dict[int, float]) -> float:
    """초 단위 동시 발생 길이. 카메라 offset 보정 가능."""
    a_s = a["t_start"] + offsets.get(a["camera"], 0.0)
    a_e = a["t_end"]   + offsets.get(a["camera"], 0.0)
    b_s = b["t_start"] + offsets.get(b["camera"], 0.0)
    b_e = b["t_end"]   + offsets.get(b["camera"], 0.0)
    return max(0.0, min(a_e, b_e) - max(a_s, b_s))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--crops", required=True, help="01단계 출력 crops 루트")
    p.add_argument("--meta", required=True, help="tracks_meta.csv")
    p.add_argument("--out", default="data/proposals.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--sim-threshold", type=float, default=0.55,
                   help="(score) 임계값. 0.5~0.6 권장")
    p.add_argument("--lambda-t", type=float, default=0.20,
                   help="시간 보너스 가중치")
    p.add_argument("--tau", type=float, default=2.0,
                   help="시간 보너스 saturation (초)")
    p.add_argument("--offsets", default=None,
                   help="(선택) JSON: {camera_id: seconds_offset}")
    p.add_argument("--max-per-track", type=int, default=32)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    crops_root = Path(args.crops).resolve()
    meta = pd.read_csv(args.meta)
    print(f"meta rows: {len(meta)}, unique tracks: "
          f"{meta.groupby(['camera','track_id']).ngroups}")

    offsets: Dict[int, float] = {}
    if args.offsets:
        offsets = {int(k): float(v) for k, v in json.loads(
            Path(args.offsets).read_text("utf-8")).items()}
        print(f"camera offsets: {offsets}")

    model = load_osnet(device)
    tracks = per_track_embedding(model, crops_root, meta, device, args.max_per_track)
    print(f"embedded tracks: {len(tracks)}")

    keys = list(tracks.keys())
    feats = np.stack([tracks[k]["feat"] for k in keys])   # (N, D)
    sim_mat = feats @ feats.T                              # cosine (already L2)

    dsu = DSU()
    edges = []
    for i in tqdm(range(len(keys)), desc="pairs"):
        for j in range(i + 1, len(keys)):
            a, b = tracks[keys[i]], tracks[keys[j]]
            if a["camera"] == b["camera"]:
                continue  # 같은 카메라 내 track 은 cross-camera 매칭 대상 아님
            sim = float(sim_mat[i, j])
            if sim < args.sim_threshold - 0.15:
                continue   # 너무 낮으면 skip (계산량 절약)
            overlap = temporal_overlap(a, b, offsets)
            bonus = args.lambda_t * np.tanh(overlap / max(args.tau, 1e-6))
            score = sim + float(bonus)
            if score >= args.sim_threshold:
                dsu.union(keys[i], keys[j])
                edges.append(
                    {"a": keys[i], "b": keys[j],
                     "sim": round(sim, 4), "overlap": round(overlap, 2),
                     "score": round(score, 4)}
                )

    # 클러스터 구성
    clusters: Dict[str, List[str]] = defaultdict(list)
    for k in keys:
        clusters[dsu.find(k)].append(k)

    cluster_list = sorted(clusters.values(), key=lambda x: -len(x))
    cluster_dict = {str(i): sorted(c) for i, c in enumerate(cluster_list)}

    # 직렬화 가능한 track 정보 (feat 제거)
    tracks_out = []
    for k, v in tracks.items():
        tracks_out.append(
            {"key": k, "camera": v["camera"], "track_id": v["track_id"],
             "n_imgs": v["n_imgs"], "t_start": v["t_start"], "t_end": v["t_end"],
             "thumb": v["thumb"]}
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"tracks": tracks_out, "edges": edges, "clusters": cluster_dict},
            ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved proposals.json: {out}")
    print(f"clusters: total={len(cluster_dict)}, "
          f"multi-track={sum(1 for c in cluster_list if len(c) > 1)}, "
          f"singletons={sum(1 for c in cluster_list if len(c) == 1)}")


if __name__ == "__main__":
    main()

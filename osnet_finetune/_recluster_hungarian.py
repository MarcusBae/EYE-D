"""_recluster_hungarian.py

기존 proposals.json 의 edges 만 재사용하여, 카메라 쌍별 Hungarian 1:1 매칭으로
클러스터를 다시 만든다. (임베딩 재계산 없음, 1~3초)

Union-Find 의 전이성(transitivity) 폭주 문제를 해결한다.
사용:
    python _recluster_hungarian.py --min-score 0.85
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    raise SystemExit("scipy 가 필요합니다. pip install scipy")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="data/proposals.json")
    p.add_argument("--out", default="data/proposals.json")
    p.add_argument("--min-score", type=float, default=0.85,
                   help="이 점수 이상의 edge 만 Hungarian 후보로 사용")
    p.add_argument("--dry-run", action="store_true",
                   help="저장하지 않고 통계만 출력")
    args = p.parse_args()

    d = json.loads(Path(args.inp).read_text("utf-8"))
    tracks = d["tracks"]
    track_keys = [t["key"] for t in tracks]
    by_cam: Dict[int, List[str]] = defaultdict(list)
    for t in tracks:
        by_cam[t["camera"]].append(t["key"])
    cams = sorted(by_cam.keys())
    print(f"Cameras: {cams}")
    print(f"Tracks per camera: {[(c, len(by_cam[c])) for c in cams]}")
    print(f"Total tracks: {len(track_keys)}")

    # edge -> score lookup
    score_lookup: Dict[Tuple[str, str], float] = {}
    for e in d["edges"]:
        a, b = e["a"], e["b"]
        s = e["score"]
        if s >= args.min_score:
            score_lookup[(a, b)] = s
            score_lookup[(b, a)] = s
    print(f"Edges above min-score {args.min_score}: {len(score_lookup) // 2}")

    # Hungarian per camera pair
    all_pairs: List[Tuple[str, str, float]] = []
    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            ca, cb = cams[i], cams[j]
            keys_a = by_cam[ca]
            keys_b = by_cam[cb]
            n_a, n_b = len(keys_a), len(keys_b)
            if n_a == 0 or n_b == 0:
                continue

            # Cost = -score (Hungarian minimizes); 후보 없는 페어는 0 (≡ 매칭 안 됨)
            cost = np.zeros((n_a, n_b), dtype=np.float32)
            for ia, ka in enumerate(keys_a):
                for ib, kb in enumerate(keys_b):
                    s = score_lookup.get((ka, kb), 0.0)
                    if s >= args.min_score:
                        cost[ia, ib] = -s
            rows, cols = linear_sum_assignment(cost)

            pair_count = 0
            for r, c in zip(rows, cols):
                if cost[r, c] < -args.min_score + 1e-9:
                    all_pairs.append((keys_a[r], keys_b[c], -float(cost[r, c])))
                    pair_count += 1
            print(f"  cam{ca:02d}-cam{cb:02d}: Hungarian pairs = {pair_count}")

    print(f"Total Hungarian pairs: {len(all_pairs)}")

    # Union-Find with Hungarian pairs only (1:1 per camera pair → 전이성 폭주 방지)
    parent = {k: k for k in track_keys}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _s in all_pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    clusters = defaultdict(list)
    for k in track_keys:
        clusters[find(k)].append(k)
    c_list = sorted(clusters.values(), key=lambda x: -len(x))
    multi = sum(1 for c in c_list if len(c) > 1)
    sizes = [len(c) for c in c_list[:10]]
    n_size3 = sum(1 for c in c_list if len(c) == 3)
    n_size2 = sum(1 for c in c_list if len(c) == 2)
    print()
    print(f"Clusters: total={len(c_list)}, multi-track={multi}, singletons={len(c_list)-multi}")
    print(f"  size 3 (모든 카메라 매칭): {n_size3}")
    print(f"  size 2 (2개 카메라 매칭): {n_size2}")
    print(f"  top10 cluster sizes: {sizes}")

    if args.dry_run:
        print("(dry-run, 저장 안 함)")
        return

    # Save
    cluster_dict = {str(i): sorted(c) for i, c in enumerate(c_list)}
    out = dict(d)
    out["clusters"] = cluster_dict
    Path(args.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()

"""_recluster_mutual.py

Mutual Best Match + 카메라 1:1 제약으로 클러스터를 재구성한다.

핵심:
    1) 각 track t 에 대해, 다른 카메라 c 마다 "최고 점수 매칭 후보" 1개를 찾는다.
    2) (t, u) 가 서로를 최고 점수 매칭으로 지목할 때만 edge 채택 (mutual best).
    3) Union-Find 로 클러스터링.
    4) 후처리: 한 클러스터 안에 같은 카메라 트랙이 2개 이상 들어가 있으면,
       전체 클러스터를 singleton 으로 분해 (Re-ID 라벨 노이즈 방지).

결과: 모든 multi-track 클러스터의 크기는 ≤ 카메라 수 (3) 이고,
     각 카메라당 정확히 1개 track 만 들어 있다.

사용:
    python _recluster_mutual.py --min-score 0.85 [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="data/proposals.json")
    p.add_argument("--out", default="data/proposals.json")
    p.add_argument("--min-score", type=float, default=0.85)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    d = json.loads(Path(args.inp).read_text("utf-8"))
    tracks = d["tracks"]
    track_keys = [t["key"] for t in tracks]
    track_cam: Dict[str, int] = {t["key"]: t["camera"] for t in tracks}

    by_cam: Dict[int, List[str]] = defaultdict(list)
    for t in tracks:
        by_cam[t["camera"]].append(t["key"])
    cams = sorted(by_cam.keys())

    # cross-camera 점수 lookup: (a, b) → score   (a < b 정렬해서 저장)
    score_lookup: Dict[Tuple[str, str], float] = {}
    for e in d["edges"]:
        a, b = e["a"], e["b"]
        if a == b or track_cam[a] == track_cam[b]:
            continue
        if e["score"] < args.min_score:
            continue
        key = (a, b) if a < b else (b, a)
        score_lookup[key] = max(score_lookup.get(key, 0.0), e["score"])

    def get_score(a: str, b: str) -> float:
        key = (a, b) if a < b else (b, a)
        return score_lookup.get(key, 0.0)

    # 1) 각 track t 에 대해, 다른 카메라마다 best 후보 찾기
    best_in: Dict[Tuple[str, int], Tuple[str, float]] = {}
    for t in track_keys:
        ct = track_cam[t]
        for cb in cams:
            if cb == ct:
                continue
            best_score = -1.0
            best_other = None
            for u in by_cam[cb]:
                s = get_score(t, u)
                if s > best_score:
                    best_score = s
                    best_other = u
            if best_other is not None and best_score >= args.min_score:
                best_in[(t, cb)] = (best_other, best_score)

    # 2) Mutual best edge 선별
    mutual_edges: List[Tuple[str, str, float]] = []
    seen = set()
    for (t, cb), (u, s) in best_in.items():
        ct = track_cam[t]
        # u 의 ct 방향 best 가 t 인지 확인
        u_best = best_in.get((u, ct))
        if u_best is None or u_best[0] != t:
            continue
        # 중복 방지
        ek = (t, u) if t < u else (u, t)
        if ek in seen:
            continue
        seen.add(ek)
        mutual_edges.append((t, u, s))

    print(f"Tracks: {len(track_keys)},  cross-cam edges >= {args.min_score}: {len(score_lookup)}")
    print(f"Mutual-best edges: {len(mutual_edges)}")

    # 3) Union-Find 로 클러스터링
    parent = {k: k for k in track_keys}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in mutual_edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    raw_clusters = defaultdict(list)
    for k in track_keys:
        raw_clusters[find(k)].append(k)

    # 4) 후처리: 한 클러스터 안에 같은 카메라 트랙이 2개 이상이면 전체 분해
    final_clusters: List[List[str]] = []
    n_split = 0
    for c in raw_clusters.values():
        cam_counts = defaultdict(int)
        for t in c:
            cam_counts[track_cam[t]] += 1
        if any(v > 1 for v in cam_counts.values()):
            # 분해: 각 트랙을 singleton 으로
            for t in c:
                final_clusters.append([t])
            n_split += 1
        else:
            final_clusters.append(sorted(c))

    final_clusters.sort(key=lambda x: -len(x))

    multi = sum(1 for c in final_clusters if len(c) > 1)
    singletons = sum(1 for c in final_clusters if len(c) == 1)
    n_size3 = sum(1 for c in final_clusters if len(c) == 3)
    n_size2 = sum(1 for c in final_clusters if len(c) == 2)
    top10 = [len(c) for c in final_clusters[:10]]

    print()
    print(f"분해된 충돌 클러스터: {n_split}개")
    print(f"Final clusters: total={len(final_clusters)}, multi-track={multi}, singletons={singletons}")
    print(f"  size 3 (모든 카메라 매칭): {n_size3}")
    print(f"  size 2 (2개 카메라 매칭): {n_size2}")
    print(f"  top10 cluster sizes: {top10}")

    if args.dry_run:
        print("(dry-run, 저장 안 함)")
        return

    cluster_dict = {str(i): c for i, c in enumerate(final_clusters)}
    out = dict(d)
    out["clusters"] = cluster_dict
    Path(args.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()

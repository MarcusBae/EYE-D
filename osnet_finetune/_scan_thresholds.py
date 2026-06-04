"""_scan_thresholds.py — 02_propose_matches 의 proposals.json 을 다시 임베딩하지 않고
여러 threshold 값에 대해 클러스터링 결과만 비교한다.

사용:
    python _scan_thresholds.py
"""
from __future__ import annotations
import json
from collections import defaultdict, Counter
from pathlib import Path

PROPOSALS = Path("data/proposals.json")
THRS = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10]


def cluster_with_threshold(tracks, all_edges, thr):
    parent = {k: k for k in tracks}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = [e for e in all_edges if e["score"] >= thr]
    for e in edges:
        ra, rb = find(e["a"]), find(e["b"])
        if ra != rb:
            parent[ra] = rb
    clusters = defaultdict(list)
    for k in tracks:
        clusters[find(k)].append(k)
    return clusters, len(edges)


def main():
    d = json.loads(PROPOSALS.read_text("utf-8"))
    tracks = {t["key"]: t for t in d["tracks"]}
    all_edges = d["edges"]

    print(f"Total tracks: {len(tracks)},  total edges: {len(all_edges)}")
    print()
    print(f"{'THR':>5} {'edges':>8} {'clusters':>9} {'multi':>6} "
          f"{'singletons':>11}  {'top5 cluster sizes':<22}  {'multi/singletons':>16}")
    print("-" * 95)

    for thr in THRS:
        clusters, n_edges = cluster_with_threshold(tracks, all_edges, thr)
        c_list = sorted(clusters.values(), key=lambda x: -len(x))
        multi = sum(1 for c in c_list if len(c) > 1)
        singletons = sum(1 for c in c_list if len(c) == 1)
        top5 = [len(c) for c in c_list[:5]]
        ratio = f"{multi}/{singletons}"
        print(f"{thr:>5.2f} {n_edges:>8} {len(c_list):>9} {multi:>6} "
              f"{singletons:>11}  {str(top5):<22}  {ratio:>16}")

    print()
    print("해석 가이드:")
    print("  - 가장 큰 클러스터 크기가 ~3~5 (= 카메라 수 × 1~2명) 정도면 정상")
    print("  - 가장 큰 클러스터가 10+ 이면 transitivity 로 잘못 묶임 → threshold 더 올려야")
    print("  - multi-track 이 30~150 정도면 라벨링 단계로 가기 적절")
    print("  - multi-track 이 10 미만이면 threshold 너무 높음 → 한 단계 낮춰서 재시도")


if __name__ == "__main__":
    main()

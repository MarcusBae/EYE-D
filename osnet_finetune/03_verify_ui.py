"""03_verify_ui.py — Streamlit 기반 동일인 검증 UI

실행:
    streamlit run 03_verify_ui.py -- --crops data/crops \
        --proposals data/proposals.json --out data/persons.json

기능:
    - 좌측: pending / approved / discarded 큐 탐색
    - 가운데: 현재 cluster 의 track 별 썸네일 그리드
    - 우측: Approve / Split track / Merge with id / Discard 버튼
    - 상단: Save persons.json (수시로 누를 것)

persons.json 포맷:
    {
        "persons": {
            "0001": {"members": ["cam01/000017", "cam02/000031"],
                     "note": "..."},
            "0002": {...},
            ...
        },
        "discarded": ["cam03/000099", ...],
        "progress": {"reviewed_clusters": [...]}
    }
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import streamlit as st


# ---------- CLI 인자 (streamlit run 의 -- 뒤) ----------
def get_args():
    import argparse
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    p = argparse.ArgumentParser()
    p.add_argument("--crops", default="data/crops")
    p.add_argument("--proposals", default="data/proposals.json")
    p.add_argument("--out", default="data/persons.json")
    p.add_argument("--max-thumbs", type=int, default=12,
                   help="track 당 표시할 썸네일 수")
    return p.parse_args(argv)


ARGS = get_args()
CROPS_ROOT = Path(ARGS.crops).resolve()
PROPOSALS = Path(ARGS.proposals)
PERSONS = Path(ARGS.out)


# ---------- 상태 로드/저장 ----------
@st.cache_data
def load_proposals(path: str):
    return json.loads(Path(path).read_text("utf-8"))


def load_persons() -> Dict:
    if PERSONS.exists():
        return json.loads(PERSONS.read_text("utf-8"))
    return {"persons": {}, "discarded": [], "progress": {"reviewed_clusters": []}}


def save_persons(state: Dict) -> None:
    PERSONS.parent.mkdir(parents=True, exist_ok=True)
    PERSONS.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def next_pid(state: Dict) -> str:
    used = set(state["persons"].keys())
    i = 1
    while f"{i:04d}" in used:
        i += 1
    return f"{i:04d}"


# ---------- 썸네일 수집 ----------
def thumbs_for_track(track_key: str, n: int) -> List[Path]:
    """cam01/000017 -> data/crops/cam01/000017/*.jpg 에서 n장 균등 샘플."""
    folder = CROPS_ROOT / track_key
    if not folder.exists():
        return []
    files = sorted(folder.glob("*.jpg"))
    if not files:
        return []
    if len(files) <= n:
        return files
    step = len(files) / n
    return [files[int(i * step)] for i in range(n)]


# ---------- UI ----------
st.set_page_config(layout="wide", page_title="EYE-D Re-ID verifier")
st.title("EYE-D Re-ID — 동일인 검증")

proposals = load_proposals(str(PROPOSALS))
tracks_idx = {t["key"]: t for t in proposals["tracks"]}
clusters: Dict[str, List[str]] = proposals["clusters"]

if "state" not in st.session_state:
    st.session_state.state = load_persons()
state = st.session_state.state

reviewed = set(state["progress"]["reviewed_clusters"])
all_cids = list(clusters.keys())
pending_cids = [c for c in all_cids if c not in reviewed]

# 상단 컨트롤
top1, top2, top3, top4 = st.columns([1, 1, 1, 2])
with top1:
    if st.button("💾 Save persons.json", use_container_width=True):
        save_persons(state)
        st.success(f"saved → {PERSONS}")
with top2:
    st.metric("Persons", len(state["persons"]))
with top3:
    st.metric("Reviewed", f"{len(reviewed)} / {len(all_cids)}")
with top4:
    show_only_multi = st.checkbox(
        "다중-track cluster만 보기 (≥2 카메라)", value=True
    )

# 클러스터 선택
candidates = pending_cids
if show_only_multi:
    candidates = [c for c in candidates if len(clusters[c]) >= 2]

if not candidates:
    st.info("처리할 cluster 가 없습니다. (모두 검토 완료 또는 필터에 걸림)")
    st.stop()

if "cursor" not in st.session_state:
    st.session_state.cursor = 0
st.session_state.cursor = min(st.session_state.cursor, len(candidates) - 1)

cid = candidates[st.session_state.cursor]
members = clusters[cid]

nav1, nav2, nav3 = st.columns([1, 4, 1])
with nav1:
    if st.button("◀ Prev", use_container_width=True) and st.session_state.cursor > 0:
        st.session_state.cursor -= 1
        st.rerun()
with nav2:
    st.subheader(f"Cluster #{cid} — {len(members)} tracks "
                 f"({st.session_state.cursor + 1}/{len(candidates)})")
with nav3:
    if st.button("Next ▶", use_container_width=True) and st.session_state.cursor < len(candidates) - 1:
        st.session_state.cursor += 1
        st.rerun()

# 멤버 별 thumbnail 표시
keep_keys: List[str] = []
discard_keys: List[str] = []

for tk in members:
    t = tracks_idx.get(tk, {})
    st.markdown(
        f"**{tk}** — cam{t.get('camera','?'):02d} · "
        f"track {t.get('track_id','?')} · "
        f"{t.get('n_imgs','?')} imgs · "
        f"{t.get('t_start',0):.1f}s → {t.get('t_end',0):.1f}s"
    )
    paths = thumbs_for_track(tk, ARGS.max_thumbs)
    if not paths:
        st.warning(f"이미지를 찾을 수 없음: {tk}")
        continue
    cols = st.columns(min(len(paths), 6))
    for i, pth in enumerate(paths):
        with cols[i % len(cols)]:
            st.image(str(pth), use_container_width=True)

    keep = st.checkbox(f"keep {tk}", value=True, key=f"keep_{cid}_{tk}")
    if keep:
        keep_keys.append(tk)
    else:
        discard_keys.append(tk)

st.divider()

# 액션 버튼
act1, act2, act3, act4 = st.columns(4)
with act1:
    if st.button("✅ Approve as ONE person", use_container_width=True,
                 disabled=len(keep_keys) == 0):
        pid = next_pid(state)
        state["persons"][pid] = {"members": keep_keys, "note": f"cluster {cid}"}
        state["discarded"].extend(discard_keys)
        reviewed.add(cid)
        state["progress"]["reviewed_clusters"] = sorted(reviewed)
        save_persons(state)
        st.success(f"승인 → person {pid}")
        st.rerun()

with act2:
    target_pid = st.text_input("Merge target PID (예: 0003)", "")
    if st.button("🔗 Merge into PID", use_container_width=True,
                 disabled=(not target_pid or target_pid not in state["persons"])):
        state["persons"][target_pid]["members"].extend(keep_keys)
        state["persons"][target_pid]["members"] = sorted(
            set(state["persons"][target_pid]["members"]))
        state["discarded"].extend(discard_keys)
        reviewed.add(cid)
        state["progress"]["reviewed_clusters"] = sorted(reviewed)
        save_persons(state)
        st.success(f"merged → {target_pid}")
        st.rerun()

with act3:
    if st.button("✂️ Split — 각 track 을 별도 person 으로",
                 use_container_width=True, disabled=len(keep_keys) == 0):
        for tk in keep_keys:
            pid = next_pid(state)
            state["persons"][pid] = {"members": [tk],
                                     "note": f"split from cluster {cid}"}
        state["discarded"].extend(discard_keys)
        reviewed.add(cid)
        state["progress"]["reviewed_clusters"] = sorted(reviewed)
        save_persons(state)
        st.success(f"split → {len(keep_keys)} persons")
        st.rerun()

with act4:
    if st.button("🗑 Discard whole cluster",
                 use_container_width=True):
        state["discarded"].extend(members)
        reviewed.add(cid)
        state["progress"]["reviewed_clusters"] = sorted(reviewed)
        save_persons(state)
        st.success("discarded")
        st.rerun()

st.caption(
    "Tip: keep 체크 해제 후 ✅ Approve 하면, 해제된 track 만 discarded 로 빠지고 "
    "나머지가 한 person 으로 묶입니다. "
    "수시로 💾 Save 를 눌러 안전하게 작업하세요."
)

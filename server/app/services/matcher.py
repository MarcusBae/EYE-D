from __future__ import annotations
import os
import asyncpg


async def find_or_create_global_id(
    conn: asyncpg.Connection,
    embedding_str: str,
    threshold: float | None = None,
) -> tuple[int, float | None, bool]:
    """
    embedding_str에 가장 가까운 사람을 찾아 global_id를 반환.
    임계값 이내가 없으면 새 person을 만들어 새 global_id 부여.

    반환: (global_id, similarity, matched)
        - matched=True 면 기존 사람과 매칭, similarity가 코사인 유사도
        - matched=False 면 새 사람 생성, similarity=None
    """
    if threshold is None:
        threshold = float(os.getenv("REID_SIMILARITY_THRESHOLD", "0.73"))
    distance_threshold = 1.0 - threshold  # 0.85 → 0.15

    # 최근 24시간 내 global_id가 부여된 탐지 중 가장 가까운 1건
    row = await conn.fetchrow(
        """
        SELECT global_id,
               (embedding_identity <=> $1::vector) AS distance
        FROM detections
        WHERE global_id IS NOT NULL
          AND detected_at > NOW() - INTERVAL '24 hours'
        ORDER BY embedding_identity <=> $1::vector
        LIMIT 1
        """,
        embedding_str,
    )

    if row and row["distance"] is not None and row["distance"] < distance_threshold:
        global_id = row["global_id"]
        distance = row["distance"]
        return int(global_id), 1.0 - float(distance), True  # type: ignore

    # 임계값 안에 못 들어왔으므로 새 person 생성
    new_id = await conn.fetchval(
        "INSERT INTO persons DEFAULT VALUES RETURNING global_id"
    )
    if new_id is None:
        raise ValueError("Failed to create new person")
    return int(new_id), None, False  # type: ignore[arg-type]

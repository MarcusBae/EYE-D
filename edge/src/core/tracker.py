"""
tracker.py
----------
BoxMOT 기반의 사람 추적(Person Tracking) 모듈.
탐지된 사람들의 Bounding Box를 기반으로 실시간으로 동일 인물을 추적하고 고유 ID를 부여합니다.
"""

import logging
from typing import List, Optional
import numpy as np
import torch

# Monkey-patch BoxMOT's assert_cuda_available to bypass CUDA validation
# and allow XPU (Intel GPU) support
try:
    import boxmot.utils.torch_utils as boxmot_torch
    boxmot_torch.assert_cuda_available = lambda device: None
except Exception:
    pass

from boxmot.trackers.tracker_zoo import create_tracker

logger = logging.getLogger(__name__)


class TrackResult:
    """단일 객체의 추적 결과 데이터 구조."""

    def __init__(self, track_id: int, bbox: List[int], confidence: float, class_id: int = 0):
        """
        Args:
            track_id: 추적 고유 ID (사람 ID)
            bbox: 바운딩 박스 좌표 [xmin, ymin, xmax, ymax] (정수형 리스트)
            confidence: 추적/탐지 신뢰도 (0.0 ~ 1.0)
            class_id: 클래스 식별자 (사람: 0)
        """
        self.track_id = track_id
        self.bbox = [int(x) for x in bbox]
        self.confidence = float(confidence)
        self.class_id = int(class_id)

    def to_dict(self) -> dict:
        """딕셔너리 포맷으로 변환 (직렬화용)."""
        return {
            'track_id': self.track_id,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'class_id': self.class_id,
        }

    def __repr__(self):
        return f"TrackResult(id={self.track_id}, bbox={self.bbox}, conf={self.confidence:.2f})"


class PersonTracker:
    """BoxMOT 추적기를 래핑하는 사람 추적 컴포넌트."""

    is_loaded = False

    def __init__(self, tracker_type: str = 'botsort', reid_weights: str = 'osnet_x0_25_msmt17.pt', device: str = None):
        """
        Args:
            tracker_type: 사용할 BoxMOT 추적기 종류 ('botsort', 'deepocsort', 'bytetrack' 등)
            reid_weights: 추적기 내부 Re-ID 동작 시 사용할 가중치 경로/이름
            device: 연산 장치 명시적 지정 ('cpu', 'cuda', 'xpu' 등)
        """
        self.tracker_type = tracker_type
        self.reid_weights = reid_weights
        self.device = device
        self.tracker = None
        self._initialize_tracker()

    def _initialize_tracker(self):
        """BoxMOT 트래커 인스턴스를 생성하고 초기화합니다."""
        try:
            if self.device is not None:
                device = self.device
            else:
                if torch.cuda.is_available():
                    device = 'cuda:0'
                elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                    device = 'xpu'
                else:
                    device = 'cpu'
            half = True if (device != 'cpu' and 'cpu' not in str(device)) else False
            
            logger.info(f"Initializing BoxMOT tracker '{self.tracker_type}' on {device} (half={half})...")
            self.tracker = create_tracker(
                self.tracker_type,
                reid_weights=self.reid_weights,
                device=device,
                half=half
            )
            self.is_loaded = True
            logger.info(f"BoxMOT tracker '{self.tracker_type}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load BoxMOT tracker: {e}")
            raise

    def update(self, dets: np.ndarray, frame: np.ndarray) -> List[TrackResult]:
        """추적 상태를 업데이트하고 현재 프레임에서 활성화된 추적 리스트를 반환합니다.

        Args:
            dets: 탐지 데이터 (NumPy array, shape: (N, 6)). 포맷: [x1, y1, x2, y2, confidence, class_id]
            frame: 현재 이미지 프레임 (numpy array BGR 형식).

        Returns:
            현재 프레임 내에서 추적된 인물들의 TrackResult 객체 리스트.
        """
        if not self.is_loaded or self.tracker is None:
            logger.warning("Tracker not initialized or failed to load. Returning empty track list.")
            return []

        if dets is None or len(dets) == 0:
            # 탐지 결과가 없으면 빈 배열 전달
            dets = np.empty((0, 6), dtype=np.float32)

        try:
            # BoxMOT 트래커 업데이트
            # output 형태: [x1, y1, x2, y2, track_id, conf, class_id, ...] 형태의 NumPy 행렬
            tracks = self.tracker.update(dets, frame)
            
            results = []
            for track in tracks:
                if len(track) < 5:
                    continue
                
                x1, y1, x2, y2 = map(int, track[:4])
                track_id = int(track[4])
                
                # confidence와 class_id 추출 (BoxMOT 버전에 따라 다를 수 있으므로 안전 처리)
                confidence = float(track[5]) if len(track) >= 6 else 1.0
                class_id = int(track[6]) if len(track) >= 7 else 0
                
                results.append(
                    TrackResult(
                        track_id=track_id,
                        bbox=[x1, y1, x2, y2],
                        confidence=confidence,
                        class_id=class_id
                    )
                )
            return results
        except Exception as e:
            logger.error(f"Error during tracker update: {e}")
            return []

    def reset(self):
        """트래커 상태를 리셋합니다. (단위 테스트 간의 격리 등에 활용)"""
        # BoxMOT의 경우 reset 기능이 내부적으로 지원되지 않을 수 있으므로 다시 초기화하거나 내부 상태를 정리합니다.
        self._initialize_tracker()

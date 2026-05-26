"""
detector.py
-----------
YOLOv8 기반의 사람 객체 탐지(Person Detection) 모듈.
입력 비디오 프레임에서 사람 객체를 감지하고 바운딩 박스와 신뢰도를 추출합니다.
"""

import logging
from typing import List
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class DetectionResult:
    """단일 객체 탐지 결과 데이터 구조."""

    def __init__(self, bbox: List[int], confidence: float, class_id: int = 0):
        """
        Args:
            bbox: 바운딩 박스 좌표 [xmin, ymin, xmax, ymax]
            confidence: 탐지 신뢰도 (0.0 ~ 1.0)
            class_id: 클래스 식별자 (사람: 0)
        """
        self.bbox = [int(x) for x in bbox]
        self.confidence = float(confidence)
        self.class_id = int(class_id)
        self.label = 'person' if class_id == 0 else f'class_{class_id}'

    def to_dict(self) -> dict:
        """딕셔너리 포맷으로 변환 (직렬화용)."""
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'label': self.label,
        }

    def __repr__(self):
        return f"DetectionResult(label={self.label}, bbox={self.bbox}, conf={self.confidence:.2f})"


class PersonDetector:
    """YOLOv8 사람 탐지 컴포넌트."""

    is_loaded = False

    def __init__(self, model_path: str = 'yolov8n.pt', conf_threshold: float = 0.5, use_tensorrt: bool = False, iou: float = 0.7, device: str = None):
        """
        Args:
            model_path: YOLOv8 가중치 파일 (.pt 또는 TensorRT .engine 파일 등)
            conf_threshold: 탐지 신뢰도 임계값 (이 값 이상의 결과만 반환)
            use_tensorrt: TensorRT 모듈 사용 여부 (향후 확장성을 위해 보존)
            iou: NMS(Non-Maximum Suppression)를 위한 IoU 임계값 (기본: 0.7)
            device: 연산 장치 명시적 지정 ('cpu', 'cuda', 'xpu' 등)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.use_tensorrt = use_tensorrt
        self.iou = iou
        self.device = device
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """YOLOv8 모델을 로드하여 초기화합니다.
        use_tensorrt=True인 경우, 자동으로 TensorRT 가속 포맷(.engine)으로 변환 및 로드합니다.
        """
        import os
        try:
            logger.info(f"Loading YOLO model from '{self.model_path}'...")
            self.model = YOLO(self.model_path)
            
            # TensorRT 가속 자동 빌드 및 로드
            if self.use_tensorrt and self.model_path.endswith('.pt'):
                engine_path = self.model_path.replace('.pt', '.engine')
                if not os.path.exists(engine_path):
                    logger.info(
                        f"TensorRT auto-export requested. Exporting model '{self.model_path}' to '{engine_path}' "
                        f"(this might take several minutes)..."
                    )
                    try:
                        # ultralytics export API를 사용하여 TensorRT(.engine) 빌드
                        # half=True (FP16 반정밀도)로 처리하여 Jetson 하드웨어 성능을 최대화
                        self.model.export(format='engine', device=0, half=True)
                        logger.info(f"Model exported to TensorRT engine successfully at '{engine_path}'")
                    except Exception as export_error:
                        logger.warning(
                            f"TensorRT export failed (likely CPU-only or CUDA toolkit conflict): {export_error}. "
                            f"Falling back to original PT model inference."
                        )
                
                # 빌드된 엔진 파일이 존재하면 리로드하여 가속화
                if os.path.exists(engine_path):
                    self.model_path = engine_path
                    self.model = YOLO(self.model_path)
                    logger.info(f"Loaded hardware-accelerated TensorRT model from '{self.model_path}'")
                else:
                    logger.warning("TensorRT engine file not found after export. Keeping original YOLO model.")

            self.is_loaded = True
            logger.info("YOLO model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """입력 이미지 프레임에서 사람(class_id = 0)을 감지합니다.

        Args:
            frame: 입력 이미지 (BGR 형식 numpy array).

        Returns:
            감지된 사람들의 DetectionResult 리스트.
        """
        if not self.is_loaded or self.model is None:
            logger.warning("YOLO model not loaded. Returning empty detection list.")
            return []

        try:
            # predict 수행 (사람 클래스 0만 감지하고 싶지만 일단 전체 감지 후 필터링하거나 YOLO conf 설정을 따름)
            predict_kwargs = {
                "conf": self.conf_threshold,
                "iou": self.iou,
                "classes": [0],  # YOLO 내부적으로 0(person) 클래스만 출력하도록 필터링
                "verbose": False
            }
            if self.device is not None:
                predict_kwargs["device"] = self.device
                
            results = self.model.predict(
                frame,
                **predict_kwargs
            )

            detections = []
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    # xmin, ymin, xmax, ymax
                    bbox = [int(x) for x in box.xyxy[0].cpu().numpy()]
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())

                    detections.append(
                        DetectionResult(
                            bbox=bbox,
                            confidence=confidence,
                            class_id=class_id
                        )
                    )
            return detections
        except Exception as e:
            logger.error(f"Error during YOLO detection: {e}")
            return []

    def to_numpy(self, detections: List[DetectionResult]) -> np.ndarray:
        """DetectionResult 리스트를 추적기(BoxMOT) 입력에 적합한 (N, 6) NumPy float32 행렬로 변환합니다.

        포맷: [xmin, ymin, xmax, ymax, confidence, class_id]

        Args:
            detections: DetectionResult 객체 리스트.

        Returns:
            shape이 (N, 6)인 numpy array.
        """
        if not detections:
            return np.empty((0, 6), dtype=np.float32)

        rows = []
        for d in detections:
            xmin, ymin, xmax, ymax = d.bbox
            rows.append([xmin, ymin, xmax, ymax, d.confidence, d.class_id])
            
        return np.array(rows, dtype=np.float32)

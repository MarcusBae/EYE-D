"""
reid_extractor.py
-----------------
OSNet 기반의 Re-Identification (Re-ID) 특징 추출 모듈.
추적 대상(사람)의 크롭 이미지 영역(ROI)으로부터 512차원 특징 벡터를 추출하여 고유의 신원 임베딩을 구성합니다.
"""

import logging
from typing import List, Dict
import cv2
import numpy as np
import torch

try:
    from torchreid.utils import FeatureExtractor
except ImportError:
    try:
        from torchreid.reid.utils import FeatureExtractor
    except ImportError:
        FeatureExtractor = None

from src.core.tracker import TrackResult

from src.core.preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)


class ReIDExtractor:
    """OSNet Re-ID 특징 벡터 추출 컴포넌트."""

    is_loaded = False

    def __init__(self, model_name: str = 'osnet_x0_25', use_onnx: bool = False, preprocessor: ImagePreprocessor = None):
        """
        Args:
            model_name: 사용할 OSNet 모델 이름 (기본값: 'osnet_x0_25')
            use_onnx: ONNX Runtime 가속 사용 여부 (기본값: False)
            preprocessor: 저해상도 악조건 극복을 위한 ROI 선명화 보정 엔진
        """
        self.model_name = model_name
        self.use_onnx = use_onnx
        self.extractor = None
        self.ort_session = None
        self.vector_dim = 512  # OSNet 피처 임베딩 기본 차원
        
        # 외부 주입이 없을 경우 기본 악조건 보정기 활성화
        self.preprocessor = preprocessor if preprocessor is not None else ImagePreprocessor()
        
        self._initialize_extractor()

    def _try_load_onnx_only(self):
        """torchreid 없이 ONNX Runtime만으로 Re-ID 추론 초기화."""
        import os
        try:
            import onnxruntime as ort
            onnx_path = f"{self.model_name}.onnx"
            if os.path.exists(onnx_path):
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.ort_session = ort.InferenceSession(onnx_path, providers=providers)
                self.is_loaded = True
                logger.info(f"Loaded hardware-accelerated ONNX model from '{onnx_path}' (torchreid-free mode)")
            else:
                logger.error(f"ONNX file not found at '{onnx_path}'. cwd={os.getcwd()}")
                self.is_loaded = False
        except ImportError:
            logger.error("onnxruntime not installed. Run: pip install onnxruntime")
            self.is_loaded = False
        except Exception as e:
            logger.error(f"ONNX-only mode init failed: {e}")
            self.is_loaded = False

    def _preprocess_for_onnx(self, roi):
        """torchreid 없이 OSNet ONNX 추론을 위한 표준 전처리 (ImageNet 정규화)."""
        img = cv2.resize(roi, (128, 256))  # W=128, H=256
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)  # HWC → CHW
        return np.expand_dims(img, axis=0).astype(np.float32)  # [1, 3, 256, 128]

    def _initialize_extractor(self):
        """FeatureExtractor를 로드하고 초기화합니다."""
        if FeatureExtractor is None:
            logger.error("torchreid library is not installed or FeatureExtractor cannot be imported.")
            if self.use_onnx:
                logger.info("use_onnx=True 감지 → torchreid 없이 ONNX-only 모드로 전환합니다.")
                self._try_load_onnx_only()
            else:
                self.is_loaded = False
            return

        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Initializing Re-ID FeatureExtractor ({self.model_name}) on {device}...")
            # FeatureExtractor 생성
            self.extractor = FeatureExtractor(
                model_name=self.model_name,
                device=device,
                verbose=False
            )
            
            # ONNX Runtime 가속 자동 빌드 및 로드
            if self.use_onnx:
                import os
                try:
                    import onnxruntime as ort
                    onnx_path = os.getenv("REID_MODEL_PATH", f"{self.model_name}.onnx")
                    
                    if not os.path.exists(onnx_path):
                        logger.info(
                            f"ONNX auto-export requested. Exporting model '{self.model_name}' to '{onnx_path}' "
                            f"(this might take several minutes)..."
                        )
                        # 파일 저장 디렉토리 자동 생성
                        dir_name = os.path.dirname(onnx_path)
                        if dir_name and not os.path.exists(dir_name):
                            os.makedirs(dir_name, exist_ok=True)
                            logger.info(f"Created model storage directory: '{dir_name}'")
                            
                        # FeatureExtractor 내부 PyTorch 모델 추출 및 평가 모드 전환
                        model = self.extractor.model
                        model.eval()
                        
                        # OSNet 표준 입력 형상: [batch, 3, 256, 128]
                        dummy_input = torch.randn(1, 3, 256, 128, device=device)
                        
                        torch.onnx.export(
                            model,
                            dummy_input,
                            onnx_path,
                            export_params=True,
                            opset_version=11,
                            do_constant_folding=True,
                            input_names=['input'],
                            output_names=['output'],
                            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                        )
                        logger.info(f"Model exported to ONNX successfully at '{onnx_path}'")
                    
                    # 빌드된 ONNX 파일 로드
                    if os.path.exists(onnx_path):
                        # GPU 가속을 위해 CUDAExecutionProvider 우선 지정
                        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                        self.ort_session = ort.InferenceSession(onnx_path, providers=providers)
                        logger.info(f"Loaded hardware-accelerated ONNX model from '{onnx_path}'")
                    else:
                        logger.warning("ONNX model file not found after export. Keeping PyTorch backend.")
                except ImportError as e:
                    logger.warning(
                        f"onnxruntime library import failed ({e}). "
                        f"Please install 'onnxruntime' or 'onnxruntime-gpu' to leverage ONNX acceleration. "
                        f"Falling back to original PyTorch backend."
                    )
                except Exception as onnx_err:
                    logger.warning(
                        f"ONNX auto-export or load failed: {onnx_err}. "
                        f"Falling back to original PyTorch backend."
                    )

            self.is_loaded = True
            logger.info("Re-ID FeatureExtractor loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Re-ID FeatureExtractor: {e}")
            self.is_loaded = False

    def extract(self, frame: np.ndarray, track_results: List[TrackResult]) -> List[Dict]:
        """현재 프레임과 추적 결과를 기반으로 각 대상의 Re-ID 특징 벡터를 추출합니다.

        Args:
            frame: 현재 프레임 이미지 (BGR 형식 numpy array).
            track_results: 현재 프레임에서 추적 성공한 인물들의 TrackResult 목록.

        Returns:
            각 추적 대상별 정보가 담긴 딕셔너리 리스트.
            [
                {
                    'track_id': int,
                    'vector': List[float], (512차원)
                    'confidence': float,
                    'bbox': List[int]
                },
                ...
            ]
        """
        if not self.is_loaded or (self.extractor is None and self.ort_session is None):
            logger.warning("ReID Extractor not loaded. Returning mock/empty vectors.")
            return []

        if not track_results:
            return []

        h, w = frame.shape[:2]
        results = []

        for track in track_results:
            x1, y1, x2, y2 = track.bbox

            # 바운딩 박스 이미지 영역 클리핑 예외 처리 (프레임 범위를 벗어나지 않도록 방지)
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            # 유효한 영역 크기 검증
            if x2 <= x1 or y2 <= y1:
                continue

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # 실환경 저해상도 악조건 대응: 인물 크롭 이미지 적응형 선명도 강화 필터 적용
            if self.preprocessor is not None:
                roi = self.preprocessor.enhance_roi(roi)

            try:
                # ONNX 가속 추론 경로
                if self.use_onnx and self.ort_session is not None:
                    if self.extractor is not None:
                        # torchreid 전처리 파이프라인 사용
                        tensor = self.extractor.preprocess(roi)
                        input_data = tensor.unsqueeze(0).cpu().numpy()  # [1, C, H, W]
                    else:
                        # torchreid 없음 → 수동 전처리
                        input_data = self._preprocess_for_onnx(roi)
                    
                    ort_inputs = {self.ort_session.get_inputs()[0].name: input_data}
                    features = self.ort_session.run(None, ort_inputs)[0]  # [1, 512]
                    vector = features[0].tolist()
                else:
                    # 기존 PyTorch 추론 경로
                    # 노트북 코드 규격 준수: (256, 128) 리사이즈 (가로 256, 세로 128)
                    # torchreid FeatureExtractor 입력에 적절하게 크기 정규화 수행
                    roi_resized = cv2.resize(roi, (256, 128))

                    # 특징 추출 수행
                    features = self.extractor(roi_resized)
                    
                    # 텐서로부터 피처 벡터를 numpy를 거쳐 파이썬 표준 float 리스트로 가공
                    vector = features[0].cpu().numpy().tolist()

                results.append({
                    'track_id': track.track_id,
                    'vector': vector,
                    'confidence': track.confidence,
                    'bbox': [x1, y1, x2, y2],
                })
            except Exception as e:
                logger.error(f"Failed to extract Re-ID features for track_id {track.track_id}: {e}")
                continue

        return results

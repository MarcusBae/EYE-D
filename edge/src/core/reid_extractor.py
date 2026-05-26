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
import base64

try:
    from torchreid.utils import FeatureExtractor
except ImportError:
    try:
        from torchreid.reid.utils import FeatureExtractor
    except ImportError:
        FeatureExtractor = None

# Support BoxMOT built-in ReID backends
PyTorchBackend = None
try:
    from boxmot.reid.backends import PyTorchBackend
except ImportError:
    try:
        from boxmot.reid.backends.pytorch_backend import PyTorchBackend
    except ImportError:
        PyTorchBackend = None

from src.core.tracker import TrackResult

from src.core.preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)


class ReIDExtractor:
    """OSNet Re-ID 특징 벡터 추출 컴포넌트."""

    is_loaded = False

    def __init__(self, model_name: str = 'osnet_x0_25', use_onnx: bool = False, preprocessor: ImagePreprocessor = None, device: str = None):
        """
        Args:
            model_name: 사용할 OSNet 모델 이름 (기본값: 'osnet_x0_25')
            use_onnx: ONNX Runtime 가속 사용 여부 (기본값: False)
            preprocessor: 저해상도 악조건 극복을 위한 ROI 선명화 보정 엔진
            device: 연산 장치 명시적 지정 ('cpu', 'cuda', 'xpu' 등)
        """
        self.model_name = model_name
        self.use_onnx = use_onnx
        self.device = device
        self.extractor = None
        self.ort_session = None
        self.vector_dim = 512  # OSNet 피처 임베딩 기본 차원
        
        # 외부 주입이 없을 경우 기본 악조건 보정기 활성화
        self.preprocessor = preprocessor if preprocessor is not None else ImagePreprocessor()
        
        self._initialize_extractor()

    def _initialize_extractor(self):
        """FeatureExtractor 또는 ONNX Runtime 세션을 로드하고 초기화합니다."""
        import os
        onnx_path = os.getenv("REID_MODEL_PATH", f"{self.model_name}.onnx")
        # edge 디렉토리 아래가 기본 실행 경로이므로, 경로가 없을 경우 edge/osnet_x0_25.onnx 등도 시도할 수 있도록 처리
        if not os.path.exists(onnx_path):
            # 대안 경로 시도
            alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), onnx_path)
            if os.path.exists(alt_path):
                onnx_path = alt_path
            elif os.path.exists(os.path.join("edge", f"{self.model_name}.onnx")):
                onnx_path = os.path.join("edge", f"{self.model_name}.onnx")

        # 1단계: BoxMOT PyTorchBackend 사용 시도 (torchreid 빌드가 실패하는 환경 대비)
        if PyTorchBackend is not None:
            try:
                if self.device is not None:
                    device = torch.device(self.device)
                else:
                    if torch.cuda.is_available():
                        device = torch.device('cuda:0')
                    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                        device = torch.device('xpu')
                    else:
                        device = torch.device('cpu')
                logger.info(f"Attempting to initialize BoxMOT PyTorchBackend for Re-ID ({self.model_name}) on {device}...")
                
                # 모델 가중치 이름 규칙 변환 (ex: osnet_x0_25 -> osnet_x0_25_msmt17.pt)
                weights_name = f"{self.model_name}_msmt17.pt" if "msmt17" not in self.model_name else f"{self.model_name}.pt"
                if not weights_name.endswith('.pt'):
                    weights_name += '.pt'
                
                self.extractor = PyTorchBackend(
                    weights=weights_name,
                    device=device,
                    half=(device.type in ('cuda', 'xpu'))
                )
                self.is_loaded = True
                logger.info("Re-ID FeatureExtractor (via BoxMOT PyTorchBackend) loaded successfully.")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize BoxMOT PyTorchBackend: {e}. Falling back to torchreid/ONNX...")

        # 2단계: torchreid 가 없을 때 ONNX Runtime 으로만 기동 시도
        if FeatureExtractor is None:
            logger.info("torchreid library not found. Attempting pure ONNX Runtime initialization...")
            if self.use_onnx:
                try:
                    import onnxruntime as ort
                    if os.path.exists(onnx_path):
                        # GPU 및 타 하드웨어 가속(OpenVINO 등)을 위해 가용한 provider 중 우선순위 지정
                        available_providers = ort.get_available_providers()
                        providers = []
                        for prov in ['CUDAExecutionProvider', 'OpenVINOExecutionProvider', 'CPUExecutionProvider']:
                            if prov in available_providers:
                                providers.append(prov)
                        for prov in available_providers:
                            if prov not in providers:
                                providers.append(prov)
                        self.ort_session = ort.InferenceSession(onnx_path, providers=providers)
                        self.is_loaded = True
                        logger.info(f"Loaded hardware-accelerated ONNX model from '{onnx_path}' (Pure ONNX mode) with providers {providers}")
                        return
                    else:
                        logger.error(f"ONNX model file not found at '{onnx_path}' and torchreid/boxmot is unavailable.")
                except ImportError as e:
                    logger.error(f"onnxruntime import failed and torchreid/boxmot is unavailable: {e}")
                except Exception as e:
                    logger.error(f"Failed to initialize pure ONNX Runtime session: {e}")
            else:
                logger.error("torchreid/boxmot is unavailable and ONNX mode is disabled.")
            
            self.is_loaded = False
            return

        # 3단계: torchreid 가 존재할 때 기존 가속화 로직 동작
        try:
            if self.device is not None:
                device = self.device
            else:
                if torch.cuda.is_available():
                    device = 'cuda'
                elif hasattr(torch, 'xpu') and torch.xpu.is_available():
                    device = 'xpu'
                else:
                    device = 'cpu'
            logger.info(f"Initializing Re-ID FeatureExtractor ({self.model_name}) on {device}...")
            # FeatureExtractor 생성
            self.extractor = FeatureExtractor(
                model_name=self.model_name,
                device=device,
                verbose=False
            )
            
            # ONNX Runtime 가속 자동 빌드 및 로드
            if self.use_onnx:
                try:
                    import onnxruntime as ort
                    
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
                        # GPU 및 타 하드웨어 가속(OpenVINO 등)을 위해 가용한 provider 중 우선순위 지정
                        available_providers = ort.get_available_providers()
                        providers = []
                        for prov in ['CUDAExecutionProvider', 'OpenVINOExecutionProvider', 'CPUExecutionProvider']:
                            if prov in available_providers:
                                providers.append(prov)
                        for prov in available_providers:
                            if prov not in providers:
                                providers.append(prov)
                        self.ort_session = ort.InferenceSession(onnx_path, providers=providers)
                        logger.info(f"Loaded hardware-accelerated ONNX model from '{onnx_path}' with providers {providers}")
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
        if not self.is_loaded:
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
                    # torchreid가 사용 가능하고 extractor가 로드되었으면 기존 전처리 사용
                    if self.extractor is not None:
                        # BGR -> RGB 변환 및 PIL Image 변환
                        from PIL import Image
                        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(roi_rgb)
                        tensor = self.extractor.preprocess(pil_img)
                        input_data = tensor.unsqueeze(0).cpu().numpy()  # [1, C, H, W]
                    else:
                        # torchreid가 없을 때는 OpenCV/NumPy로 직접 전처리 수행 (ImageNet 정규화 동일 매핑)
                        roi_resized = cv2.resize(roi, (128, 256), interpolation=cv2.INTER_LINEAR)
                        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
                        
                        img_data = roi_rgb.astype(np.float32) / 255.0
                        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                        img_data = (img_data - mean) / std
                        
                        img_data = np.transpose(img_data, (2, 0, 1))  # HWC -> CHW
                        input_data = np.expand_dims(img_data, axis=0)  # [1, C, H, W]
                    
                    ort_inputs = {self.ort_session.get_inputs()[0].name: input_data}
                    features = self.ort_session.run(None, ort_inputs)[0]  # [1, 512]
                    vector = features[0].tolist()
                else:
                    # 기존 PyTorch 추론 경로 (torchreid 또는 boxmot PyTorch 백엔드)
                    if self.extractor is None:
                        raise RuntimeError("FeatureExtractor is not initialized, cannot run PyTorch inference.")
                    
                    extractor_name = type(self.extractor).__name__
                    if extractor_name in ('PyTorchBackend', 'ONNXBackend') or not hasattr(self.extractor, 'model_name'):
                        # boxmot PyTorchBackend / ONNXBackend 계열인 경우
                        if hasattr(self.extractor, 'extract'):
                            features = self.extractor.extract([roi])
                        elif callable(self.extractor):
                            features = self.extractor([roi])
                        elif hasattr(self.extractor, 'get_features'):
                            h_roi, w_roi = roi.shape[:2]
                            features = self.extractor.get_features(np.array([[0, 0, w_roi, h_roi]]), roi)
                        else:
                            raise AttributeError(f"BoxMOT ReID backend ({extractor_name}) lacks known inference methods ('extract', '__call__', 'get_features').")
                        
                        if hasattr(features, 'cpu'):
                            vector = features[0].cpu().numpy().tolist()
                        elif hasattr(features, 'tolist'):
                            vector = features[0].tolist()
                        else:
                            vector = np.array(features[0]).tolist()
                    else:
                        # 기존 torchreid FeatureExtractor 경로
                        from PIL import Image
                        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(roi_rgb)
                        features = self.extractor([pil_img])
                        vector = features[0].cpu().numpy().tolist()

                # 썸네일 이미지 메모리 인코딩 및 Base64 변환
                thumbnail_base64 = None
                try:
                    h_roi, w_roi = roi.shape[:2]
                    max_size = 128
                    if h_roi > max_size or w_roi > max_size:
                        scale = max_size / max(h_roi, w_roi)
                        roi_resized = cv2.resize(roi, (int(w_roi * scale), int(h_roi * scale)), interpolation=cv2.INTER_AREA)
                    else:
                        roi_resized = roi

                    _, img_buffer = cv2.imencode('.jpg', roi_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    thumbnail_base64 = "data:image/jpeg;base64," + base64.b64encode(img_buffer).decode('utf-8')
                except Exception as img_err:
                    logger.warning(f"Failed to encode ROI to base64: {img_err}")

                results.append({
                    'track_id': track.track_id,
                    'vector': vector,
                    'confidence': track.confidence,
                    'bbox': [x1, y1, x2, y2],
                    'thumbnail_base64': thumbnail_base64,
                })
            except Exception as e:
                logger.error(f"Failed to extract Re-ID features for track_id {track.track_id}: {e}")
                continue

        return results

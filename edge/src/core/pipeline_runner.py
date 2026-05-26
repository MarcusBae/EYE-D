"""
pipeline_runner.py
------------------
Re-ID 파이프라인 오케스트레이터.
detector.py, tracker.py, reid_extractor.py를 조립하여 전체 흐름을 제어합니다.

역할 분리:
  - PersonDetector   (detector.py)      : YOLOv8 탐지 전담
  - PersonTracker    (tracker.py)       : BoxMOT 추적 전담
  - ReIDExtractor    (reid_extractor.py): OSNet Re-ID 특징 추출 전담
  - PipelineRunner   (본 파일)          : 위 세 모듈 + DB 저장을 조립하는 오케스트레이터
"""

import logging
import time

from src.core.preprocessor import ImagePreprocessor
from src.core.detector import PersonDetector
from src.core.tracker import PersonTracker
from src.core.reid_extractor import ReIDExtractor
from src.core.best_shot import BestShotSelector
from src.infrastructure.null_objects import NullDBClient, NullSender

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Re-ID 파이프라인 오케스트레이터.

    각 컴포넌트(detector, tracker, reid_extractor)를 조립하여
    단일 프레임 또는 다중 채널 배치를 처리합니다.

    Usage:
        runner = PipelineRunner(config={'tracker_type': 'botsort'})
        runner.start()
        result = runner.process_frame(frame, camera_id='cam_0')
        runner.stop()
    """

    def __init__(self, config: dict = None, db_client=None, http_sender=None):
        """
        Args:
            config: 파이프라인 설정 딕셔너리.
                - yolo_model      (str)  : YOLO 모델 경로 (기본: 'yolov8n.pt')
                - conf_threshold  (float): 탐지 신뢰도 임계값 (기본: 0.5)
                - use_tensorrt    (bool) : TensorRT 변환 여부 (기본: False)
                - tracker_type    (str)  : 추적기 종류 (기본: 'botsort')
                - reid_weights    (str)  : Re-ID 가중치 파일 (기본: 'osnet_x0_25_msmt17.pt')
                - reid_model      (str)  : Re-ID 모델 이름 (기본: 'osnet_x0_25')
                - collection_name (str)  : 벡터 DB 컬렉션 이름 (기본: 'reid_collection')
            db_client: VectorDBClient 인스턴스.
                       None이면 NullDBClient(로그만 출력)가 자동으로 사용됩니다.
            http_sender: ServerSender 인스턴스 (Phase 3에서 주입).
                         None이면 NullSender(로그만 출력)가 자동으로 사용됩니다.
        """
        self.config = config or {}
        self.running = False
        self.frames_processed = 0
        self._last_camera_id = "cam_0"
        self.dry_run = self.config.get('dry_run', False)

        use_awb = self.config.get('use_awb', False)
        use_blur = self.config.get('use_blur', False)
        self.preprocessor = ImagePreprocessor(use_awb=use_awb, use_blur=use_blur)
        # Null Object 패턴: None 대신 기본 Null 객체를 사용하여 None 체크 제거
        self.db_client = db_client if db_client is not None else NullDBClient()
        self.http_sender = http_sender if http_sender is not None else NullSender()
        self.collection_name = self.config.get('collection_name', 'reid_collection')

        # 대표 프레임 선별 셀렉터 초기화
        max_missing_frames = self.config.get('max_missing_frames', 30)
        min_bbox_size = self.config.get('min_bbox_size', 40)
        send_interval_frames = self.config.get('send_interval_frames', 0)
        self.best_shot_selector = BestShotSelector(
            max_missing_frames=max_missing_frames,
            min_bbox_size=min_bbox_size,
            send_interval_frames=send_interval_frames
        )

        # 각 컴포넌트는 start() 시점에 초기화
        self._detector: PersonDetector = None
        self._tracker: PersonTracker = None
        self._reid: ReIDExtractor = None

    # ------------------------------------------------------------------
    # 생명주기
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """파이프라인을 시작하고 모든 모델을 초기화합니다."""
        self.running = True
        self.frames_processed = 0
        logger.info('Pipeline starting...')
        self._initialize_models()
        logger.info('Pipeline started.')
        return True

    def stop(self) -> bool:
        """파이프라인을 종료합니다."""
        self.running = False
        logger.info('Pipeline stopped.')

        # 분석 정지 시 캐시에 남은 트랙들의 베스트 샷을 강제 전송(Flush)
        if hasattr(self, 'best_shot_selector'):
            remaining_vectors = self.best_shot_selector.get_remaining_and_flush()
            if remaining_vectors:
                last_cam = getattr(self, '_last_camera_id', 'cam_0')
                logger.info(f"Flushing {len(remaining_vectors)} remaining tracks for camera {last_cam}")
                self._save_to_db(remaining_vectors, last_cam)
                self._send_to_server(remaining_vectors, last_cam)

        return True

    def _initialize_models(self):
        """detector, tracker, reid_extractor를 순서대로 초기화합니다."""
        try:
            device = self.config.get('device', None)

            self._detector = PersonDetector(
                model_path=self.config.get('yolo_model', 'yolov8n.pt'),
                conf_threshold=self.config.get('conf_threshold', 0.5),
                use_tensorrt=self.config.get('use_tensorrt', False),
                iou=self.config.get('yolo_iou', 0.7),
                device=device,
            )
            logger.info(f"Detector initialized (model={self.config.get('yolo_model', 'yolov8n.pt')})")

            self._tracker = PersonTracker(
                tracker_type=self.config.get('tracker_type', 'botsort'),
                reid_weights=self.config.get('reid_weights', 'osnet_x0_25_msmt17.pt'),
                device=device,
            )
            logger.info(f"Tracker initialized (type={self.config.get('tracker_type', 'botsort')})")

            self._reid = ReIDExtractor(
                model_name=self.config.get('reid_model', 'osnet_x0_25'),
                use_onnx=self.config.get('use_onnx', False),
                device=device,
            )
            logger.info(f"ReID Extractor initialized (model={self.config.get('reid_model', 'osnet_x0_25')})")

        except Exception as e:
            logger.error(f'Model initialization failed: {e}')
            raise

    # ------------------------------------------------------------------
    # 프레임 처리 (오케스트레이션)
    # ------------------------------------------------------------------

    def process_frame(self, frame, camera_id: str = "cam_0") -> dict:
        """단일 프레임의 전체 파이프라인을 실행합니다.

        흐름: 전처리 → 탐지 → 추적 → Re-ID 추출 → DB 저장

        Args:
            frame: BGR 형식의 numpy 배열.
            camera_id: 카메라 식별자 (DB 페이로드에 기록됨).

        Returns:
            {
                'frame_index': int,
                'camera_id': str,
                'detections': List[dict],
                'tracks': List[dict],
                'reid_vectors': List[dict],
                'processing_time_ms': float,
            }
        """
        if not self.running or frame is None:
            return None

        start_time = time.time()

        self._last_camera_id = camera_id

        # 1. 전처리
        processed_frame = self.preprocessor.process(frame)

        # 2. 탐지 (PersonDetector)
        det_results = self._detector.detect(processed_frame)
        dets_np = self._detector.to_numpy(det_results)   # (N, 6)

        # 3. 추적 (PersonTracker)
        track_results = self._tracker.update(dets_np, processed_frame)

        # 4. Re-ID 특징 추출 (ReIDExtractor)
        reid_vectors = self._reid.extract(processed_frame, track_results)

        # 5. 대표 프레임 (Best-shot) 필터링 및 소멸 궤적 데이터 획득
        expired_vectors = self.best_shot_selector.update(reid_vectors, self.frames_processed)

        # 6. 벡터 DB 저장 (소멸 확정된 대표 프레임만 저장)
        self._save_to_db(expired_vectors, camera_id)

        # 7. 서버 전송 (Phase 3: ServerSender 주입 시 실제 동작, 미주입 시 NullSender)
        self._send_to_server(expired_vectors, camera_id)

        self.frames_processed += 1

        return {
            'frame_index': self.frames_processed,
            'camera_id': camera_id,
            'detections': [d.to_dict() for d in det_results],
            'tracks': [t.to_dict() for t in track_results],
            'reid_vectors': reid_vectors,
            'processing_time_ms': (time.time() - start_time) * 1000,
        }

    def process_batch(self, frames_dict: dict) -> dict:
        """다중 채널 프레임을 순차 처리합니다.

        Args:
            frames_dict: {camera_id: frame} 형태의 딕셔너리.

        Returns:
            {camera_id: process_frame 결과} 딕셔너리.
        """
        return {
            channel_id: self.process_frame(frame, camera_id=channel_id)
            if frame is not None else None
            for channel_id, frame in frames_dict.items()
        }

    def _save_to_db(self, reid_vectors: list, camera_id: str):
        """Re-ID 벡터를 DB에 저장합니다.

        db_client가 NullDBClient인 경우 저장을 건너뜁니다(로그만 출력).
        """
        if self.dry_run:
            logger.debug("Dry-run mode active. Skipping Vector DB save.")
            return

        if not reid_vectors:
            return

        db_records = [
            {
                'id': hash(f"{camera_id}_{v['track_id']}_{time.time()}") % (10 ** 8),
                'vector': v['vector'],
                'payload': {
                    'camera_id': camera_id,
                    'track_id': v['track_id'],
                    'timestamp': time.time(),
                    'confidence': v['confidence'],
                },
            }
            for v in reid_vectors
        ]

        try:
            self.db_client.upsert(self.collection_name, db_records)
        except Exception as e:
            logger.warning(f'Vector DB upsert failed: {e}')

    def _send_to_server(self, reid_vectors: list, camera_id: str):
        """Re-ID 벡터를 서버로 HTTP POST 전송합니다.

        http_sender가 NullSender인 경우 전송을 건너뜁니다(로그만 출력).
        Phase 3에서 ServerSender를 주입하면 실제 전송이 동작합니다.
        """
        if self.dry_run:
            logger.debug("Dry-run mode active. Skipping server transmission.")
            return

        if not reid_vectors:
            return

        from datetime import datetime

        for v in reid_vectors:
            # 서버 Pydantic Schema(DetectionIn)에 맞게 개별 데이터 페이로드 빌드
            # timestamp는 ISO 8601 형식의 문자열로 변환하여 전송
            timestamp_str = datetime.fromtimestamp(time.time()).isoformat()
            
            payload = {
                'camera_id': camera_id,
                'tracklet_id': str(v['track_id']),
                'embedding_identity': [float(x) for x in v['vector']],
                'timestamp': timestamp_str,
                'bbox': [float(x) for x in v.get('bbox', [])],
                'event_type': 'detection',
                'is_final': v.get('is_final'),
                'thumbnail_base64': v.get('thumbnail_base64')
            }

            try:
                # 서버 endpoint를 /api/v1/security/detections 로 변경
                # 로컬 버퍼링 성공 시 202, 즉시 전송 성공 시 200이 반환되므로 허용 목록에 추가
                status_code, _ = self.http_sender.post('/api/v1/security/detections', payload)
                if status_code not in (0, 200, 201, 202):
                    logger.warning(
                        f'Server POST failed (status={status_code}): '
                        f'track_id={v["track_id"]} for camera {camera_id}'
                    )
            except Exception as e:
                logger.warning(f'Server send error for track {v["track_id"]}: {e}')

    # ------------------------------------------------------------------
    # 상태 조회
    # ------------------------------------------------------------------

    def get_intermediate_results(self) -> dict:
        """각 컴포넌트의 로드 상태를 반환합니다."""
        return {
            'frames_processed': self.frames_processed,
            'detector_loaded': self._detector is not None and self._detector.is_loaded,
            'tracker_loaded': self._tracker is not None and self._tracker.is_loaded,
            'reid_loaded': self._reid is not None and self._reid.is_loaded,
            'db_configured': type(self.db_client).__name__ != 'NullDBClient',
            'sender_configured': type(self.http_sender).__name__ != 'NullSender',
        }

    def flush(self) -> bool:
        return True


import queue
import threading
import cv2


class ThreadedPipelineRunner:
    """생산자-소비자 큐 패턴을 적용한 실시간 멀티스레드 파이프라인 러너.

    비디오 소스(IP 카메라, 웹캠, 동영상 파일) 디코딩과 딥러닝 추론 연산을
    각각 독립된 스레드로 분리하여 수행함으로써 임베디드 디바이스(Jetson 등)의 처리 속도를 극대화합니다.
    실시간성을 보장하기 위해 큐가 포화 상태일 때 예전 프레임을 강제 Drop하는 최신 프레임 보존 전략을 제공합니다.
    """

    def __init__(self, source=0, config: dict = None, db_client=None, http_sender=None, queue_size: int = 15):
        """
        Args:
            source: 비디오 파일 경로(str), RTSP 스트림 주소(str), 또는 웹캠 인덱스(int)
            config: PipelineRunner와 동일한 가중치 및 탐지 임계값 설정 딕셔너리
            db_client: VectorDBClient 인스턴스 (NullDBClient 기본값)
            http_sender: ServerSender/ResilientServerSender 인스턴스 (NullSender 기본값)
            queue_size: 프레임 버퍼 큐의 최대 크기 (메모리 제어용)
        """
        self.source = source
        self.queue_size = queue_size
        self.runner = PipelineRunner(config=config, db_client=db_client, http_sender=http_sender)

        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.result_queue = queue.Queue()

        self.running = False
        self.cap = None
        self.reader_thread: threading.Thread = None
        self.worker_thread: threading.Thread = None

    def start(self) -> bool:
        """비디오 리더 및 추론 워커 스레드를 작동시킵니다."""
        if self.running:
            logger.warning("ThreadedPipelineRunner is already running.")
            return False

        logger.info(f"Opening video source: {self.source}...")
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            return False

        # 내장 오케스트레이터 모델 시작
        self.runner.start()

        self.running = True
        self.reader_thread = threading.Thread(target=self._reader_loop, name="VideoReaderThread", daemon=True)
        self.worker_thread = threading.Thread(target=self._worker_loop, name="PipelineWorkerThread", daemon=True)

        self.reader_thread.start()
        self.worker_thread.start()
        logger.info("ThreadedPipelineRunner successfully started with multi-threading backend.")
        return True

    def stop(self) -> bool:
        """모든 비동기 스레드를 안전하게 중단하고 비디오 리소스를 해제합니다."""
        if not self.running:
            return False

        self.running = False
        logger.info("ThreadedPipelineRunner stopping threads...")

        # 스레드 종료 대기
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=2.0)
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)

        # OpenCV 및 러너 리소스 해제
        if self.cap:
            self.cap.release()
            self.cap = None

        self.runner.stop()
        logger.info("ThreadedPipelineRunner successfully stopped.")
        return True

    def _reader_loop(self):
        """카메라나 비디오 입력 스트림으로부터 쉬지 않고 프레임을 디코딩하여 큐에 적재하는 스레드 루프."""
        camera_id = f"cam_{self.source}" if isinstance(self.source, int) else "cam_file"
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logger.info("Video source reached End-Of-File (EOF) or connection lost.")
                # 비디오 파일 분석의 경우 루프 종료, 실시간 스트림일 경우 재연결 대기(여기선 일단 종료 처리)
                self.running = False
                break

            # 실시간성 보장을 위한 Frame Drop 전략:
            # 큐가 꽉 차 있다면 가장 오래된(가장 앞의) 프레임을 드랍(제거)하여 딜레이 누적 방지
            if self.frame_queue.full():
                try:
                    _ = self.frame_queue.get_nowait()
                    logger.debug("Frame queue is saturated. Squeezing out oldest frame to keep real-time sync.")
                except queue.Empty:
                    pass

            try:
                self.frame_queue.put((frame, camera_id, time.time()), timeout=0.1)
            except queue.Full:
                # 매우 드문 타임아웃 상황 처리
                pass

    def _worker_loop(self):
        """프레임 큐에서 데이터를 가져와 순차적으로 탐지, 추적 및 특징 추출 연산을 수행하는 스레드 루프."""
        while self.running or not self.frame_queue.empty():
            try:
                # 큐가 비어 있을 때 CPU를 과도하게 사용하지 않도록 짧은 타임아웃 지정
                item = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            frame, camera_id, enqueue_time = item
            
            # 큐 대기 중의 레이턴시(딜레이) 모니터링용
            queue_delay_ms = (time.time() - enqueue_time) * 1000
            
            try:
                # 실제 코어 파이프라인 동기 처리
                report = self.runner.process_frame(frame, camera_id=camera_id)
                if report:
                    report['queue_delay_ms'] = queue_delay_ms
                    # 결과 큐에 담아 외부 소비 가능하게 전달
                    self.result_queue.put(report)
            except Exception as e:
                logger.error(f"Error executing core pipeline inside worker thread: {e}")
            finally:
                self.frame_queue.task_done()

    def get_next_result(self, timeout: float = None) -> dict:
        """최종 처리 완료된 다음 인물 Re-ID 분석 결과를 결과 큐에서 읽어옵니다.

        Args:
            timeout: 결과 획득 대기 타임아웃 (초 단위)

        Returns:
            분석 완료 결과 리포트 딕셔너리. 없을 시 None 반환.
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_queue_status(self) -> dict:
        """현재 멀티스레드 큐들의 잔여 버퍼 상황을 모니터링용으로 반환합니다."""
        return {
            'frame_queue_size': self.frame_queue.qsize(),
            'result_queue_size': self.result_queue.qsize(),
            'frames_processed': self.runner.frames_processed
        }


class MultiStreamPipelineRunner:
    """다중 RTSP/카메라 스트림을 단일 GPU 자원(모델 공유)으로 동시 처리하는 멀티스트림 파이프라인 러너.

    메모리(VRAM)가 제한적인 Jetson 환경을 고려하여, 딥러닝 추론 모델(YOLO, OSNet)을
    단 하나만 인스턴스화하고 여러 카메라 리더 스레드가 공용 프레임 큐를 통해 GPU 추론을 공유하도록 설계되었습니다.
    """

    def __init__(self, sources: dict, config: dict = None, db_client=None, http_sender=None, queue_size: int = 15):
        """
        Args:
            sources: {camera_id: source_path_or_index} 형태의 딕셔너리
            config: PipelineRunner 설정 딕셔너리
            db_client: VectorDBClient 인스턴스
            http_sender: ServerSender 인스턴스
            queue_size: 공용 프레임 버퍼의 최대 크기
        """
        self.sources = sources
        self.queue_size = queue_size
        # 공유 추론 오케스트레이터 생성
        self.runner = PipelineRunner(config=config, db_client=db_client, http_sender=http_sender)

        self.shared_frame_queue = queue.Queue(maxsize=queue_size)
        self.shared_result_queue = queue.Queue()

        self.running = False
        self.caps = {}
        self.reader_threads = []
        self.worker_thread = None

    def start(self) -> bool:
        """모든 카메라 소스를 열고, 리더 스레드들과 단일 공유 워커 스레드를 기동합니다."""
        if self.running:
            logger.warning("MultiStreamPipelineRunner is already running.")
            return False

        logger.info("Initializing multi-stream camera sources...")
        for camera_id, src in self.sources.items():
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                logger.error(f"Failed to open video source for {camera_id}: {src}")
                # 열린 캡처 객체 정리 후 실패 반환
                self._release_caps()
                return False
            self.caps[camera_id] = cap

        # 단일 공유 모델 오케스트레이터 시작
        self.runner.start()

        self.running = True
        self.reader_threads = []

        # 각 카메라 채널마다 전용 디코딩 리더 스레드 기동
        for camera_id in self.sources.keys():
            t = threading.Thread(
                target=self._reader_loop,
                args=(camera_id,),
                name=f"ReaderThread_{camera_id}",
                daemon=True
            )
            self.reader_threads.append(t)
            t.start()

        # GPU 추론 및 가속을 공유하여 도맡아 처리할 단일 워커 스레드 기동
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            name="SharedPipelineWorkerThread",
            daemon=True
        )
        self.worker_thread.start()

        logger.info(f"MultiStreamPipelineRunner successfully started with {len(self.sources)} streams.")
        return True

    def stop(self) -> bool:
        """모든 스레드를 안전하게 중지하고 모든 비디오 자원을 해제합니다."""
        if not self.running:
            return False

        self.running = False
        logger.info("MultiStreamPipelineRunner stopping all threads...")

        # 스레드 조인 대기
        for t in self.reader_threads:
            if t.is_alive():
                t.join(timeout=1.0)
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)

        self._release_caps()
        self.runner.stop()
        logger.info("MultiStreamPipelineRunner successfully stopped.")
        return True

    def _release_caps(self):
        """모든 카메라 캡처 리소스를 해제합니다."""
        for camera_id, cap in list(self.caps.items()):
            try:
                cap.release()
            except Exception as e:
                logger.warning(f"Error releasing cap for {camera_id}: {e}")
        self.caps.clear()

    def _reader_loop(self, camera_id: str):
        """특정 카메라 스트림으로부터 프레임을 실시간 디코딩하여 공유 큐에 적재하는 루프."""
        cap = self.caps.get(camera_id)
        if not cap:
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Stream lost or EOF reached for camera: {camera_id}")
                # EOF 도달 또는 스트림 유실 시 루프 종료
                break

            # 실시간성 보장을 위해 큐 포화 시 가장 오래된 프레임 제거 (공용 큐 제어)
            if self.shared_frame_queue.full():
                try:
                    _ = self.shared_frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.shared_frame_queue.put((frame, camera_id, time.time()), timeout=0.1)
            except queue.Full:
                pass

    def _worker_loop(self):
        """공유 큐에서 프레임을 가져와 단일 GPU 자원을 공유하여 순차적으로 딥러닝 추론을 수행하는 루프."""
        while self.running or not self.shared_frame_queue.empty():
            try:
                item = self.shared_frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            frame, camera_id, enqueue_time = item
            queue_delay_ms = (time.time() - enqueue_time) * 1000

            try:
                # 공용 딥러닝 파이프라인 실행
                report = self.runner.process_frame(frame, camera_id=camera_id)
                if report:
                    report['queue_delay_ms'] = queue_delay_ms
                    self.shared_result_queue.put(report)
            except Exception as e:
                logger.error(f"Error in shared GPU inference worker: {e}")
            finally:
                self.shared_frame_queue.task_done()

    def get_next_result(self, timeout: float = None) -> dict:
        """최종 처리 완료된 결과를 결과 큐에서 하나 가져옵니다."""
        try:
            return self.shared_result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_status(self) -> dict:
        """현재 멀티스트림 파이프라인의 큐 크기 및 스트림별 상태를 모니터링하여 반환합니다."""
        return {
            'shared_frame_queue_size': self.shared_frame_queue.qsize(),
            'shared_result_queue_size': self.shared_result_queue.qsize(),
            'frames_processed': self.runner.frames_processed,
            'active_streams': list(self.caps.keys()),
            'running': self.running
        }



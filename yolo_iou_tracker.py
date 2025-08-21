import cv2
import numpy as np
from ultralytics import YOLO
import json
from datetime import datetime
import os
import time
import argparse
from collections import deque
import yaml


# 상수 정의
DEFAULT_FPS = 30
DEFAULT_KEEP_DURATION_SEC = 2.0
DEFAULT_IOU_THRESHOLD = 0.3
DEFAULT_MIN_CONFIDENCE = 0.25
DEFAULT_HISTORY_SIZE = 20
DEFAULT_TEXT_SCALE = 0.7
DEFAULT_TEXT_THICKNESS = 2
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")
PROGRESS_LOG_INTERVAL = 100


class IOUTracker:
    def __init__(
        self,
        max_disappeared=None,
        keep_duration_sec=None,
        estimated_fps=DEFAULT_FPS,
        iou_threshold=DEFAULT_IOU_THRESHOLD,
        min_confidence=DEFAULT_MIN_CONFIDENCE,
        history_size=DEFAULT_HISTORY_SIZE,
        tracking_classes=None,  # 추적할 클래스 목록
    ):
        """
        객체 트래킹을 위한 IOUTracker 클래스

        매개변수:
        - max_disappeared: detection이 매칭되지 않을 경우 ID를 유지할 최대 프레임 수
        - keep_duration_sec : max_disappeared을 유지하고 싶은 시간 (초)
        - estimated_fps: 예상 프레임 레이트
        - iou_threshold: 기존 트랙과 detection을 매칭할 IOU 임계값
        - min_confidence: 추적 시작에 사용할 최소 신뢰도
        - history_size: 이동 경로를 저장할 최대 위치 수
        - tracking_classes: 추적할 클래스 그룹 딕셔너리 {'그룹이름': [클래스목록]} (None이면 모든 클래스 추적)
        """

        self.estimated_fps = estimated_fps
        self.keep_duration_sec = keep_duration_sec
        # max_disappeared와 keep_duration_sec 중 하나만 제공된 경우 처리
        if max_disappeared is not None:
            self.max_disappeared = max_disappeared
        elif keep_duration_sec is not None:
            self.max_disappeared = int(self.estimated_fps * self.keep_duration_sec)
        else:
            # 둘 다 None인 경우 기본값 설정
            self.keep_duration_sec = DEFAULT_KEEP_DURATION_SEC
            self.max_disappeared = int(self.estimated_fps * self.keep_duration_sec)

        self.iou_threshold = iou_threshold
        self.min_confidence = min_confidence
        self.history_size = history_size
        self.tracking_classes = tracking_classes  # 추적할 클래스 그룹 목록
        self.tracks = {}
        # {track_id: {'bbox': [x1,y1,x2,y2], 'age': n, 'history': deque([...])}}

        # 클래스 그룹별 ID 카운터 초기화
        self.next_id = {}
        if tracking_classes:
            for group_name in tracking_classes:
                self.next_id[group_name] = 0
        else:
            self.next_id["default"] = 0

    def get_class_group(self, class_name):
        """클래스 이름에 해당하는 그룹 이름을 반환"""
        if not self.tracking_classes:
            return "default"

        for group_name, class_list in self.tracking_classes.items():
            if class_name in class_list:
                return group_name
        return None  # 어떤 그룹에도 속하지 않음

    def calculate_iou(self, box1, box2):
        """
        두 bbox ([x1, y1, x2, y2]) 간의 IOU 계산
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        inter_area = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def update(self, detections):
        """
        현재 프레임의 detection 리스트(detection은 dict 형식)를 받아 기존 트랙과 IOU 매칭 수행.
        매칭된 detection은 기존 track id를 부여하고(신뢰도가 낮은 detection은 무시), 매칭되지 않은 detection은 새 트랙으로 생성.
        또한 매칭되지 않은 기존 트랙의 age를 증가시키고, max_disappeared를 초과하면 해당 트랙을 삭제합니다.
        매개변수:
        detections: 감지된 객체 리스트 (각 항목은 {'bbox': [x1,y1,x2,y2], 'class': idx, ...} 형태)

        반환:
        track_id가 할당된 detections
        """
        # 필터링: 충분한 신뢰도를 가진 감지만 처리 + 추적 대상 클래스만 선택
        filtered = []
        for d in detections:
            # 신뢰도 체크
            if d.get("confidence", 0) < self.min_confidence:
                continue

            # 그룹 체크
            class_name = d.get("class_name", "Unknown")
            group_name = self.get_class_group(class_name)

            # 추적 대상에 포함되지 않으면 그룹이 None
            if group_name is None:
                filtered.append(d)  # 추적 대상이 아니어도 반환 결과에는 포함
                continue

            # 그룹 정보 추가
            d["group_name"] = group_name
            filtered.append(d)

        assigned = set()  # 이미 할당된 detection의 인덱스

        # 기존 트랙과 감지 결과 매칭 (같은 그룹 내에서만 매칭)
        for track_id, track in list(self.tracks.items()):
            best_match = None
            best_iou = 0
            best_det_index = -1
            track_group = track.get("group_name", "default")

            for i, det in enumerate(filtered):
                # 같은 그룹 내에서만 매칭
                det_group = det.get("group_name")
                if det_group != track_group or i in assigned:
                    continue

                if i in assigned:
                    continue

                iou_score = self.calculate_iou(track["bbox"], det["bbox"])
                if iou_score > best_iou and iou_score > self.iou_threshold:
                    best_iou = iou_score
                    best_match = det
                    best_det_index = i

            if best_match is not None:
                # 매칭된 detection으로 트랙 업데이트
                self.tracks[track_id].update(
                    {
                        "bbox": best_match["bbox"],
                        "age": 0,
                        "class": best_match.get("class", 0),
                        "class_name": best_match.get("class_name", "Unknown"),
                        "confidence": best_match.get("confidence", 1.0),
                    }
                )

                # 중심점 계산 및 이력에 추가
                x1, y1, x2, y2 = best_match["bbox"]
                centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                self.tracks[track_id]["history"].append(centroid)

                # detection에 트랙 id 부여 (그룹 접두사 포함)
                best_match["track_id"] = track_id
                assigned.add(best_det_index)
            else:
                # 매칭되지 않으면 age 증가
                self.tracks[track_id]["age"] += 1

        # 새 detection에 대해 새로운 트랙 생성
        for i, det in enumerate(filtered):
            # 그룹이 없거나 이미 할당된 경우 건너뛰기
            group_name = det.get("group_name")
            if group_name is None or i in assigned:
                continue
            
            # 그룹별 ID 생성 (예: "E0", "P0" 등)
            group_prefix = group_name[0].upper()  # 첫 글자를 대문자로
            numeric_id = self.next_id[group_name]
            self.next_id[group_name] += 1
            track_id = f"{group_prefix}{numeric_id}"

            # 중심점 계산
            x1, y1, x2, y2 = det["bbox"]
            centroid = ((x1 + x2) / 2, (y1 + y2) / 2)

            # 새 트랙 등록
            self.tracks[track_id] = {
                "bbox": det["bbox"],
                "age": 0,
                "class": det.get("class", 0),
                "class_name": det.get("class_name", "Unknown"),
                "confidence": det.get("confidence", 1.0),
                "group_name": group_name,  # 그룹 이름 저장
                "history": deque([centroid], maxlen=self.history_size),
            }
            det["track_id"] = track_id

        # 오래된 트랙 제거
        remove_ids = [
            tid
            for tid, track in self.tracks.items()
            if track["age"] > self.max_disappeared
        ]
        for tid in remove_ids:
            del self.tracks[tid]

        return filtered


class ObjectDetectionTracker:
    def __init__(
        self,
        model_path,
        save_dir="detection_results",
        conf_threshold=DEFAULT_MIN_CONFIDENCE,
        iou_threshold=0.45,
        device="",  # 자동 선택
        max_disappeared=None,
        keep_duration_sec=None,
        draw_history=True,
        history_size=DEFAULT_HISTORY_SIZE,
        frame_interval=1,
        # 시각화
        text_scale=DEFAULT_TEXT_SCALE,
        text_thickness=DEFAULT_TEXT_THICKNESS,
        # 건설 현장 이벤트 관련 선택적 매개변수
        overlap_threshold=0.1,
        head_ratio=0.3,
        helmet_thr=0.1,
        vest_thr=0.1,
        danger_zone_ratio=0.2,
        equip_event_thr=30,
        helmet_event_thr=30,
        signal_exist_thr=0.6,
    ):
        """
        YOLO 객체 감지 및 IOUTracker ID 추적 클래스

        매개변수:
        model_path: YOLO 모델 가중치 파일 경로
        save_dir: 결과 저장 디렉토리
        conf_threshold: 객체 감지 신뢰도 임계값
        iou_threshold: 비최대 억제(NMS)를 위한 IOU 임계값
        device: 추론 장치 (빈 문자열이면 자동 선택)
        max_disappeared: 객체 ID를 유지할 최대 프레임 수
        keep_duration_sec: 객체 ID를 유지할 최대 시간(초)
        draw_history: 객체 이동 경로 시각화 여부
        history_size: 경로 시각화를 위해 저장할 최근 위치 수
        use_triton: Triton 추론 서버 사용 여부

        triton_url: Triton 추론 서버 URL
        triton_model_name: Triton 서버에 배포된 모델 이름
        frame_interval: YOLO 감지를 실행할 프레임 간격

        text_scale: 텍스트 크기 설정
        text_thickness: 텍스트 두께 설정
        """
        # YOLO 모델 초기화
        self._setup_device(device)
        self.model = YOLO(model_path)

        # 설정 저장
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.draw_history = draw_history
        self.frame_interval = frame_interval

        # max_disappeared와 keep_duration_sec 설정 저장
        self.max_disappeared = max_disappeared
        self.keep_duration_sec = keep_duration_sec

        # 저장 디렉토리 생성
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 텍스트/시각화 설정
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        self.draw_history = draw_history
        # Triton 설정

        # 클래스 목록 등 (건설 현장 특화)
        self.equipment_classes = [
            "Forklift",
            "Excavator",
            "DumpTruck",
            "MobileCrane",
            "Crane",
        ]
        self.ko_equipment_classes = [
            "지게차",
            "굴착기",
            "덤프트럭",
            "이동식 크레인",
            "크레인",
        ]
        self.helmet_classes = [
            "Helmet_BU",
            "Helmet_OR",
            "Helmet_RD",
            "Helmet_WT",
            "Helmet_YE",
            "Helmet_ETC",
            "Helmet_GR",
        ]
        self.vest_classes = [
            "SafetyVest_RD",
            "SafetyVest_OR",
            "SafetyVest_YE",
            "SafetyVest_ETC",
        ]

        # 기본 클래스 이름
        self.names = [
            "Forklift",
            "Excavator",
            "DumpTruck",
            "MobileCrane",
            "Crane",
            "Person",
            "LightWand",
            "Helmet_RD",
            "Helmet_OR",
            "Helmet_WT",
            "Helmet_YE",
            "Helmet_BU",
            "Helmet_GR",
            "Helmet_ETC",
            "SafetyVest_RD",
            "SafetyVest_OR",
            "SafetyVest_YE",
            "SafetyVest_ETC",
        ]

        # 색상 설정 (통합된 색상 매핑)
        self.colors = {
            "default": (0, 255, 0),  # 녹색
            "person": (0, 165, 255),  # 주황색
            "vehicle": (255, 0, 0),  # 파란색
            "equipment": (0, 255, 255),  # 노란색
            "helmet": (255, 0, 255),  # 자홍색
            "vest": (255, 255, 0),  # 청록색
            "danger": (0, 0, 255),  # 빨간색
            "warning": (0, 255, 255),  # 노란색
            "track": (0, 0, 255),  # 빨간색
            "text": (255, 255, 255),  # 흰색
        }

        # 추적할 클래스 그룹 정의 (장비와 사람을 별도 그룹으로)
        tracking_classes = {
            "Equipment": self.equipment_classes,  # E로 시작하는 ID 부여
            "Person": ["Person"],  # P로 시작하는 ID 부여
        }
        # IOUTracker 초기화
        self.tracker = IOUTracker(
            max_disappeared=max_disappeared,
            keep_duration_sec=keep_duration_sec,
            iou_threshold=DEFAULT_IOU_THRESHOLD,
            min_confidence=conf_threshold,
            history_size=history_size,
            tracking_classes=tracking_classes,  # 추적할 클래스 그룹 전달
        )

        # 통계 정보
        self.processing_times = []
        self.frame_count = 0

        # 건설 현장 이벤트 관련 매개변수 (추후 이벤트 분석에 활용)
        self.overlap_threshold = overlap_threshold
        self.head_ratio = head_ratio
        self.helmet_thr = helmet_thr
        self.vest_thr = vest_thr
        self.danger_zone_ratio = danger_zone_ratio
        self.equip_event_thr = equip_event_thr
        self.helmet_event_thr = helmet_event_thr
        self.signal_exist_thr = signal_exist_thr
    
    def _setup_device(self, device):
        """디바이스 설정"""
        if device and device.strip():
            try:
                gpu_ids = [int(x.strip()) for x in device.split(",") if x.strip()]
                if gpu_ids:
                    self.device = f"cuda:{gpu_ids[0]}"
                else:
                    self.device = ""
            except ValueError:
                self.device = device
        else:
            self.device = ""
        print(f"Using device: {self.device}")

    def detect_objects(self, frame):  # yolo_inference
        """
        YOLO 모델을 사용하여 GPU에서 객체 감지 수행

        매개변수:
        frame: 입력 프레임 (numpy 배열)

        반환:
        감지된 객체 리스트

        YOLO 객체 감지를 수행하여 detection 결과 (bbox, class, confidence 등)를 반환.
        각 detection은 bbox, class, confidence, timestamp 등을 포함.
        bbox 좌표 형식은 [x1, y1, x2, y2].
        """
        results = self.model.predict(
            frame, conf=self.conf_threshold, iou=self.iou_threshold, device=self.device
        )
        # 모델의 클래스 이름 가져오기 (가능하면)
        names = self.model.names if hasattr(self.model, "names") else self.names
        detections = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # 결과 처리
        for box in results[0].boxes:
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            cls_id = (
                int(box.cls[0])
                if isinstance(box.cls, (list, np.ndarray))
                else int(box.cls)
            )
            conf = (
                float(box.conf[0])
                if isinstance(box.conf, (list, np.ndarray))
                else float(box.conf)
            )
            detections.append(
                {
                    "class": cls_id,
                    "class_name": (
                        names[cls_id] if cls_id < len(names) else f"Class_{cls_id}"
                    ),
                    "confidence": conf,
                    "bbox": bbox,
                    "timestamp": timestamp,
                }
            )

        # 결과 저장
        filename = f"{self.save_dir}/detection_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(detections, f, indent=4)
        return detections


    def get_color_for_class(self, class_name):
        """
        클래스 이름에 따라 색상을 반환합니다.
        """
        class_lower = class_name.lower()

        if "person" in class_lower:
            return self.colors["person"]
        elif any(
            vehicle in class_lower for vehicle in ["car", "truck", "bus", "dumptruck"]
        ):
            return self.colors["vehicle"]
        elif any(
            equip in class_lower
            for equip in ["forklift", "excavator", "crane", "mobilecrane"]
        ):
            return self.colors["equipment"]
        elif "helmet" in class_lower:
            return self.colors["helmet"]
        elif "vest" in class_lower or "safetyvest" in class_lower:
            return self.colors["vest"]
        else:
            return self.colors["default"]

    def draw_detection(self, frame, detection):
        """
        감지된 객체를 프레임에 그립니다.

        매개변수:
        frame: 그림을 그릴 프레임
        detection: 감지된 객체 정보
        """
        bbox = detection["bbox"]
        class_name = detection.get("class_name", "Unknown")
        conf = detection.get("confidence", 0)
        track_id = detection.get("track_id", None)

        # 객체 유형에 따른 색상 선택
        color = self.get_color_for_class(class_name)

        # 바운딩 박스 그리기
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 클래스 이름, 신뢰도 텍스트 생성
        text = f"{class_name}: {conf:.2f}"

        # track_id가 있는 클래스에만 ID 표시
        if track_id is not None:
            text += f" ID:{track_id}"

        # 텍스트 배경 그리기
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )
        cv2.rectangle(
            frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1
        )

        # 텍스트 그리기
        cv2.putText(
            frame,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.text_scale,
            self.colors["text"],
            self.text_thickness,
        )

        # 객체의 트랙 ID가 있으면 이동 경로 그리기
        if (
            self.draw_history
            and track_id is not None
            and track_id in self.tracker.tracks
        ):
            history = list(self.tracker.tracks[track_id]["history"])
            for i in range(1, len(history)):
                # 연속된 점을 연결하는 선 그리기
                pt1 = tuple(map(int, history[i - 1]))
                pt2 = tuple(map(int, history[i]))
                cv2.line(frame, pt1, pt2, self.colors["track"], 2)

    def process_video(
        self,
        video_source,
        output_path=None,
        detection_interval=1,
        display=True,
        fps_limit=None,
    ):
        """
        비디오를 처리하고 객체 감지 및 추적 수행 / 결과 시각화

        매개변수:
        video_source: 비디오 파일 경로 또는 카메라 인덱스
        output_path: 결과 영상을 저장할 경로 (없으면 기본 경로에 output폴더 생성)
        detection_interval: 몇 프레임마다 감지할지 결정 (기본 1)
        display: 화면에 결과 표시 여부
                fps_limit: 출력 FPS 제한 (None이면 제한 없음)
        """
        # 비디오 파일, 카메라 또는 스트림 열기
        if isinstance(video_source, int) or (
            isinstance(video_source, str) and video_source.isdigit()
        ):
            cap = cv2.VideoCapture(int(video_source))
            print(f"카메라 {video_source} 열기...")
        else:
            cap = cv2.VideoCapture(video_source)
            if video_source.startswith(('rtsp://', 'http://', 'https://')):
                print(f"스트림 '{video_source}' 열기...")
                # 스트림에 대한 버퍼 크기 설정
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                print(f"비디오 파일 '{video_source}' 열기...")

        if not cap.isOpened():
            print(f"Error: 비디오 소스 '{video_source}'를 열 수 없습니다.")
            print("해결 방법:")
            print("- 파일 경로가 올바른지 확인")
            print("- 카메라가 연결되어 있는지 확인")
            print("- RTSP/HTTP URL이 올바른지 확인")
            print("- FFmpeg가 설치되어 있는지 확인 (sudo apt-get install ffmpeg)")
            return

        # 비디오 속성 가져오기
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:  # 카메라 스트림의 경우
            total_frames = float("inf")

        # 출력 비디오 설정
        out = self._setup_video_writer(output_path, video_source, fps, frame_width, frame_height)

        print(f"비디오 처리 시작... (해상도: {frame_width}x{frame_height}, FPS: {fps:.1f})")
        self.frame_count = 0
        self.processing_times = []
        # FPS 제한을 위한 설정
        frame_time = 1.0 / fps if fps_limit is None else 1.0 / fps_limit
        
        # 첫 프레임 처리 시간 측정을 위한 변수
        first_frame_processed = False

        try:
            while cap.isOpened() and (self.frame_count < total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1
                frame_start_time = time.time()

                # detection_interval에 따라 객체 감지 수행
                detections = []
                if self.frame_count % self.frame_interval == 0:
                    detections = self.detect_objects(frame)

                # IOUTracker 업데이트 (감지 결과가 없으면 기존 트랙은 age 증가)
                tracked_detections = self.tracker.update(detections)

                # 결과 시각화
                for det in tracked_detections:
                    self.draw_detection(frame, det)

                # 프레임 정보 및 성능 지표 표시
                frame_end_time = time.time()
                processing_time = frame_end_time - frame_start_time
                self.processing_times.append(processing_time)

                # 프레임 정보 표시
                self._draw_frame_info(frame, total_frames)
                
                # 첫 프레임 처리 완료 시 알림
                if not first_frame_processed:
                    print(f"첫 프레임 처리 완료. 추론 시간: {processing_time:.3f}초")
                    first_frame_processed = True

                # 결과 화면에 표시
                if display:
                    cv2.imshow("YOLO + IOUTracker", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord("q"):  # ESC 또는 q 키로 종료
                        break

                # 비디오 파일에 저장
                if out is not None:
                    out.write(frame)

                # 진행 상황 출력
                if self.frame_count % PROGRESS_LOG_INTERVAL == 0 and total_frames != float("inf"):
                    progress = self.frame_count / total_frames * 100
                    print(
                        f"처리 중... {self.frame_count}/{total_frames} ({progress:.1f}%)"
                    )

                # FPS 제한
                if fps_limit is not None:
                    elapsed = time.time() - frame_start_time
                    sleep_time = max(0, frame_time - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("사용자에 의해 중단됨")

        finally:
            # 리소스 정리
            avg_fps = (
                1.0 / np.mean(self.processing_times) if self.processing_times else 0
            )
            print(f"\n비디오 처리 완료:")
            print(f"- 처리된 프레임: {self.frame_count}")
            print(f"- 평균 FPS: {avg_fps:.1f}")
            print(f"- 총 처리 시간: {sum(self.processing_times):.1f}초")
            if out:
                print(f"- 출력 파일: {output_path}")

            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
    
    def _setup_video_writer(self, output_path, video_source, fps, frame_width, frame_height):
        """비디오 라이터 설정"""
        if output_path is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "output"
            )
            os.makedirs(output_dir, exist_ok=True)

            if isinstance(video_source, str):
                base_name = os.path.basename(video_source)
                name, ext = os.path.splitext(base_name)
                output_path = os.path.join(output_dir, f"{name}_inference{ext}")
            else:
                output_path = os.path.join(output_dir, "camera_inference.mp4")
        
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            print(f"출력 비디오 '{output_path}' 설정...")
            return out
        return None
    
    def _draw_frame_info(self, frame, total_frames):
        """프레임 정보 표시"""
        avg_time = np.mean(self.processing_times[-30:]) if self.processing_times else 0
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        
        info_texts = [
            f"Frame: {self.frame_count}/{total_frames if total_frames != float('inf') else 'Live'}",
            f"FPS: {avg_fps:.1f}",
            f"Objects: {len(self.tracker.tracks)}"
        ]
        
        for i, text in enumerate(info_texts):
            cv2.putText(frame, text, (10, 30 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


    def process_folder(
        self,
        folder_path,
        output_folder=None,
        detection_interval=1,
        display=True,
        fps_limit=None,
        video_extensions=VIDEO_EXTENSIONS,
    ):
        """
        폴더 내의 모든 비디오 파일을 처리합니다.

        매개변수:
        folder_path: 비디오 파일이 있는 폴더 경로
        output_folder: 결과 비디오를 저장할 폴더 경로 (없으면 저장하지 않음)
        detection_interval: 몇 프레임마다 감지할지 결정 (기본 1)
        display: 화면에 결과 표시 여부
        fps_limit: 출력 FPS 제한 (None이면 제한 없음)
        video_extensions: 처리할 비디오 파일 확장자 튜플
        """
        # 폴더가 존재하는지 확인
        if not os.path.isdir(folder_path):
            print(f"Error: '{folder_path}'는 유효한 폴더가 아닙니다.")
            return

        # 출력 폴더 생성 (필요한 경우)
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)

        # 폴더 내 모든 파일 리스트
        all_files = os.listdir(folder_path)
        video_files = [f for f in all_files if f.lower().endswith(video_extensions)]

        if not video_files:
            print(f"'{folder_path}' 폴더에 비디오 파일이 없습니다.")
            return

        print(f"총 {len(video_files)}개의 비디오 파일을 처리합니다...")

        # 각 비디오 파일 처리
        for i, video_file in enumerate(video_files):
            video_path = os.path.join(folder_path, video_file)
            print(f"\n[{i+1}/{len(video_files)}] '{video_file}' 처리 중...")

            # 출력 파일 경로 설정
            output_path = None
            if output_folder:
                base_name = os.path.splitext(video_file)[0]
                output_path = os.path.join(output_folder, f"{base_name}_processed.mp4")

            # 비디오 처리
            try:
                self.reset_tracker()  # 새 영상 처리 전 트래커 초기화
                self.process_video(
                    video_source=video_path,
                    output_path=output_path,
                    detection_interval=detection_interval,
                    display=display,
                    fps_limit=fps_limit,
                )
            except Exception as e:
                print(f"'{video_file}' 처리 중 오류 발생: {e}")

        print(f"\n모든 비디오 처리 완료. 총 {len(video_files)}개 파일 처리됨.")


    def reset_tracker(self):
        """
        객체 추적기를 초기화하여 새 영상 처리 시 이전 영상의 추적 정보를 삭제합니다.
        """
        tracking_classes = {
            "Equipment": self.equipment_classes,
            "Person": ["Person"],
        }
        self.tracker = IOUTracker(
            max_disappeared=self.max_disappeared,
            keep_duration_sec=self.keep_duration_sec,
            iou_threshold=DEFAULT_IOU_THRESHOLD,
            min_confidence=self.conf_threshold,
            history_size=self.tracker.history_size,
            tracking_classes=tracking_classes,
        )


def load_config(config_path):
    """설정 파일 로드"""
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"설정 파일 읽기 오류: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description="YOLO 객체 감지 및 IOUTracker ID 추적")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="설정 파일 경로"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="비디오 파일/폴더 경로 또는 카메라 인덱스",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="YOLO 모델 가중치 파일 경로",
    )
    parser.add_argument(
        "--output_path", type=str, default=None, help="출력 비디오 파일 경로 (선택적)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="객체 감지 신뢰도 임계값"
    )
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IOU 임계값")
    parser.add_argument(
        "--device", type=str, default="", help="추론 장치 (예: cuda:0, cpu)"
    )
    parser.add_argument(
        "--max-disappeared",
        type=int,
        default=None,
        help="객체 ID를 유지할 최대 프레임 수",
    )
    parser.add_argument(
        "--keep-duration-sec",
        type=float,
        default=None,
        help="객체 ID를 유지할 최대 시간(초)",
    )
    parser.add_argument(
        "--save-dir", type=str, default="detection_results", help="결과 저장 디렉토리"
    )
    parser.add_argument("--no-display", action="store_true", help="결과 표시 비활성화")
    parser.add_argument(
        "--no-history", action="store_true", help="객체 이동 경로 그리기 비활성화"
    )
    parser.add_argument(
        "--history-size", type=int, default=20, help="경로 시각화를 위한 이력 크기"
    )
    parser.add_argument(
        "--fps-limit", type=int, default=None, help="출력 FPS 제한 (선택적)"
    )
    parser.add_argument(
        "--frame-interval", type=int, default=1, help="YOLO 감지를 실행할 프레임 간격"
    )

    args = parser.parse_args()
    
    # 설정 파일 로드 및 기본값 업데이트
    config = load_config(args.config)
    if config:
        print(f"설정 파일 '{args.config}' 로드 성공")
        # 설정 파일의 값으로 기본값 업데이트 (명령행 인자가 우선)
        if 'model' in config and 'path' in config['model']:
            if not hasattr(args, 'model') or not args.model:
                args.model = config['model']['path']
        if 'detection' in config:
            if args.conf == 0.25:
                args.conf = config['detection'].get('confidence_threshold', 0.25)
            if args.iou == 0.45:
                args.iou = config['detection'].get('iou_threshold', 0.45)

    # 비디오 소스 처리
    source = args.source
    if source.isdigit():
        source = int(source)
    
    # 설정 정보 출력
    print_config_info(args, source)

    # ObjectDetectionTracker 객체 생성
    tracker = ObjectDetectionTracker(
        model_path=args.model,
        save_dir=args.save_dir,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        max_disappeared=args.max_disappeared,
        keep_duration_sec=args.keep_duration_sec,
        draw_history=not args.no_history,
        history_size=args.history_size,
        frame_interval=args.frame_interval,
    )

    # 폴더 또는 단일 비디오 처리
    process_source(tracker, source, args)


def print_config_info(args, source):
    """설정 정보 출력"""
    print(f"설정 정보: 소스={source}, 모델={args.model}, 출력={args.output_path}")
    if args.max_disappeared is not None:
        print(f"객체 ID 유지할 최대 프레임 수: {args.max_disappeared}")
    if args.keep_duration_sec is not None:
        print(f"객체 ID 유지할 최대 시간(초): {args.keep_duration_sec}")


def process_source(tracker, source, args):
    """소스 유형에 따른 처리"""
    if isinstance(source, str) and os.path.isdir(source):
        tracker.process_folder(
            folder_path=source,
            output_folder=args.output_path,
            display=not args.no_display,
            fps_limit=args.fps_limit,
        )
    else:
        tracker.process_video(
            video_source=source,
            output_path=args.output_path,
            display=not args.no_display,
            fps_limit=args.fps_limit,
        )


if __name__ == "__main__":
    import sys

    main()

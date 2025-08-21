# YOLO-IOUTracker

YOLO 기반 실시간 객체 감지 및 IOU 추적 시스템

## 기능

- Ultralytics YOLO 모델을 사용한 객체 감지
- IOU 기반 객체 추적 (ID 할당 및 유지)
- 클래스 그룹별 독립적 ID 체계 (Equipment: E0, E1... / Person: P0, P1...)
- 일시적 가림 시에도 ID 유지 (설정 가능한 유지 시간)
- 객체 이동 경로 시각화

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 기본 사용
```bash
python yolo_iou_tracker.py --source video.mp4 --model yolov8n.pt
```

### 웹캠
```bash
python yolo_iou_tracker.py --source 0 --model yolov8n.pt
```

### RTSP 스트림
```bash
python yolo_iou_tracker.py --source rtsp://ip:port/stream --model yolov8n.pt
```

### 폴더 일괄 처리
```bash
python yolo_iou_tracker.py --source /video/folder --model yolov8n.pt --output-path /output
```

## 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--source` | 비디오 파일, 폴더, 웹캠 인덱스, 스트림 URL | 필수 |
| `--model` | YOLO 모델 경로 | 필수 |
| `--conf` | 객체 감지 신뢰도 임계값 | 0.25 |
| `--iou` | NMS IOU 임계값 | 0.45 |
| `--keep-duration-sec` | 객체 ID 유지 시간(초) | 2.0 |
| `--max-disappeared` | ID 유지 최대 프레임 수 | None |
| `--device` | 추론 장치 (cuda:0, cpu) | auto |
| `--frame-interval` | 감지 프레임 간격 | 1 |
| `--output-path` | 결과 저장 경로 | None |
| `--save-dir` | 감지 결과 저장 디렉토리 | detection_results |
| `--no-display` | 화면 표시 비활성화 | False |
| `--no-history` | 이동 경로 표시 비활성화 | False |
| `--history-size` | 이동 경로 저장 크기 | 20 |
| `--fps-limit` | 출력 FPS 제한 | None |

## 설정 파일 (config.yaml)

```yaml
model:
  path: "yolov8n.pt"
  device: "cuda:0"
  
detection:
  confidence_threshold: 0.25
  iou_threshold: 0.45
  frame_interval: 1
  
tracking:
  iou_threshold: 0.3
  keep_duration_sec: 2.0
  history_size: 20
  
  tracking_classes:
    Equipment: [Forklift, Excavator, Crane]
    Person: [Person]
    
visualization:
  draw_history: true
  text_scale: 0.7
  text_thickness: 2
  colors:
    default: [0, 255, 0]
    person: [0, 165, 255]
    equipment: [0, 255, 255]
    track: [0, 0, 255]
    
output:
  save_dir: "detection_results"
  video_format: "mp4v"
  save_json: true
```

## 프로젝트 구조

```
├── yolo_iou_tracker.py    # 메인 실행 파일
├── test_tracker.py         # 테스트 스크립트
├── config.yaml            # 설정 파일
├── requirements.txt       # 의존성
├── detection_results/     # 결과 저장 폴더
└── run_examples.sh       # 예제 스크립트
```

## 출력

### 비디오
- 바운딩 박스와 클래스 라벨
- 추적 ID (E0, P0 등)
- 신뢰도 점수
- 이동 경로 (빨간색 선)

### JSON
```json
{
  "class": 5,
  "class_name": "Person",
  "confidence": 0.892,
  "bbox": [320.5, 180.2, 420.8, 380.6],
  "timestamp": "20240315_143052_123456",
  "track_id": "P0"
}
```


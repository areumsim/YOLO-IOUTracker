#!/bin/bash

echo ""YOLO-IOUTracker" 예제 실행 스크립트"
echo "=================================="
echo ""

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 모델 파일 확인
if [ ! -f "yolov8n.pt" ]; then
    echo -e "${YELLOW}YOLOv8n 모델 다운로드 중...${NC}"
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
fi

echo -e "${GREEN}사용 가능한 옵션:${NC}"
echo "1. 웹캠 실시간 처리"
echo "2. 비디오 파일 처리"
echo "3. RTSP 스트림 처리"
echo "4. 폴더 내 모든 비디오 처리"
echo "5. 설정 파일 사용 예제"
echo ""

read -p "옵션을 선택하세요 (1-5): " choice

case $choice in
    1)
        echo -e "${GREEN}웹캠 실시간 처리 시작...${NC}"
        python yolo_iou_tracker.py --source 0 --model yolov8n.pt
        ;;
    2)
        read -p "비디오 파일 경로를 입력하세요: " video_path
        echo -e "${GREEN}비디오 파일 처리 시작...${NC}"
        python yolo_iou_tracker.py --source "$video_path" --model yolov8n.pt
        ;;
    3)
        read -p "RTSP URL을 입력하세요 (예: rtsp://192.168.1.100:554/stream): " rtsp_url
        echo -e "${GREEN}RTSP 스트림 처리 시작...${NC}"
        python yolo_iou_tracker.py --source "$rtsp_url" --model yolov8n.pt --no-display
        ;;
    4)
        read -p "비디오 폴더 경로를 입력하세요: " folder_path
        echo -e "${GREEN}폴더 내 비디오 처리 시작...${NC}"
        python yolo_iou_tracker.py --source "$folder_path" --model yolov8n.pt
        ;;
    5)
        echo -e "${GREEN}설정 파일을 사용한 실행...${NC}"
        python yolo_iou_tracker.py --config config.yaml --source 0
        ;;
    *)
        echo -e "${RED}잘못된 선택입니다.${NC}"
        exit 1
        ;;
esac
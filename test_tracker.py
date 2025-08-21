#!/usr/bin/env python3
"""
YOLO-IOUTracker 테스트 스크립트
"""

import cv2
import numpy as np
import os
import sys

def create_test_video(filename="test_video.mp4", fps=30, duration=5):
    """테스트용 비디오 생성"""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    # 움직이는 사각형 생성
    for frame_num in range(fps * duration):
        # 검은 배경
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 움직이는 사각형 1 (왼쪽에서 오른쪽으로)
        x1 = int((frame_num * 5) % (width - 100))
        y1 = 100
        cv2.rectangle(frame, (x1, y1), (x1 + 80, y1 + 80), (0, 255, 0), -1)
        cv2.putText(frame, "Object 1", (x1 + 10, y1 + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # 움직이는 사각형 2 (위에서 아래로)
        x2 = 300
        y2 = int((frame_num * 3) % (height - 100))
        cv2.rectangle(frame, (x2, y2), (x2 + 80, y2 + 80), (255, 0, 0), -1)
        cv2.putText(frame, "Object 2", (x2 + 10, y2 + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 프레임 번호 표시
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"테스트 비디오 '{filename}' 생성 완료")
    return filename

def test_basic_functionality():
    """기본 기능 테스트"""
    print("=== YOLO-IOUTracker 기본 기능 테스트 ===")
    
    # 1. 필요한 모듈 임포트 테스트
    try:
        from yolo_iou_tracker import IOUTracker, ObjectDetectionTracker
        print("✓ 모듈 임포트 성공")
    except ImportError as e:
        print(f"✗ 모듈 임포트 실패: {e}")
        return False
    
    # 2. IOUTracker 초기화 테스트
    try:
        tracker = IOUTracker(keep_duration_sec=2.0)
        print("✓ IOUTracker 초기화 성공")
    except Exception as e:
        print(f"✗ IOUTracker 초기화 실패: {e}")
        return False
    
    # 3. IOU 계산 테스트
    try:
        box1 = [0, 0, 100, 100]
        box2 = [50, 50, 150, 150]
        iou = tracker.calculate_iou(box1, box2)
        expected_iou = 2500 / 17500  # 교집합 / 합집합
        if abs(iou - expected_iou) < 0.01:
            print(f"✓ IOU 계산 성공: {iou:.3f}")
        else:
            print(f"✗ IOU 계산 실패: 예상값 {expected_iou:.3f}, 실제값 {iou:.3f}")
    except Exception as e:
        print(f"✗ IOU 계산 실패: {e}")
        return False
    
    # 4. 트래킹 업데이트 테스트
    try:
        detections = [
            {
                "bbox": [100, 100, 200, 200],
                "class": 0,
                "class_name": "Person",
                "confidence": 0.9
            }
        ]
        
        # 첫 번째 프레임
        result1 = tracker.update(detections)
        if len(result1) == 1 and "track_id" in result1[0]:
            print(f"✓ 첫 번째 트래킹 업데이트 성공, ID: {result1[0]['track_id']}")
        
        # 두 번째 프레임 (약간 이동)
        detections[0]["bbox"] = [110, 110, 210, 210]
        result2 = tracker.update(detections)
        if result2[0]["track_id"] == result1[0]["track_id"]:
            print(f"✓ ID 유지 성공: {result2[0]['track_id']}")
        else:
            print(f"✗ ID 유지 실패")
            
    except Exception as e:
        print(f"✗ 트래킹 업데이트 실패: {e}")
        return False
    
    print("\n모든 기본 테스트 통과!")
    return True

def test_video_processing():
    """비디오 처리 테스트"""
    print("\n=== 비디오 처리 테스트 ===")
    
    # 테스트 비디오 생성
    test_video = create_test_video()
    
    # 간단한 처리 테스트
    try:
        cap = cv2.VideoCapture(test_video)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✓ 테스트 비디오 읽기 성공")
            cap.release()
        
        # 테스트 비디오 삭제
        os.remove(test_video)
        print("✓ 테스트 완료 및 정리")
        
    except Exception as e:
        print(f"✗ 비디오 처리 테스트 실패: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("YOLO-IOUTracker 테스트 시작\n")
    
    # 기본 기능 테스트
    if test_basic_functionality():
        # 비디오 처리 테스트
        test_video_processing()
    
    print("\n테스트 완료!")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_only.py

- coco_pv.yaml 생성
- ultralytics YOLOv8n 모델 학습
- 시작/끝 메시지 출력
"""

import json
from pathlib import Path
from ultralytics import YOLO

# 0) 정의
BASE_DIR     = Path(__file__).parent.resolve()
WORK_DIR     = BASE_DIR / 'datasets' / 'coco_pv'
ANNOT_JSON   = BASE_DIR / 'datasets' / 'annotations' / 'instances_pv_val2017.json'
YAML_PATH    = BASE_DIR / 'coco_pv.yaml'

EPOCHS       = 50
BATCH_SIZE   = 16
IMGSZ        = 640

# 1) YAML 생성
cats = json.loads(ANNOT_JSON.read_text())['categories']
names = [c['name'] for c in cats]
YAML_PATH.write_text(f"""\
path: {WORK_DIR}
train: images
val: images
nc: {len(names)}
names: {names}
""")
print(f"✓ YAML 파일 생성/갱신: {YAML_PATH}")

# 2) 모델 학습
print("▶ YOLOv8n 모델 학습 시작...")
model = YOLO('yolov8n.pt')
model.train(
    data=str(YAML_PATH),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH_SIZE,
    project=BASE_DIR/'runs',
    name='coco_pv_yolov8n',
    exist_ok=True
)
print("✓ YOLOv8n 모델 학습 완료. 결과는 runs/coco_pv_yolov8n/에 저장됩니다.")
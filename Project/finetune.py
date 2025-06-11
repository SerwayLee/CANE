#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetune.py

- coco_pv.yaml + datasets2/data.yaml 병합
- train.txt / val.txt 생성
- merged_data.yaml 생성
- 기존 best.pt 기반 파인튜닝
- best_final.pt 저장
- 전체 클래스(0–12) 검증 리포트 출력
- 디렉터리별 이미지 스캔 결과 디버깅
"""

import yaml, shutil
from pathlib import Path
from ultralytics import YOLO
import sys

# ── 0) 경로 정의 ───────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.resolve()
COCO_YAML    = BASE_DIR / 'coco_pv.yaml'
NEW_YAML     = BASE_DIR / 'datasets2' / 'data.yaml'
MERGED_YAML  = BASE_DIR / 'merged_data.yaml'
TRAIN_LIST   = BASE_DIR / 'train.txt'
VAL_LIST     = BASE_DIR / 'val.txt'
COCO_BEST    = BASE_DIR / 'runs' / 'coco_pv_yolov8n' / 'weights' / 'best.pt'
FINAL_MODEL  = BASE_DIR / 'runs' / 'coco_pv_yolov8n' / 'weights' / 'best_final.pt'

# ── 1) YAML 로드 및 names 병합 ─────────────────────────────────
coco_cfg = yaml.safe_load(open(COCO_YAML))
new_cfg  = yaml.safe_load(open(NEW_YAML))
merged_names = list(dict.fromkeys(coco_cfg['names'] + new_cfg['names']))
merged_nc    = len(merged_names)
print(f"DEBUG: merged_names ({merged_nc}): {merged_names}")

# ── 2) train/val 경로 추출 함수 ─────────────────────────────────
def extract_paths(cfg, key, base_yaml_path=None):
    paths = []
    v = cfg.get(key)
    if 'path' in cfg and isinstance(v, str):
        paths.append(str((Path(cfg['path']) / v).resolve()))
    elif base_yaml_path is not None:
        if 'path' in cfg and isinstance(v, str):
            paths.append(str((Path(cfg['path']) / v).resolve()))
        elif isinstance(v, list):
            for item in v:
                sub = item[3:] if item.startswith('../') else item
                paths.append(str((base_yaml_path.parent / sub).resolve()))
        elif isinstance(v, str):
            sub = v[3:] if v.startswith('../') else v
            paths.append(str((base_yaml_path.parent / sub).resolve()))
    return paths

# extract directories
coco_train = extract_paths(coco_cfg, 'train', None)
coco_val   = extract_paths(coco_cfg, 'val',   None)
new_train  = extract_paths(new_cfg,  'train', NEW_YAML)
new_val    = extract_paths(new_cfg,  'val',   NEW_YAML)
train_dirs = coco_train + new_train
val_dirs   = coco_val   + new_val
print(f"DEBUG: train_dirs: {train_dirs}")
print(f"DEBUG: val_dirs: {val_dirs}")

# ── 3) train.txt / val.txt 생성 (디렉터리 스캔 디버깅 추가) ───────────────────────────────
def make_list(dirs, out_file):
    exts = ('*.jpg','*.jpeg','*.png')
    # 디버깅: 각 디렉터리별 이미지 개수 확인
    for d in dirs:
        p = Path(d)
        count = 0
        if p.exists() and p.is_dir():
            for ext in exts:
                count += len(list(p.rglob(ext)))
        print(f"DEBUG: scanning dir={d}, exists={p.exists()}, is_dir={p.is_dir()}, images_found={count}")
    # 파일 리스트 작성
    with open(out_file, 'w') as f:
        for d in dirs:
            for ext in exts:
                for img in sorted(Path(d).rglob(ext)):
                    f.write(str(img.resolve()) + '\n')
    size = out_file.stat().st_size
    lines = sum(1 for _ in open(out_file))
    print(f"DEBUG: {out_file.name} 생성 ({size} bytes, {lines} lines)")

# 리스트 파일 생성 실행\ nmake_list(train_dirs, TRAIN_LIST)
make_list(train_dirs, TRAIN_LIST)
make_list(val_dirs,   VAL_LIST)

# ── 디버깅: train.txt/val.txt 내용 예시 ─────────────────────────
print("DEBUG: train.txt 첫 5줄:")
for i, line in enumerate(open(TRAIN_LIST)):
    if i >= 5: break
    print("  ", line.strip())
print("DEBUG: val.txt 첫 5줄:")
for i, line in enumerate(open(VAL_LIST)):
    if i >= 5: break
    print("  ", line.strip())

# ── 4) merged_data.yaml 생성 및 내용 확인 ─────────────────────────────────
merged = {
    'nc':    merged_nc,
    'names': merged_names,
    'train': str(TRAIN_LIST.resolve()),
    'val':   str(VAL_LIST.resolve())
}
with open(MERGED_YAML, 'w') as f:
    yaml.dump(merged, f, sort_keys=False)
print(f"DEBUG: merged_data.yaml 생성됨: {MERGED_YAML}")
print("DEBUG: merged_data.yaml 내용:")
print(open(MERGED_YAML).read())

# ── 5) Fine-tuning 시작 ─────────────────────────────────────
print(f"DEBUG: Starting training with data={MERGED_YAML}")
model = YOLO(str(COCO_BEST))
model.train(
    data=str(MERGED_YAML),
    epochs=100,
    imgsz=640,
    batch=8,
    project=BASE_DIR/'runs',
    name='scooter_finetune',
    exist_ok=True
)
print("DEBUG: Training completed")

# ── 6) best_final.pt 저장 ───────────────────────────────────
src = BASE_DIR / 'runs' / 'scooter_finetune' / 'weights' / 'best.pt'
if src.exists():
    shutil.copy(src, FINAL_MODEL)
    print(f"DEBUG: 최종 모델 복사됨: {FINAL_MODEL}")
else:
    print("ERROR: best.pt를 찾을 수 없습니다:", src, file=sys.stderr)

# ── 7) 전체 클래스 검증 리포트 ───────────────────────────────
print(f"DEBUG: Calling validation with data={MERGED_YAML}")
model.val(
    data=str(MERGED_YAML),
    batch=8,
    imgsz=640,
    verbose=True
)

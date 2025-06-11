#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_data.py

- COCO 2017 val 애노테이션 다운로드
- person/vehicle 필터링
- 이미지 다운로드 및 YOLO 레이블(.txt) 생성
"""

import json, zipfile, shutil, requests
from pathlib import Path
from tqdm import tqdm

# 0) 정의
BASE_DIR      = Path(__file__).parent.resolve()
DATA_DIR      = BASE_DIR / 'datasets'
ANN_DIR       = DATA_DIR / 'annotations'
WORK_DIR      = DATA_DIR / 'coco_pv'
IMG_DIR       = WORK_DIR / 'images'
LABEL_DIR     = WORK_DIR / 'labels'
ANN_URL       = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
VAL_JSON      = 'instances_val2017.json'
FILTERED_JSON = 'instances_pv_val2017.json'
IMG_BASE_URL  = 'http://images.cocodataset.org/val2017'
SUPER_CATS    = {'person', 'vehicle'}

# 1) 애노테이션 다운로드 + 압축 해제
ANN_DIR.mkdir(parents=True, exist_ok=True)
orig_json = ANN_DIR / VAL_JSON
zip_path  = ANN_DIR / 'trainval2017.zip'
if not orig_json.exists():
    print('▶ 애노테이션 다운로드...')
    with requests.get(ANN_URL, stream=True) as r, open(zip_path, 'wb') as f:
        for chunk in r.iter_content(1024*1024): f.write(chunk)
    print('▶ 압축 해제...')
    with zipfile.ZipFile(zip_path) as z:
        z.extract(f'annotations/{VAL_JSON}', path=ANN_DIR)
    tmp = ANN_DIR / 'annotations' / VAL_JSON
    shutil.move(tmp, orig_json)
    shutil.rmtree(ANN_DIR / 'annotations')
    zip_path.unlink()
else:
    print('✓ 애노테이션 이미 존재')

# 2) person/vehicle 필터링
filtered_json = ANN_DIR / FILTERED_JSON
if not filtered_json.exists():
    print('▶ 필터링 중...')
    coco = json.loads(orig_json.read_text())
    cats = [c for c in coco['categories'] if c['supercategory'] in SUPER_CATS]
    ids  = {c['id'] for c in cats}
    anns = [a for a in coco['annotations'] if a['category_id'] in ids]
    img_ids = {a['image_id'] for a in anns}
    imgs    = [i for i in coco['images'] if i['id'] in img_ids]
    filtered = {'images': imgs, 'annotations': anns, 'categories': cats}
    filtered_json.write_text(json.dumps(filtered, ensure_ascii=False, indent=2))
    print('✓ 필터링된 JSON 저장')
else:
    print('✓ 필터링된 JSON 이미 존재')

# 3) 이미지 다운로드 + 레이블 생성
if not WORK_DIR.exists():
    print('▶ 이미지 다운로드 및 레이블 생성...')
    IMG_DIR.mkdir(parents=True)
    LABEL_DIR.mkdir(parents=True)
    data = json.loads(filtered_json.read_text())
    cat2idx = {c['id']: i for i, c in enumerate(data['categories'])}
    id2img  = {i['id']: i for i in data['images']}
    for ann in tqdm(data['annotations'], desc='Annotations'):
        img = id2img[ann['image_id']]
        fname, w, h = img['file_name'], img['width'], img['height']
        # 다운로드
        imf = IMG_DIR / fname
        if not imf.exists():
            with requests.get(f"{IMG_BASE_URL}/{fname}", stream=True) as r, open(imf, 'wb') as f:
                for chunk in r.iter_content(1024*1024): f.write(chunk)
        # 레이블
        x,y,bw,bh = ann['bbox']
        xc, yc = (x+bw/2)/w, (y+bh/2)/h
        nw, nh = bw/w, bh/h
        cls = cat2idx[ann['category_id']]
        lbl = LABEL_DIR / (Path(fname).stem + '.txt')
        with open(lbl, 'a') as f:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
    print('✓ 준비 완료')
else:
    print('✓ 데이터 디렉터리 이미 준비됨')

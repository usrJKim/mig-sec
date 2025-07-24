import os
import shutil
import random

DATASET_DIR = "tiny-imagenet-200"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
VAL_ANNOT_FILE = os.path.join(VAL_DIR, "val_annotations.txt")
VAL_IMG_DIR = os.path.join(VAL_DIR, "images")

NUM_CLASSES = 20
SEED = 42  # 재현성

# 1. 전체 클래스 목록 불러오기 (train 기준)
all_classes = sorted(os.listdir(TRAIN_DIR))
random.seed(SEED)
selected_classes = set(random.sample(all_classes, NUM_CLASSES))
print("Selected classes:", selected_classes)

# 2. train 디렉토리 정리
for cls in all_classes:
    if cls not in selected_classes:
        shutil.rmtree(os.path.join(TRAIN_DIR, cls))

# 3. val 디렉토리 정리
# (1) 먼저 이미지들을 클래스별로 분류
os.makedirs(os.path.join(VAL_DIR, "filtered_images"), exist_ok=True)
with open(VAL_ANNOT_FILE, "r") as f:
    for line in f:
        img_file, cls = line.strip().split("\t")[:2]
        if cls in selected_classes:
            cls_folder = os.path.join(VAL_DIR, "filtered_images", cls)
            os.makedirs(cls_folder, exist_ok=True)
            src = os.path.join(VAL_IMG_DIR, img_file)
            dst = os.path.join(cls_folder, img_file)
            shutil.copyfile(src, dst)

# (2) 원래 val 이미지 폴더 삭제 및 새 구조 반영
shutil.rmtree(VAL_IMG_DIR)
shutil.move(os.path.join(VAL_DIR, "filtered_images"), VAL_IMG_DIR)

print(f"✅ {NUM_CLASSES} random classes kept in train/ and val/")


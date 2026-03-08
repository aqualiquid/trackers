# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from pathlib import Path

import cv2
import numpy as np

gt_path = Path("datasets/mot17/mot17/val/MOT17-09-FRCNN/gt/gt.txt")
img_dir = Path("datasets/mot17/mot17/val/MOT17-09-FRCNN/img1")
out_path = Path("datasets/mot17/gt_09.mp4")

gt: dict[int, list[tuple[int, int, int, int, int]]] = {}
for line in gt_path.read_text().splitlines():
    f, tid, x, y, w, h, conf, cls, vis = map(float, line.split(","))
    if conf == 0:
        continue
    gt.setdefault(int(f), []).append((int(tid), int(x), int(y), int(w), int(h)))

imgs = sorted(img_dir.glob("*.jpg"))
h, w = cv2.imread(str(imgs[0])).shape[:2]
writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))

np.random.seed(42)
colors = {i: tuple(np.random.randint(50, 255, 3).tolist()) for i in range(1, 100)}

for img_path in imgs:
    frame_id = int(img_path.stem)
    img = cv2.imread(str(img_path))
    for tid, x, y, w2, h2 in gt.get(frame_id, []):
        c = colors.get(tid, (0, 255, 0))
        cv2.rectangle(img, (x, y), (x + w2, y + h2), c, 2)
        cv2.putText(img, str(tid), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    writer.write(img)

writer.release()
print("Done:", out_path)

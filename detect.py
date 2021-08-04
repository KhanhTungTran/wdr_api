from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import torch
from numpy import random

from removal import preprocess
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def detect(remover, detector, device, half):
    # Directories
    save_dir = Path('results')  # increment run


    stride = int(detector.stride.max())  # model stride
    imgsz = 640
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # Set Dataloader
    # dataset = LoadImages('100_chotot_images/100_chotot_images', img_size=imgsz, stride=stride)
    # dataset = LoadImages('bds_ct', img_size=imgsz, stride=stride)
    dataset = LoadImages('images_to_infer', img_size=imgsz, stride=stride)

    # Get names and colors
    names = detector.module.names if hasattr(detector, 'module') else detector.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        detector(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(detector.parameters())))  # run once

    paths = []
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = detector(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.2, 0.55, classes=0, agnostic=False)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            s += '%gx%g ' % img.shape[2:]  # print string

            watermarks = []
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    watermarks.append(plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3))

            im0 = tf.cast(im0, tf.float32).numpy()
            im1 = im0.copy()
            for _, (c1, c2) in enumerate(watermarks):
                im0[c1[1]:c2[1], c1[0]:c2[0]] = remover.generate_image(preprocess(im0[c1[1]:c2[1], c1[0]:c2[0]]), c2[1] - c1[1], c2[0] - c1[0])

            cv2.imwrite(save_path[:-4]+'.jpg', cv2.hconcat([im1, im0]))
            paths.append(save_path[:-4]+'.jpg')

    return paths

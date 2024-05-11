# -*- coding: utf-8 -*-
# @Time    : 2024/5/10/010 23:34
# @Author  : Shining
# @File    : trsin.py
# @Description : yolov8训练

from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO(r'D:\yolo\ultralytics\yolov8n.pt') # pass any model type
    results = model.train(
        epochs=100,
        data=r"D:\yolo\ultralytics\ultralytics\cfg\datasets\person.yaml",
        batch=8,
        device=0,
        workers=4,
        cos_lr=True
        )
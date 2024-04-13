# # -*- coding: utf-8 -*-
# # @Time    : 2024/4/7 9:32
# # @Author  : Lee
# # @Project ：yolov5
# # @File    : deal.py

# import cv2
# import os

# path = r"E:\WiderPerson\Annotations"
# path_ = r"E:\WiderPerson\Images"

# for item in os.scandir(path):
#     print(item.path)
#     file = open(item.path)
#     rows = file.readlines()[1:]
#     name = item.name
#     new_path = os.path.join(path,name[:-8]+".txt")
#     new_file = open(new_path,"a")
#     print(os.path.join(path_,name[:-8]+".jpg"))
#     image = cv2.imread(os.path.join(path_,name[:-8]+".jpg"))
#     h,w,c = image.shape
#     for row in rows:
#         lines = row.split(" ")
#         cls,xmin,ymin,xmax,ymax = lines[0],int(lines[1]),int(lines[2]),int(lines[3]),int(lines[4][:-1])
#         h_ = (ymax-ymin)/h
#         w_ = (xmax-xmin)/w
#         y_c = (ymin+((ymax-ymin)/2))/h
#         x_c = (xmin+((xmax-xmin)/2))/w

#         new_file.write("{} {} {} {} {}\n".format(cls,x_c,y_c,w_,h_))

#     new_file.close()
#     file.close()
#     os.remove(item.path)
#

# train_path = r"E:\WiderPerson\test.txt"
# new_train_path = r"E:\WiderPerson\test_new.txt"


# file = open(train_path)
# new_file = open(new_train_path,"a")

# rows = file.readlines()

# for row in rows:
#     new_file.write("/media/WiderPerson/images/{}.jpg\n".format(row[:-1]))

# import torch

# print(torch.cuda.is_available())
# import os
#
# for item in os.scandir('/media/WiderPerson/labels/'):
#     file = open(item.path)
#     new_file = open('/media/WiderPerson/new_labels/{}'.format(item.name),"a")
#     rows = file.readlines()
#     for row in rows:
#         cls,x,y,w,h = row.split(" ")
#         cls = int(cls)-1
#         new_file.write("{} {} {} {} {}".format(cls,x,y,w,h))

# import torch

# x = torch.rand(1,3,20,20,85)

# xy, wh, conf = x.split((2, 2, 80 + 1), 4)

# obj_conf = conf[..., 0:1]
# print(obj_conf.shape)
# cls_conf = conf[..., 1:]
# print(cls_conf.shape)
# cls_conf *= obj_conf

# print(cls_conf.shape)

# print(xy.shape)
# print(wh.shape)
# print(conf.shape)

# import onnx
# import onnxsim
# import torch
# from models.experimental import attempt_load

# import platform
# import pathlib
# plt = platform.system()
# if plt != 'Windows':
#   pathlib.WindowsPath = pathlib.PosixPath

# model = attempt_load("/home/kasm-user/yolov5/runs/train/exp/weights/best.pt")  # load FP32 model
# image = torch.randn(1,3,640,640)
# onnx_patn = "/home/kasm-user/yolov5/runs/train/exp/weights/best.onnx"
# input_names = ["images"]
# output_names = ['box', 'cls_conf']
# dynamic_axes={
#                 'images':{
#                   0: 'batch_size',
#                   1: 'channels',
#                   2: 'height',
#                   3: 'width'},
#                 'box': {
#                     0: 'batch_size',
#                     1: 'number_boxes',
#                     2: 'number_classes',
#                     3:'number_box_parameters'},
#                 'cls_conf': {
#                     0: 'batch_size',
#                     1: 'number_boxes',
#                     2: 'number_classes'},
#             }

# torch.onnx.export(
#         model,
#         image,
#         onnx_patn,
#         verbose=False,
#         input_names = input_names,
#         output_names=output_names,
#         dynamic_axes=dynamic_axes)

# model_onnx = onnx.load(onnx_patn)
# onnx.checker.check_model(model_onnx)  # check onnx model

# onnx_simplify_path = "/home/kasm-user/yolov5/runs/train/exp/weights/best_simplify.onnx"

# model_onnx, check = onnxsim.simplify(model_onnx)
# onnx.save(model_onnx, onnx_simplify_path)

import numpy as np
import onnx
import onnx_graphsurgeon as gs

onnx_model = onnx.load("/home/kasm-user/yolov5/runs/train/exp/weights/best_simplify.onnx")
graph = gs.import_onnx(onnx_model)

box = graph.outputs[0]
cls_conf = graph.outputs[1]

# ��
output_1 = gs.Variable(
    "num_detections",
    dtype=np.int32
)
output_2 = gs.Variable(
    "nmsed_boxes",
    dtype=np.float32
)
output_3 = gs.Variable(
    "nmsed_scores",
    dtype=np.float32
)
output_4 = gs.Variable(
    "nmsed_classes",
    dtype=np.float32
)

# ���p
decode_attrs = dict()
decode_attrs['shareLocation'] = True
decode_attrs['backgroundLabelId'] = -1
decode_attrs['numClasses'] = 5
decode_attrs['topK'] = 100
decode_attrs['keepTopK'] = 100
decode_attrs['scoreThreshold'] = 0.25
decode_attrs['iouThreshold'] = 0.45
decode_attrs['isNormalized'] = False
decode_attrs['clipBoxes'] = False

# (onnx��node
plugin = gs.Node(

    op="BatchedNMSDynamic_TRT",
    name="BatchedNMSDynamic_TRT",
        inputs=[box, cls_conf],
        outputs=[output_1, output_2,output_3,output_4],
        attrs=decode_attrs
    )

graph.nodes.append(plugin)
graph.outputs = plugin.outputs
graph.cleanup().toposort()
model_onnx = gs.export_onnx(graph)
onnx.save(model_onnx, "/home/kasm-user/yolov5/runs/train/exp/weights/best_final.onnx")
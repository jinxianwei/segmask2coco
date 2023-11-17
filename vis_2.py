import cv2
import random
import json, os
from pycocotools.coco import COCO
from skimage import io
from matplotlib import pyplot as plt
 
train_json = '/home/bennie/bennie/temp/binary-to-coco-json-converter/output/demo.json'
train_path = '/home/bennie/bennie/temp/binary-to-coco-json-converter/dataset/demo_mask/material/'
 
def visualization_bbox_seg(num_image, json_path, img_path, *str):# 需要画图的是第num副图片， 对应的json路径和图片路径
 
    coco = COCO(json_path)
 
    if len(str) == 0:
        catIds = []
    else:
        catIds = coco.getCatIds(catNms = [str[0]])  # 获取给定类别对应的id 的dict（单个内嵌字典的类别[{}]）
        catIds = coco.loadCats(catIds)[0]['id'] # 获取给定类别对应的id 的dict中的具体id
 
    list_imgIds = coco.getImgIds(catIds=catIds ) # 获取含有该给定类别的所有图片的id
    img = coco.loadImgs(list_imgIds[num_image-1])[0]  # 获取满足上述要求，并给定显示第num幅image对应的dict
    image = io.imread(img_path + img['file_name'])  # 读取图像
    image_name =  img['file_name'] # 读取图像名字
    print(image_name)
    image_id = img['id'] # 读取图像id
 
    img_annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None) # 读取这张图片的所有seg_id
    img_anns = coco.loadAnns(img_annIds)
 
    # for i in range(len(img_annIds)):
    #     x, y, w, h = img_anns[i-1]['bbox']  # 读取边框
    #     image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)
 
    plt.rcParams['figure.figsize'] = (20.0, 20.0)
    plt.imshow(image)
    coco.showAnns(img_anns)
    # plt.show()
    plt.savefig('./vis1_mask.png')
 
if __name__ == "__main__":
   visualization_bbox_seg(1, train_json, train_path,'material') # 最后一个参数不写就是画出一张图中的所有类别
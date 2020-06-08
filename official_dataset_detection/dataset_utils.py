import os
import time
from glob import glob
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from progressbar import ProgressBar
from xml.dom import minidom
import torch
import torch.nn as nn
import torch.nn.functional as F

# Note in official dataset they are in reverse....
COLOR = 'red'

np.random.seed(0)


def load_dataset_from(
        num=10,
        official_dataset_paths='./data/DJI ROCO/',
        plot=False):
    official_dataset_paths = list(
        filter(lambda x: os.path.isdir(x), glob(official_dataset_paths + '*')))
    print('loading from: ', official_dataset_paths)

    img_names = []
    annot_names = []
    for path in official_dataset_paths:
        annots = glob(path + '/image_annotation/*')
        imgs = list(
            map(lambda x: x.replace('_annotation', '').replace('xml', 'jpg'),
                annots))
        img_names += imgs
        annot_names += annots
    print(f'{len(annot_names)} annotations, {len(img_names)} images found')
    imgs = []
    annots = []
    for i in ProgressBar()(range(min(num, len(annot_names)))):  # For each image
        image = cv2.imread(img_names[i], cv2.IMREAD_UNCHANGED)
        annot = minidom.parse(annot_names[i])
        if(annot.getElementsByTagName('review_status')[0].firstChild.nodeValue != 'passed'):
            print('not passed')

        objs = annot.getElementsByTagName('object')
        objs_parsed = []
        for obj in objs:  # For each label in image
            o = dict()
            o['name'] = obj.getElementsByTagName(
                'name')[0].firstChild.nodeValue
            if(int(obj.getElementsByTagName('is_incorrect')[0].firstChild.nodeValue) != 0):
                print('incorrect object')
                continue
            if 'armor' in o['name']:
                o['armor_class'] = int(obj.getElementsByTagName(
                    'armor_class')[0].firstChild.nodeValue)
                o['armor_color'] = obj.getElementsByTagName(
                    'armor_color')[0].firstChild.nodeValue
            if len(obj.getElementsByTagName('difficulty')) != 0:
                o['difficulty'] = int(obj.getElementsByTagName(
                    'difficulty')[0].firstChild.nodeValue)
            bb = obj.getElementsByTagName('bndbox')[0]
            o['bbmin'] = np.array([float(bb.getElementsByTagName('xmin')[0].firstChild.nodeValue),
                                   float(bb.getElementsByTagName('ymin')[0].firstChild.nodeValue), ])
            o['bbmax'] = np.array([float(bb.getElementsByTagName('xmax')[0].firstChild.nodeValue),
                                   float(bb.getElementsByTagName('ymax')[0].firstChild.nodeValue), ])
            objs_parsed.append(o)
            if plot:
                cv2.rectangle(image,
                              tuple(o['bbmin'].astype(int)),
                              tuple(o['bbmax'].astype(int)),
                              color=(255, 255, 0), thickness=2)

        imgs.append(image)
        annots.append(objs_parsed)
        if plot:
            plt.figure(figsize=[18, 18])
            plt.imshow(image)
            plt.title(img_names[i]+'\n'+annot_names[i])
    return imgs, annots

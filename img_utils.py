import imgaug as ia
import imgaug.augmenters as iaa
import torch
import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib
from PIL import Image
from progressbar import ProgressBar

is_HAL = len(glob.glob('/home/shared/imagenet/raw/val_nodir/'))
if not is_HAL:
    fg_path = './models/Batch Renderding/scene/image_out/*'
    bg_paths = ['/run/media/d0048/DATA/data/imagenet/raw/val_nodir/*']
    fg_num = 10
else:  # On server
    fg_path = '/home/xiaoboh2/rm_synethesis_hal/data/Synethesized_Dataset/dataset_out/image_out/*'
    bg_paths = ['/home/shared/imagenet/raw/val_nodir/'] + \
        glob.glob('/home/shared/imagenet/raw/train/*')
    fg_num = 10

fg_seg_pairs = []


def crop_zero(image, reference=None):
    if(reference is None):
        reference = image
    nonzeros = cv2.findNonZero(reference[:, :, -1])
    upper = np.squeeze(np.max(nonzeros, axis=0).astype(int))
    lower = np.squeeze(np.min(nonzeros, axis=0).astype(int))
    return image[lower[1]:upper[1], lower[0]:upper[0]]


print(f'Loading forgrounds from {fg_path}:')
pbar = ProgressBar()
files = glob.glob(fg_path)
files.sort()
for name in pbar(files[0:fg_num]):
    try:
        image = cv2.imread(name, cv2.IMREAD_UNCHANGED)
        image[:, :, 0], image[:, :, 2] = image[:, :, 2], image[:, :, 0].copy()
        label = cv2.imread(name.replace('image', 'label'),
                           cv2.IMREAD_UNCHANGED)
        label = crop_zero(label, reference=image)
        if(np.sum(label) == 0):
            continue
        # label=np.sum(label,axis=2)
        # label[label>0]=1
        image = crop_zero(image)
        fg_seg_pairs += [[image, label]]
    except:
        pass
print(f'{len(fg_seg_pairs)} pairs of foreground loaded.')

ia.seed(1)


def sometimes(aug): return iaa.Sometimes(0.5, aug)


seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.SomeOf((0, 5),
                   [
            sometimes(
                iaa.Superpixels(
                    p_replace=(0, .2),
                    n_segments=(40, 400)
                )
            ),
            iaa.OneOf([
                iaa.GaussianBlur((0, 3.0)),
                iaa.AverageBlur(k=(2, 7)),
                iaa.MedianBlur(k=(3, 11)),
            ]),
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
            iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
            iaa.AdditiveGaussianNoise(
                loc=0, scale=(0.0, 0.05*255), per_channel=0.5
            ),
            iaa.OneOf([
                iaa.Dropout((0.01, 0.1), per_channel=0.5),
                # iaa.CoarseDropout(
                   # (0.03, 0.15), size_percent=(0.02, 0.05),
                   # per_channel=0.2
                # ),
            ]),
            iaa.Add((-40, 40), per_channel=0.5),
            iaa.Multiply((0.8, 1.5), per_channel=0.5),
            # iaa.imgcorruptlike.Contrast(severity=1),
            # iaa.imgcorruptlike.Brightness(severity=2),
            iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
        ],
            random_order=True
        )
    ],
    random_order=True
)


def augment_pair(fg, label):
    # print(fg.shape)
    # print(label.shape)
    label_i, segmaps_aug_i = seq(images=fg, segmentation_maps=label)
    #ret = seq(images=fg, segmentation_maps=label)
    return label_i, segmaps_aug_i


# res = [1080, 960]
res = [int(1080/4), int(960/4)]
print('Resolution: ', res)
# res = [270, 240]
# size ratio range, numbers, blur, shear, explosure
para_space_range = {'size': [0.4, 0.8],
                    'min_num': [1, 1], 'min_area': [0.001, 0.002]}


def get_para():
    para = para_space_range.copy()
    para['min_num'] = np.random.uniform(
        low=para_space_range['min_num'][0], high=para_space_range['min_num'][1])
    para['size'] = np.random.uniform(low=para_space_range['size'][0], high=para_space_range['size'][1],
                                     size=int(para['min_num'])*20)
    para['min_area'] = np.random.uniform(
        low=para_space_range['min_area'][0], high=para_space_range['min_area'][1])
    return para


def get_pair_PIL():
    idx = np.random.randint(0, fg_seg_pairs.__len__())
    fg_pair = fg_seg_pairs[idx]
    fg = fg_pair[0].copy()
    # fg[:,:,0:2]=0
    # fg[:,:,3][fg[:,:,2]<140]=0
    fg = Image.fromarray(fg, 'RGBA')
    label = Image.fromarray(fg_pair[1], 'RGBA')
    return fg, label


# Background File Buffer
bg_files = []
for bg_path in bg_paths:
    bg_files += glob.glob(bg_path+'*')
print('{} backgrounds found. '.format(bg_files.__len__()))
buffer = []
buffered_files = []


def get_bg_pair():
    # print(buffer.__len__())
    files = bg_files
    idx = np.random.randint(0, bg_path.__len__())
    if(files[idx] in buffered_files):
        idx = np.random.randint(0, buffer.__len__())
        bg = buffer[idx][0]
        bg_label = buffer[idx][1]
    else:
        bg = Image.open(files[idx])
        bg = bg.resize(res)
        bg_label = Image.new('RGBA', bg.size, (0, 0, 0, 0))
        buffer.append((bg, bg_label))
        buffered_files.append(files[idx])
    return bg.copy(), bg_label.copy()


def area_percent(img):
    img_full = img.copy()
    img_full[:] = 255
    return np.sum(img)/np.sum(img_full)


def get_blended(plot=False, augment=True):
    bg, bg_label = get_bg_pair()
    para = get_para()
    n = 0
    while area_percent(np.array(bg_label)) < para['min_area'] and n <= np.min(para['min_num']):
        # for n in range(int(para['min_num'])):
        fg, label = get_pair_PIL()
        newsize = (np.array(res)*para['size'][n %
                                              int(para['size'].shape[0])]).astype(np.int)
        fg = fg.resize(newsize)
        label = label.resize(newsize)

        loc = (np.random.randint(0, res[0]), np.random.randint(0, res[1]))
        bg.paste(fg, loc, fg)
        bg_label.paste(label, loc, label)
        n += 1

    fg = np.expand_dims(fg, axis=0)  # [:,:,:,0:3]
    label = np.expand_dims(label, axis=0)
    # Augmentation
    if augment:
        bg, bg_label = augment_pair(np.array(bg), np.array(bg_label))
        bg, bg_label = bg.squeeze(), bg_label.squeeze()
        # print('Agumented')
    else:
        bg, bg_label = np.array(bg), np.array(bg_label)
    if plot:
        print(para)
        plt.figure(figsize=(5, 5))
        plt.imshow(bg)
        plt.imshow(bg_label)
    return bg, bg_label

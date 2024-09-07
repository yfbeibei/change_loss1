# coding: utf-8

# In[1]:
import h5py
import PIL.Image as Image
import numpy as np
import os
import glob
import argparse
import cv2
import xml.etree.ElementTree as ET
import shutil

# get_ipython().magic(u'matplotlib inline')
parser = argparse.ArgumentParser(description='CLTR')
parser.add_argument('--data_path', type=str, default='/cluster/work/cvl/guosun/datasets/crowded-counting/cc_dataset_release/released-dataset',
                    help='the data path of jhu')

args = parser.parse_args()

with open(os.path.join(args.data_path, 'train_id.txt'), "r") as f:
    train_list = [im.strip() for im in f.readlines()]

with open(os.path.join(args.data_path, 'val_id.txt'), "r") as f:
    val_list = [im.strip() for im in f.readlines()]

with open(os.path.join(args.data_path, 'test_id.txt'), "r") as f:
    test_list = [im.strip() for im in f.readlines()]

rootpath = os.path.join(args.data_path, 'images')

train = args.data_path + '/train/images_2048/'
val = args.data_path + '/val/images_2048/'
test = args.data_path + '/test/images_2048/'
os.makedirs(train, exist_ok=True); os.makedirs(val, exist_ok=True); os.makedirs(test, exist_ok=True)

all_list=[train_list, val_list, test_list]
path_sets = [train, val, test]

count = 0
for ii in range(len(path_sets)):
    path=path_sets[ii]
    for im in all_list[ii]:
        img_path=os.path.join(rootpath, im)
        print(img_path)
        shutil.copyfile(img_path, os.path.join(path, im))

        img = cv2.imread(img_path)
        Img_data_pil = Image.open(img_path).convert('RGB')

        k = np.zeros((img.shape[0], img.shape[1]))
        mat_path = img_path.replace('images', 'annotations').replace('jpg', 'xml')
        root = ET.parse(mat_path)
        points = []
        for point in root.findall("./object/point"):
            x = int(point[0].text)
            y = int(point[1].text)
            if not (x >= 0) * (x < img.shape[1]) * (y >= 0) * (y < img.shape[0]):
                count += 1
                continue
            points.append([x,y])
            # print('x: {}; y: {}'.format(x, y))
        # if len(points) == 0:
        #     continue
        rate=1.0
        gt_file = np.array(points)
        try:
            y = gt_file[:, 0] * rate
            x = gt_file[:, 1] * rate
            for i in range(0, len(x)):
                if int(x[i]) < img.shape[0] and int(y[i]) < img.shape[1]:
                    k[int(x[i]), int(y[i])] += 1
        except Exception:
            try:
                y = gt_file[0] * rate
                x = gt_file[1] * rate

                for i in range(0, 1):
                    if int(x) < img.shape[0] and int(y) < img.shape[1]:
                        k[int(x), int(y)] += 1
            except Exception:
                ''' this image without person'''
                k = np.zeros((img.shape[0], img.shape[1]))
        if len(points)==0:
            print("here")
            assert k.sum()==0
        kpoint = k.copy()
        kpoint = kpoint.astype(np.uint8)

        os.makedirs(path.replace('images_2048', 'gt_detr_map_2048'), exist_ok=True)
        with h5py.File(os.path.join(path.replace('images_2048', 'gt_detr_map_2048'), im.replace('.jpg', '.h5')), 'w') as hf:
            hf['kpoint'] = kpoint
            hf['image'] = Img_data_pil
        # cv2.imwrite(img_path.replace('images', 'images_2048'), img)


print(f'skip {count}')
import os
import numpy as np
import argparse

if not os.path.exists('./npydata'):
    os.makedirs('./npydata')

'''please set your dataset path'''

parser = argparse.ArgumentParser(description='CLTR')
parser.add_argument('--data_path', type=str, default='/cluster/work/cvl/guosun/datasets/crowded-counting/cc_dataset_release',
                    help='the data path of cod')

args = parser.parse_args()
jhu_root = args.data_path

try:

    Jhu_train_path = jhu_root + '/train/images_2048/'
    Jhu_val_path = jhu_root + '/val/images_2048/'
    jhu_test_path = jhu_root + '/test/images_2048/'

    train_list = []
    for filename in os.listdir(Jhu_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(Jhu_train_path + filename)
    train_list.sort()
    np.save('./npydata/cod_train2048.npy', train_list)

    val_list = []
    for filename in os.listdir(Jhu_val_path):
        if filename.split('.')[1] == 'jpg':
            val_list.append(Jhu_val_path + filename)
    val_list.sort()
    np.save('./npydata/cod_val2048.npy', val_list)

    test_list = []
    for filename in os.listdir(jhu_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(jhu_test_path + filename)
    test_list.sort()
    np.save('./npydata/cod_test2048.npy', test_list)

    print("Generate JHU image list successfully", len(train_list), len(val_list), len(test_list))
except:
    print("The JHU dataset path is wrong. Please check your path.")

import numpy as np
import os
import argparse
import glob
import json
from shutil import copyfile
from collections import OrderedDict

#FIXME
SRC_ROOT_DIR = 'data/face'
DST_ROOT_DIR = 'data/face'
postfix = 'ellipseList.txt'
anno_dir = os.path.join(SRC_ROOT_DIR, 'FDDB-folds/')
img_dir = os.path.join(SRC_ROOT_DIR, 'originalPics/')
dest_img_dir = os.path.join(DST_ROOT_DIR, 'images/')
dest_anno_dir = os.path.join(DST_ROOT_DIR, 'annotations')

def _main():
    if not os.path.isdir(dest_img_dir):
        os.mkdir(dest_img_dir)
    if not os.path.isdir(dest_anno_dir):
        os.mkdir(dest_anno_dir)

    fold_path = []
    fold_path.extend(glob.glob(os.path.join(anno_dir, '*{}'.format(postfix))))

    list_dict = {}
    for fold in fold_path:
        with open(fold, 'r') as f:
            filelist = f.readlines()
        flag = True
        count = 0
        for i in filelist:
            x = i.replace('\n', '')
            if flag:
                path = x
                list_dict[path] = []
                flag = False
            elif count == 0:
                count = int(x)
            else:
                list_dict[path].append(x)
                count -= 1
                if count == 0:
                    flag = True

    for i in list_dict:
        name = '_'.join(i.split('/'))
        src = os.path.join(img_dir, '{}.jpg'.format(i))
        dest = os.path.join(dest_img_dir, '{}.jpg'.format(name))
        copyfile(src, dest)

        class_map = set()

        annos = list_dict[i]
        annotations = OrderedDict()
        annotations['face'] = []
        for j in annos:
            anno = list(map(float, j.split(' ')[:-2]))
            rad = abs(anno[2])
            h = anno[0] * np.sin(rad)
            w = anno[1] * np.sin(rad)
            x1, x2 = anno[3] - w, anno[3] + w
            y1, y2 = anno[4] - h, anno[4] + h
            annotations['face'].append([x1, y1, x2, y2])

        with open(os.path.join(dest_anno_dir, '{}.anno'.format(name)), 'w', encoding="utf-8") as m:
            json.dump(annotations, m, ensure_ascii=False, indent="\t")

        for ano_class in annotations:
            class_map.add(ano_class)

    cls_map = dict()
    k = 0
    for i in class_map:
        cls_map[str(k)] = i
    with open(os.path.join(DST_ROOT_DIR, 'classes.json'), 'w') as f:
        json.dump(cls_map, f, indent='\t')
if __name__ == '__main__':
    _main()

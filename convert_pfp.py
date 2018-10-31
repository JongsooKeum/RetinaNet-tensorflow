from glob import glob
from tqdm import tqdm
import os
import os.path as osp
import json
import re


# FIXME
SRC_IMG_DIR = './data/PennFudanPed/PNGImages'
SRC_ANNO_DIR = './data/PennFudanPed/Annotation'

# FIXME
DST_ROOT_DIR = 'PennFudanPed_converted'

RE_LABEL = 'Original label for object (?P<idx>\d+) "(?P<cls_name>\w+)" : "PennFudanPed"'
RE_BBOX = 'Bounding box for object (?P<idx>\d+) "(?P<cls_name>\w+)" \(Xmin, Ymin\) - \(Xmax, Ymax\) : \((?P<xmin>\d+), (?P<ymin>\d+)\) - \((?P<xmax>\d+), (?P<ymax>\d+)\)'


def extract_obj_dict(fpath):
    with open(fpath, 'r') as f:
        all_lines = [line.strip() for line in f.readlines()]

    # Extract all objects in a sample using regex.
    cls_name_map, bbox_map = dict(), dict()
    for line in all_lines:
        if re.match(RE_LABEL, line):
            gdict = re.match(RE_LABEL, line).groupdict()
            idx, cls_name = int(gdict['idx']), gdict['cls_name']
            cls_name_map[idx] = cls_name
        elif re.match(RE_BBOX, line):
            gdict = re.match(RE_BBOX, line).groupdict()
            idx, xmin, ymin, xmax, ymax = [
                int(gdict[t]) for t in ['idx', 'xmin', 'ymin', 'xmax', 'ymax']
            ]
            bbox_map[idx] = [xmin, ymin, xmax, ymax]
        else:
            continue

    # Create an annotation object in json format.
    anno_dict = dict()
    for idx in sorted(cls_name_map.keys()):
        cls_name, bbox = cls_name_map[idx], bbox_map[idx]
        if cls_name not in anno_dict:
            anno_dict[cls_name] = list()
        anno_dict[cls_name].append(bbox)

    return anno_dict


def main(verbose=False):
    # Get a list of source annotation file paths.
    txt_fpaths = sorted(glob(osp.join(SRC_ANNO_DIR, '*.txt')))
    print('\nNumber of files to parse: {}'.format(len(txt_fpaths)))

    anno_dicts = [extract_obj_dict(fp) for fp in txt_fpaths]

    # Verify and prepare the output directories.
    if osp.isdir(DST_ROOT_DIR):
        raise FileExistsError('Output directory ({}) already exists.'.format(DST_ROOT_DIR))
    dst_img_dir = osp.join(DST_ROOT_DIR, 'images')
    dst_anno_dir = osp.join(DST_ROOT_DIR, 'annotations')
    os.makedirs(dst_img_dir)
    os.makedirs(dst_anno_dir)

    class_map = set()
    for txt_fpath, anno_dict in zip(tqdm(txt_fpaths), anno_dicts):
        sample_name = osp.splitext(osp.basename(txt_fpath))[0]

        # Write the annotation files.
        with open(osp.join(dst_anno_dir, '{}.anno'.format(sample_name)), 'w') as f:
            json.dump(anno_dict, f, indent='\t')

        for ano_class in anno_dict:
            class_map.add(ano_class)
        # Copy the image files from the source directory.
        cmd = 'cp {} {}'.format(
            osp.join(SRC_IMG_DIR, '{}.png'.format(sample_name)),
            osp.join(dst_img_dir, '{}.png'.format(sample_name))
        )
        os.system(cmd)

    cls_map = dict()
    k = 0
    for i in class_map:
        cls_map[str(k)] = i
    with open(osp.join(DST_ROOT_DIR, 'classes.json'), 'w') as f:
        json.dump(cls_map, f, indent='\t')

if __name__ == '__main__':
    main(verbose=True)

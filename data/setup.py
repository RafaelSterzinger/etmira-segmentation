import re
from data.data_instance import DatasetInstance

import os

from config import DIR_GT, DIR_MASKS, GT_DATA, REGEX_MASK, STORAGE, VAL_PATH, TRAIN_PATH, UNLABELLED_TRAIN_PATH

from utils.utils import get_paths
import click


@click.option('--use_labelled', default=True, help='setup data for labelled instances')
def collect_data(use_labelled: bool = True, setup=False):
    instances = []
    if not os.path.exists(STORAGE):
        os.mkdir(STORAGE)
    if not os.path.exists(VAL_PATH):
        os.mkdir(VAL_PATH)
    if not os.path.exists(TRAIN_PATH):
        os.mkdir(TRAIN_PATH)
    if not os.path.exists(UNLABELLED_TRAIN_PATH):
        os.mkdir(UNLABELLED_TRAIN_PATH)

    for f in sorted(os.listdir(DIR_GT if use_labelled else DIR_MASKS)):
        res = re.search(REGEX_MASK, f)
        if res:
            view_id = res.group(1)
            # only mirrors for which no gt exists and no front side
            if not use_labelled and (view_id in GT_DATA or '_R' != view_id[-2:]):
                continue
            f_gt = os.path.join(DIR_GT if use_labelled else DIR_MASKS, f)
            obj_id = res.group(2)

            f_normals, f_albedo, f_depth = get_paths(view_id, obj_id)
            if os.path.isfile(f_normals) and os.path.isfile(f_albedo) and os.path.isfile(f_depth):
                instance = DatasetInstance(id=view_id,
                                           file_gt=f_gt,
                                           file_normals=f_normals,
                                           file_albedo=f_albedo,
                                           file_depth=f_depth,
                                           has_label=use_labelled,
                                           )
                if setup:
                    instance.set_up_data()
                    del instance
                    continue
                if not instance.is_in_validation():
                    print(f'Added dataset instance: {view_id}')
                    instances.append(instance)
            else:
                print(f'Warning: no source data found for {view_id}')
    return instances



if __name__ == '__main__':
    collect_data(setup=True)

from enum import Enum
import os
import re
from user_config import *

REGEX_MASK = r'(([a-zA-Z-]+([0-9-]+|xx))[-_].*)_(drawings|mask)'
TRAIN_PATH = os.path.join(STORAGE, "train")
VAL_PATH = os.path.join(STORAGE, "val")
TEST_PATH = os.path.join(STORAGE, "test")
UNLABELLED_TRAIN_PATH = os.path.join(STORAGE, "train_unlabelled")
DIR_GT = os.path.join(DIR_ROOT, GT_REL)
DIR_MASKS = os.path.join(DIR_ROOT, MASKS_REL)
GT_DATA = set(list(map(lambda x: x.group(1), filter(lambda x: x, map(
    lambda x: re.search(REGEX_MASK, x), os.listdir(DIR_GT))))))

SHAPE = ()
PATCH_SIZE_X = 2988
PATCH_SIZE_Y = 2240
STRIDE_X = PATCH_SIZE_X//2
STRIDE_Y = PATCH_SIZE_Y//2
SAMPLE_SIZE = 256

PREPROCESSING_PARAMS = {'depth_highpass_sigma_mm': 1,
                        'depth_normalize_stds': 4}


class MapType(Enum):
    NX = 'NX'
    NY = 'NY'
    NZ = 'NZ'
    ALBEDO = 'A'
    DEPTH = 'Z'
    GT = 'GT'

    @staticmethod
    def get_inputs_enums():
        return [MapType.NX, MapType.NY, MapType.NZ, MapType.ALBEDO, MapType.DEPTH]

    def get_channel_dim_map():
        return {MapType.NX: 0, MapType.NY: 1,
                MapType.NZ: 2, MapType.ALBEDO: 3, MapType.DEPTH: 4, MapType.GT: 5}


CHANNELS = [MapType.DEPTH]
# CHANNELS = [MapType.DEPTH, MapType.ALBEDO]
# CHANNELS = [MapType.NX, MapType.NY, MapType.NZ, MapType.DEPTH, MapType.ALBEDO]
# CHANNELS = [MapType.NX, MapType.NY, MapType.NZ]
# CHANNELS = [MapType.NX, MapType.NY, MapType.NZ, MapType.ALBEDO]

CHANNELS_TO_ID = {type: id for id, type in enumerate(CHANNELS)}

MEAN_A = 0.0911411
MEAN_D = 0.5140403
MEAN_X = 0.41616833
MEAN_Y = 0.41096762
MEAN_Z = 0.7927827
MEANS = {MapType.DEPTH: MEAN_D, MapType.ALBEDO: MEAN_A,
         MapType.NX: MEAN_X, MapType.NY: MEAN_Y, MapType.NZ: MEAN_Z}

STD_A = 0.05621303
STD_D = 0.15302444
STD_X = 0.113035046
STD_Y = 0.110428646
STD_Z = 0.06655099
STDS = {MapType.DEPTH: STD_D, MapType.ALBEDO: STD_A,
        MapType.NX: STD_X, MapType.NY: STD_Y, MapType.NZ: STD_Z}


def get_slice():
    channel_dim_map = MapType.get_channel_dim_map()
    return [channel_dim_map[c] for c in CHANNELS]


def get_input_slice(types):
    return [CHANNELS_TO_ID[c] for c in types]

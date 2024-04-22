import os
import cv2

import numpy as np

from config import DIR_MASKS, MapType
from data.setup import collect_data
from tqdm import tqdm


normals_x = []
normals_y = []
normals_z = []
albedo = []
depth = []

instances = collect_data(use_labelled=True, setup=False)
for instance in tqdm(instances):
    f_mask = os.path.join(DIR_MASKS, f"{instance.id}_mask.png")
    mask = cv2.imread(f_mask, cv2.IMREAD_UNCHANGED)[
        :, :, 2].astype(bool)

    depth.append(instance.get_map(MapType.DEPTH)[mask].flatten())
    normals_x.append(instance.get_map(MapType.NX)[mask].flatten())
    normals_y.append(instance.get_map(MapType.NY)[mask].flatten())
    normals_z.append(instance.get_map(MapType.NZ)[mask].flatten())
    albedo.append(instance.get_map(MapType.ALBEDO)[mask].flatten())
    del instance

depth = np.concatenate(depth)
normals_x = np.concatenate(normals_x)
normals_y = np.concatenate(normals_y)
normals_z = np.concatenate(normals_z)
albedo = np.concatenate(albedo)

MEAN_A = np.mean(albedo)
print(MEAN_A)
MEAN_D = np.mean(depth)
print(MEAN_D)
MEAN_X = np.mean(normals_x)
print(MEAN_X)
MEAN_Y = np.mean(normals_y)
print(MEAN_Y)
MEAN_Z = np.mean(normals_z)
print(MEAN_Z)

STD_A = np.std(albedo)
print(STD_A)
STD_D = np.std(depth)
print(STD_D)
STD_X = np.std(normals_x)
print(STD_X)
STD_Y = np.std(normals_y)
print(STD_Y)
STD_Z = np.std(normals_z)
print(STD_Z)

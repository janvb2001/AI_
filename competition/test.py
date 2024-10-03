import os
import torch
import h5py
import hdf5plugin

file_path = "competition/data/autonomous_flight-07a-lemniscate.h5"

with h5py.File(file_path, "r") as f:
    print("Keys: %s" % f.keys())

    img = torch.from_numpy(f["images"][:])

    targets = []
    targetF = f["targets"]

    for key in targetF.keys():
        targets.append(torch.from_numpy(targetF[key][:]))

# img --> img[time instance int][rgb (0,1,2)][height][width]
# targets --> targets[time instance int][gate n int][x1,y1,p1, x2,y2,p2, x3,y3,p3, x4,y4,p4]

print(img.shape)
print(targets[0].shape)

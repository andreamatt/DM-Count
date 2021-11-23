from glob import glob
import os
from typing import List
from PIL import Image
import numpy as np
import shutil
from joblib import Parallel, delayed
from scipy.io import loadmat

from preprocess.util import (
    ImageInfo,
    cal_new_size,
    printStats,
    random_phase,
    random_quality,
)


def main(input_dataset_path, output_dataset_path, min_size, max_size, threads):
    input = input_dataset_path
    output = output_dataset_path

    # Pick from all scenes, all pngs
    files = list(
        sorted(glob(os.path.join(input, "scenes*", "scene*", "pngs", "*.png")))
    )

    if os.path.exists(output):
        shutil.rmtree(output)
        os.mkdir(os.path.join(output))
    else:
        os.mkdir(os.path.join(output))
    if not os.path.exists(os.path.join(output, "train")):
        os.mkdir(os.path.join(output, "train"))
    if not os.path.exists(os.path.join(output, "val")):
        os.mkdir(os.path.join(output, "val"))
    if not os.path.exists(os.path.join(output, "test")):
        os.mkdir(os.path.join(output, "test"))

    print(f"{len(files)} images found")
    infos = Parallel(n_jobs=threads, verbose=10)(
        delayed(Process)(files[i], i, output, min_size, max_size)
        for i in range(len(files))
    )
    printStats(infos)


def generate_data(im_path, min_size, max_size):
    im = Image.open(im_path).convert("RGB")
    im_w, im_h = im.size
    mat_path = im_path.replace("pngs", "mats").replace(".png", ".mat")
    points = loadmat(mat_path)["image_info"][0][0][0].astype(np.float32)
    if len(points) > 0:  # some image has no crowd
        idx_mask = (
            (points[:, 0] >= 0)
            * (points[:, 0] <= im_w)
            * (points[:, 1] >= 0)
            * (points[:, 1] <= im_h)
        )
        points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = im.resize((im_w, im_h), Image.BICUBIC)
        points = points * rr
    return Image.fromarray(im), points


def Process(img_path, i, output, min_size, max_size):
    im, points = generate_data(img_path, min_size, max_size)

    phase = random_phase()

    im.save(os.path.join(output, phase, f"img_{i}.jpg"), quality=random_quality())
    np.save(os.path.join(output, phase, f"img_{i}.npy"), np.array(points))

    return [ImageInfo(im.width, im.height, len(points))]

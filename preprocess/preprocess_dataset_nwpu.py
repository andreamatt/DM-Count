from typing import List
from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
from joblib import Parallel, delayed
import shutil
from preprocess.util import ImageInfo, cal_new_size, printStats, random_phase


def generate_data(img_path, min_size, max_size):
    im = Image.open(img_path).convert("RGB")
    im_w, im_h = im.size
    mat_path = img_path.replace("images", "mats").replace(".jpg", ".mat")
    points = loadmat(mat_path)["annPoints"].astype(np.float32)
    if len(points) > 0:  # some image has no crowd
        idx_mask = (
            (points[:, 0] >= 0)
            * (points[:, 0] <= im_w)
            * (points[:, 1] >= 0)
            * (points[:, 1] <= im_h)
        )
        points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    if rr != 1.0:
        im = im.resize((im_w, im_h), Image.BICUBIC)
        points = points * rr
    return im, points


def Process(img_path, i, output, min_size, max_size) -> List[ImageInfo]:
    im, points = generate_data(img_path, min_size, max_size)

    phase = random_phase()

    im.save(os.path.join(output, phase, f"img_{i}.jpg"))
    np.save(os.path.join(output, phase, f"img_{i}.npy"), np.array(points))

    return [ImageInfo(im.width, im.height, len(points))]


def main(input_dataset_path, output_dataset_path, min_size, max_size, threads):
    if os.path.exists(output_dataset_path):
        shutil.rmtree(output_dataset_path)
        os.mkdir(os.path.join(output_dataset_path))
    else:
        os.mkdir(os.path.join(output_dataset_path))
    if not os.path.exists(os.path.join(output_dataset_path, "train")):
        os.mkdir(os.path.join(output_dataset_path, "train"))
    if not os.path.exists(os.path.join(output_dataset_path, "val")):
        os.mkdir(os.path.join(output_dataset_path, "val"))
    if not os.path.exists(os.path.join(output_dataset_path, "test")):
        os.mkdir(os.path.join(output_dataset_path, "test"))

    ori_img_path = os.path.join(input_dataset_path, "images")

    all_images = []
    for phase in ["train", "val"]:
        sub_save_dir = os.path.join(output_dataset_path, phase)
        with open(os.path.join(input_dataset_path, f"{phase}.txt")) as f:
            lines = f.readlines()
            for i in lines:
                i = i.strip().split(" ")[0]
                im_path = os.path.join(ori_img_path, i + ".jpg")
                all_images.append(im_path)

    print(f"Found {len(all_images)} images")
    infos = Parallel(n_jobs=threads, verbose=10)(
        delayed(Process)(all_images[i], i, output_dataset_path, min_size, max_size)
        for i in range(len(all_images))
    )
    printStats(infos)

from glob import glob
import os
import cv2
import numpy as np
import csv
import random
import shutil
from joblib import Parallel, delayed
from scipy.io import loadmat


def hex_to_rgb(hex):
	return list(int(hex[i:i + 2], 16) for i in (0, 2, 4))


train_percent = 0.8
train_val_percent = 0.2


def main(input_dataset_path, output_dataset_path):
	input = input_dataset_path
	output = output_dataset_path

	# Pick from all scenes, all mat
	files = list(sorted(glob(os.path.join(input, "scenes*", "scene*", "mats", "*.mat"))))

	if os.path.exists(output):
		shutil.rmtree(output)
		os.mkdir(os.path.join(output))
	else:
		os.mkdir(os.path.join(output))
	if not os.path.exists(os.path.join(output, 'train')):
		os.mkdir(os.path.join(output, 'train'))
	if not os.path.exists(os.path.join(output, 'val')):
		os.mkdir(os.path.join(output, 'val'))
	if not os.path.exists(os.path.join(output, 'test')):
		os.mkdir(os.path.join(output, 'test'))

	print(f"{len(files)} images found")
	Parallel(n_jobs=16, verbose=10)(delayed(Process)(files[i], i, output) for i in range(len(files)))


def Process(mat_path, i, output):
	img_path = mat_path.replace("mats", "pngs").replace(".mat", ".png")
	standard = cv2.imread(img_path)
	points = loadmat(mat_path)['image_info'][0][0][0].astype(np.float32)

	if random.random() < train_percent:
		if random.random() < train_val_percent:
			phase = 'val'
		else:
			phase = 'train'
	else:
		phase = 'test'

	cv2.imwrite(os.path.join(output, phase, f"img_{i}.jpg"), standard, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(85, 95)])
	np.save(os.path.join(output, phase, f"img_{i}.npy"), np.array(points))

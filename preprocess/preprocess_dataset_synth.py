from glob import glob
import os
from PIL import Image, ImageFilter
import numpy as np
import csv
import random
import shutil
from joblib import Parallel, delayed
import cv2
from preprocess.util import cal_new_size, hex_to_rgb, noisy, random_blur, random_phase, random_quality


def main(input_dataset_path, output_dataset_path, augmentation, min_size, max_size, threads):
	input = input_dataset_path
	output = output_dataset_path

	# Pick from all simulations, all cameras, all txt
	files = list(sorted(glob(os.path.join(input, "Simulation*", "*", "*.txt"))))

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
	Parallel(n_jobs=threads, verbose=10)(delayed(Process)(files[i], i, output, augmentation, min_size, max_size) for i in range(len(files)))
	# for i in range(len(files)):
	# 	print(files[i])
	# 	Process(files[i], i, output, augmentation, min_size, max_size)


def generate_data(im_path, min_size, max_size):
	im = Image.open(im_path).convert('RGB')
	im_w, im_h = im.size
	txt_path = im_path.replace(".bmp", ".txt")
	segmentation_path = txt_path.replace(".txt", "_mask.bmp")
	points = []
	segmentation = Image.open(segmentation_path).convert('RGB')
	with open(txt_path) as csvfile:
		reader = csv.reader(csvfile, delimiter=";")
		for x, y, c in reader:
			r, g, b = hex_to_rgb(c)
			if segmentation.getpixel((float(x), float(y))) == (r, g, b):
				points.append([float(x), float(y)])
	if len(points)==0:
		points = np.empty((0, 2))
	else:
		points = np.array(points)

	im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
	if rr != 1.0:
		im = im.resize((im_w, im_h), Image.BICUBIC)
		points = points * rr
	return im, points


def Process(txt_path, i, output, augmentation, min_size, max_size):
	standard_path = txt_path.replace(".txt", ".bmp")
	segmentation_path = txt_path.replace(".txt", "_mask.bmp")
	if os.path.exists(standard_path) and os.path.exists(segmentation_path):
		im, points = generate_data(standard_path, min_size, max_size)

		phase = random_phase()

		im.save(os.path.join(output, phase, f"img_{i}.jpg"), quality=random_quality())
		np.save(os.path.join(output, phase, f"img_{i}.npy"), points)

		if augmentation:
			phase = 'train'
			noisy_standard = noisy(im)
			noisy_standard.save(os.path.join(output, phase, f"img_{i}_noise.jpg"), quality=random_quality())
			np.save(os.path.join(output, phase, f"img_{i}_noise.npy"), points)

			blur_size = random_blur()
			blurred_standard = Image.fromarray(cv2.GaussianBlur(np.array(im), (blur_size, blur_size)))
			blurred_standard.save(os.path.join(output, phase, f"img_{i}_blur.jpg"), quality=random_quality())
			np.save(os.path.join(output, phase, f"img_{i}_blur.npy"), points)

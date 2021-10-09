from glob import glob
import os
import cv2
import numpy as np
import csv
import random
import shutil
from joblib import Parallel, delayed


def hex_to_rgb(hex):
	return list(int(hex[i:i + 2], 16) for i in (0, 2, 4))


train_percent = 0.8
train_val_percent = 0.2
blur_chance = 0.2
blur_kernels = (3, 5, 7)
blur_kernels_weights = (0.6, 0.3, 0.1)
noise_chance = 0.2
noises = ('gauss', 's&p', 'poisson', 'speckle')
noises_weights = (0.3, 0.3, 0.3, 0.1)


def main(input_dataset_path, output_dataset_path):
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
	Parallel(n_jobs=16, verbose=10)(delayed(Process)(files[i], i, output) for i in range(len(files)))


def Process(txt_path, i, output):
	standard_path = txt_path.replace(".txt", ".bmp")
	segmentation_path = txt_path.replace(".txt", "_mask.bmp")
	if os.path.exists(standard_path) and os.path.exists(segmentation_path):
		standard = cv2.imread(standard_path)
		segmentation = cv2.imread(segmentation_path)
		if random.random() < noise_chance:
			standard = noisy(random.choices(noises, noises_weights)[0], standard)
		if random.random() < blur_chance:
			blur_size = random.choices(blur_kernels, blur_kernels_weights)[0]
			standard = cv2.blur(standard, (blur_size, blur_size))
		points = []
		with open(txt_path) as csvfile:
			reader = csv.reader(csvfile, delimiter=";")
			for x, y, c in reader:
				r, g, b = hex_to_rgb(c)
				if (segmentation[round(float(y)) - 1][round(float(x)) - 1] == [b, g, r]).all():
					points.append((float(x), float(y)))

		if random.random() < train_percent:
			if random.random() < train_val_percent:
				phase = 'val'
			else:
				phase = 'train'
		else:
			phase = 'test'

		cv2.imwrite(os.path.join(output, phase, f"img_{i}.jpg"), standard, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(85, 95)])
		np.save(os.path.join(output, phase, f"img_{i}.npy"), np.array(points))


def noisy(noise_typ, image):
	if noise_typ == "gauss":
		row, col, ch = image.shape
		mean = 0
		var = 0.1
		sigma = var**0.5
		gauss = np.random.normal(mean, sigma, (row, col, ch))
		gauss = gauss.reshape(row, col, ch)
		noisy = image + gauss
		return noisy

	elif noise_typ == "s&p":
		row, col, ch = image.shape
		s_vs_p = 0.5
		amount = 0.004
		out = np.copy(image)
		# Salt mode
		num_salt = np.ceil(amount * image.size * s_vs_p)
		coords = tuple([np.random.randint(0, i - 1, int(num_salt)) for i in image.shape])
		out[coords] = 1
		# Pepper mode
		num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
		coords = tuple([np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape])
		out[coords] = 0
		return out

	elif noise_typ == "poisson":
		vals = len(np.unique(image))
		vals = 2**np.ceil(np.log2(vals))
		noisy = np.random.poisson(image * vals) / float(vals)
		return noisy

	elif noise_typ == "speckle":
		row, col, ch = image.shape
		gauss = np.random.randn(row, col, ch)
		gauss = gauss.reshape(row, col, ch)
		noisy = image + image * gauss
		return noisy
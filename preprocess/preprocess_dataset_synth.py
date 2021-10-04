from glob import glob
import os
import cv2
import numpy as np
import csv
import random


def main(input_dataset_path, output_dataset_path):
	source = input_dataset_path
	output = output_dataset_path
	train_percent = 0.8

	# Pick from all simulations, all cameras, all txt
	path = os.path.join(source, "Simulation*", "*", "*.txt")
	files = list(sorted(glob(path)))

	def hex_to_rgb(hex):
		return list(int(hex[i:i+2], 16) for i in (0, 2, 4))

	if not os.path.exists(os.path.join(output)):
		os.mkdir(output)

	print(f"{len(files)} images found")
	for i in range(len(files)):
		file = files[i]
		standard = cv2.imread(file.replace(".txt", ".bmp"))
		segmentation = cv2.imread(file.replace(".txt", "_mask.bmp"))
		points = []
		with open(file) as csvfile:
			reader = csv.reader(csvfile, delimiter=";")
			for x, y, c in reader:
				r, g, b = hex_to_rgb(c)
				if (segmentation[round(float(y))-1][round(float(x))-1] == [b, g, r]).all():
					points.append((float(x), float(y)))

		phase = 'train' if random.random() < train_percent else 'val'
		cv2.imwrite(os.path.join(output, phase, f"img_{i}.jpg"), standard, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(60, 100)])
		np.save(os.path.join(output, phase, f"img_{i}.npy"), np.array(points))
		
		print(f"Copied {int(i*100.0/len(files))}%")

	print("Finished")


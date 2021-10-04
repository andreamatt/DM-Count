import argparse
from glob import glob
import os
import cv2
import pandas as pd
import numpy as np
import sys


def parse_args(args):
	parser = argparse.ArgumentParser(description='Display original')
	parser.add_argument('--data', default='DATA/raw/Synth', help='data path')
	return parser.parse_args(args)


def main(args):
	source = parse_args(args).data

	# Pick from all simulations, all cameras, all bmp
	path = os.path.join(source, "Simulation*", "*", "*.txt")
	files = list(sorted(glob(path)))

	def hex_to_rgb(hex):
		return list(int(hex[i:i + 2], 16) for i in (0, 2, 4))

	i = 0
	while i < len(files):
		test_file = files[i]
		annot_file = files[i]
		annot = pd.read_csv(annot_file, names=["x", "y", "c"], sep=";")
		image = cv2.imread(test_file.replace("txt", "bmp"))

		x = np.rint(annot["x"]).astype(int)
		y = np.rint(annot["y"]).astype(int)

		for p in range(len(x)):
			image = cv2.circle(image, (x[p], y[p]), radius=1, color=(0, 0, 255, 0), thickness=1)
			r, g, b = hex_to_rgb(annot.iloc[p, 2])
			image = cv2.circle(image, (x[p], y[p]), radius=3, color=(r, g, b), thickness=2)

		cv2.imshow("image", image)
		k = cv2.waitKey(0)
		if k == 97:
			i -= 1
		elif k == 100:
			i += 1
		elif k == 32:
			i = len(files)


if __name__ == '__main__':
	main(sys.argv)
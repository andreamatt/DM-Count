import argparse
from glob import glob
import os
import sys
import cv2
import pandas as pd
import numpy as np


def parse_args(args):
	parser = argparse.ArgumentParser(description='Display original')
	parser.add_argument('--data', default='DATA/processed/Synth', help='data path')
	return parser.parse_args(args)


def main(args):
	source = parse_args(args).data

	# Pick from all phases
	path = os.path.join(source, '*', "*.npy")
	files = list(sorted(glob(path)))

	i = 0
	while i < len(files):
		annot_file = files[i]
		annot = np.load(annot_file)
		image = cv2.imread(annot_file.replace("npy", "jpg"))
		x = np.rint(annot[:, 0]).astype(int)
		y = np.rint(annot[:, 1]).astype(int)

		for p in range(len(x)):
			image = cv2.circle(image, (x[p], y[p]), radius=1, color=(0, 0, 255, 0), thickness=1)

		cv2.imshow("image", image)
		k = cv2.waitKey(0)
		if k == 97:
			i -= 1
		elif k == 100:
			i += 1
		elif k == 32:
			i = len(files)


if __name__ == '__main__':
	main(sys.argv[1:])
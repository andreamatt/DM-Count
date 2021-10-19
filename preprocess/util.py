import random
from skimage.util import random_noise
from PIL import Image
import numpy as np


def cal_new_size(im_h, im_w, min_size, max_size):
	# horizontal or vertical
	if max(im_h, im_w) > max_size:
		ratio = max_size / max(im_h, im_w)
		return round(ratio * im_h), round(ratio * im_w), ratio
	elif min(im_h, im_w) < min_size:
		ratio = min_size / min(im_h, im_w)
		return round(ratio * im_h), round(ratio * im_w), ratio
	else:
		return im_h, im_w, 1.0


def random_phase():
	return random.choices(['train', 'val', 'test'], [0.6, 0.1, 0.3])[0]


def random_quality():
	return random.choices([75, 80, 85, 90, 95, 100], [0.05, 0.05, 0.15, 0.15, 0.20, 0.40])[0]


def random_blur():
	return random.choices((3, 5, 7), (0.6, 0.3, 0.1))[0]


def hex_to_rgb(hex):
	return list(int(hex[i:i + 2], 16) for i in (0, 2, 4))


def noisy(image):
	image = np.array(image)
	noise_typ = random.choices(('gaussian', 's&p', 'poisson', 'speckle'), (0.3, 0.3, 0.3, 0.1))[0]
	noisy = random_noise(image, mode=noise_typ)
	return Image.fromarray((noisy * 255).astype(np.uint8))

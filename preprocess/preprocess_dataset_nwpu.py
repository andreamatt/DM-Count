from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
import cv2


def cal_new_size_v2(im_h, im_w, min_size, max_size):
	rate = 1.0 * max_size / im_h
	rate_w = im_w * rate
	if rate_w > max_size:
		rate = 1.0 * max_size / im_w
	tmp_h = int(1.0 * im_h * rate / 16) * 16

	if tmp_h < min_size:
		rate = 1.0 * min_size / im_h
	tmp_w = int(1.0 * im_w * rate / 16) * 16

	if tmp_w < min_size:
		rate = 1.0 * min_size / im_w
	tmp_h = min(max(int(1.0 * im_h * rate / 16) * 16, min_size), max_size)
	tmp_w = min(max(int(1.0 * im_w * rate / 16) * 16, min_size), max_size)

	rate_h = 1.0 * tmp_h / im_h
	rate_w = 1.0 * tmp_w / im_w
	assert tmp_h >= min_size and tmp_h <= max_size
	assert tmp_w >= min_size and tmp_w <= max_size
	return tmp_h, tmp_w, rate_h, rate_w


def generate_data(im_path, mat_path, min_size, max_size):
	im = Image.open(im_path).convert('RGB')
	im_w, im_h = im.size
	points = loadmat(mat_path)['annPoints'].astype(np.float32)
	if len(points) > 0:  # some image has no crowd
		idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
		points = points[idx_mask]
	im_h, im_w, rr_h, rr_w = cal_new_size_v2(im_h, im_w, min_size, max_size)
	im = np.array(im)
	if rr_h != 1.0 or rr_w != 1.0:
		im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
		if len(points) > 0:  # some image has no crowd
			points[:, 0] = points[:, 0] * rr_w
			points[:, 1] = points[:, 1] * rr_h

	return Image.fromarray(im), points


def generate_image(im_path, min_size, max_size):
	im = Image.open(im_path).convert('RGB')
	im_w, im_h = im.size
	im_h, im_w, rr_h, rr_w = cal_new_size_v2(im_h, im_w, min_size, max_size)
	im = np.array(im)
	if rr_h != 1.0 or rr_w != 1.0:
		im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
	return Image.fromarray(im)


def main(input_dataset_path, output_dataset_path, min_size=384, max_size=1920):
	ori_img_path = os.path.join(input_dataset_path, 'images')
	ori_anno_path = os.path.join(input_dataset_path, 'mats')

	for phase in ['train', 'val']:
		sub_save_dir = os.path.join(output_dataset_path, phase)
		if not os.path.exists(sub_save_dir):
			os.makedirs(sub_save_dir)
		with open(os.path.join(input_dataset_path, f'{phase}.txt')) as f:
			lines = f.readlines()
			for i in lines:
				i = i.strip().split(' ')[0]
				im_path = os.path.join(ori_img_path, i + '.jpg')
				mat_path = os.path.join(ori_anno_path, i + '.mat')
				name = os.path.basename(im_path)
				im_save_path = os.path.join(sub_save_dir, name)
				print(name)
				im, points = generate_data(im_path, mat_path, min_size, max_size)
				im.save(im_save_path)
				gd_save_path = im_save_path.replace('jpg', 'npy')
				np.save(gd_save_path, points)

	for phase in ['test']:
		sub_save_dir = os.path.join(output_dataset_path, phase)
		if not os.path.exists(sub_save_dir):
			os.makedirs(sub_save_dir)
		with open(os.path.join(input_dataset_path, f'{phase}.txt')) as f:
			lines = f.readlines()
			for i in lines:
				i = i.strip().split(' ')[0]
				im_path = os.path.join(ori_img_path, i + '.jpg')
				name = os.path.basename(im_path)
				im_save_path = os.path.join(sub_save_dir, name)
				print(name)
				im = generate_image(im_path, min_size, max_size)
				im.save(im_save_path)

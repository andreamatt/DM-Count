from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
from glob import glob
from preprocess.util import ImageInfo, cal_new_size, printStats


def generate_data(im_path, min_size, max_size):
	im = Image.open(im_path).convert('RGB')
	im_w, im_h = im.size
	mat_path = im_path.replace('.jpg', '_ann.mat')
	points = loadmat(mat_path)['annPoints'].astype(np.float32)
	if len(points) > 0:  # some image has no crowd
		idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
		points = points[idx_mask]
	im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
	if rr != 1.0:
		im = im.resize((im_w, im_h), Image.BICUBIC)
		points = points * rr
	return im, points


def main(input_dataset_path, output_dataset_path, min_size, max_size):
	infos = []
	for phase in ['Train', 'Test']:
		sub_dir = os.path.join(input_dataset_path, phase)
		if phase == 'Train':
			sub_phase_list = ['train', 'val']
			for sub_phase in sub_phase_list:
				sub_save_dir = os.path.join(output_dataset_path, sub_phase)
				if not os.path.exists(sub_save_dir):
					os.makedirs(sub_save_dir)
				with open(os.path.join(input_dataset_path, f'qnrf_{sub_phase}.txt')) as f:
					for i in f:
						im_path = os.path.join(sub_dir, i.strip())
						name = os.path.basename(im_path)
						print(name)
						im, points = generate_data(im_path, min_size, max_size)
						im_save_path = os.path.join(sub_save_dir, name)
						im.save(im_save_path)
						gd_save_path = im_save_path.replace('jpg', 'npy')
						np.save(gd_save_path, points)
						infos.append([ImageInfo(im.width, im.height, len(points))])
		else:
			sub_save_dir = os.path.join(output_dataset_path, 'test')
			if not os.path.exists(sub_save_dir):
				os.makedirs(sub_save_dir)
			im_list = glob(os.path.join(sub_dir, '*jpg'))
			for im_path in im_list:
				name = os.path.basename(im_path)
				print(name)
				im, points = generate_data(im_path, min_size, max_size)
				im_save_path = os.path.join(sub_save_dir, name)
				im.save(im_save_path)
				gd_save_path = im_save_path.replace('jpg', 'npy')
				np.save(gd_save_path, points)
				infos.append([ImageInfo(im.width, im.height, len(points))])
	printStats(infos)

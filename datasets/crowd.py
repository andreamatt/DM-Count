from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np
import scipy.io as sio


def random_crop(im_h, im_w, crop_h, crop_w):
	res_h = im_h - crop_h
	res_w = im_w - crop_w
	i = random.randint(0, res_h)
	j = random.randint(0, res_w)
	return i, j, crop_h, crop_w


def gen_discrete_map(im_height, im_width, points):
	"""
	func: generate the discrete map.
	points: [num_gt, 2], for each row: [width, height]
	"""
	discrete_map = np.zeros([im_height, im_width], dtype=np.float32)
	h, w = discrete_map.shape[:2]
	num_gt = points.shape[0]
	if num_gt == 0:
		return discrete_map

	# fast create discrete map
	points_np = np.array(points).round().astype(int)
	p_h = np.minimum(points_np[:, 1], np.array([h - 1] * num_gt).astype(int))
	p_w = np.minimum(points_np[:, 0], np.array([w - 1] * num_gt).astype(int))
	p_index = torch.from_numpy(p_h * im_width + p_w)
	discrete_map = torch.zeros(im_width * im_height).scatter_add_(0, index=p_index.long(), src=torch.ones(im_width * im_height)).view(im_height, im_width).numpy()
	''' slow method
	for p in points:
		p = np.round(p).astype(int)
		p[0], p[1] = min(h - 1, p[1]), min(w - 1, p[0])
		discrete_map[p[0], p[1]] += 1
	'''
	assert np.sum(discrete_map) == num_gt
	return discrete_map


class Crowd(data.Dataset):
	def __init__(self, dataset, root_path, crop_size, downsample_ratio=8, method='train', mixed=False, synth_path=None):
		self.root_path = root_path
		self.c_size = crop_size
		self.d_ratio = downsample_ratio
		assert self.c_size % self.d_ratio == 0
		self.dc_size = self.c_size // self.d_ratio
		# transform images: normalize them with ImageNET mean and std (https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2)
		self.trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

		self.method = method
		if method not in ['train', 'val']:
			raise Exception("Method not implemented")

		if dataset in ('sha', 'shb'):
			self.im_list = sorted(glob(os.path.join(self.root_path, 'images', '*.jpg')))
			self.kp_list = []
			for im_path in self.im_list:
				name = os.path.basename(im_path).split('.')[0]
				gd_path = os.path.join(self.root_path, 'ground-truth', f'GT_{name}.mat')
				keypoints = sio.loadmat(gd_path)['image_info'][0][0][0][0][0]
				self.kp_list.append(keypoints)

		elif dataset in ('qnrf', 'nwpu', 'synth', 'gcc'):
			self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
			self.kp_list = []
			for im_path in self.im_list:
				name = os.path.basename(im_path).split('.')[0]
				gd_path = im_path.replace('jpg', 'npy')
				keypoints = np.load(gd_path)
				self.kp_list.append(keypoints)

		else:
			raise Exception("Dataset not supported")

		if mixed:
			if dataset == 'synth':
				raise Exception("Cannot mix synth with synth")
			else:
				synth = Crowd('synth', synth_path, crop_size, downsample_ratio, method)
				self.im_list += synth.im_list
				self.kp_list += synth.kp_list

		print(f'number of img: {len(self.im_list)}')
		pass

	def __len__(self):
		return len(self.im_list)

	def __getitem__(self, index):
		img_path = self.im_list[index]
		keypoints = self.kp_list[index]
		name = os.path.basename(img_path).split('.')[0]
		img = Image.open(img_path).convert('RGB')

		if self.method == 'train':
			return self.train_transform(img, keypoints)
		elif self.method == 'val':
			img = self.trans(img)
			return img, len(keypoints), name

	def train_transform(self, img, keypoints):
		wd, ht = img.size
		st_size = 1.0 * min(wd, ht)
		# resize the image to fit the crop size
		if st_size < self.c_size:
			rr = 1.0 * self.c_size / st_size
			wd = round(wd * rr)
			ht = round(ht * rr)
			st_size = 1.0 * min(wd, ht)
			img = img.resize((wd, ht), Image.BICUBIC)
			keypoints = keypoints * rr
		assert st_size >= self.c_size, print(wd, ht)

		i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
		img = F.crop(img, i, j, h, w)
		if len(keypoints) > 0:
			keypoints = keypoints - [j, i]
			idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
			keypoints = keypoints[idx_mask]
		else:
			keypoints = np.empty([0, 2])

		gt_discrete = gen_discrete_map(h, w, keypoints)
		down_w = w // self.d_ratio
		down_h = h // self.d_ratio
		gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
		assert np.sum(gt_discrete) == len(keypoints)

		if random.random() > 0.5:
			img = F.hflip(img)
			gt_discrete = np.fliplr(gt_discrete)
			if len(keypoints) > 0:
				keypoints[:, 0] = w - keypoints[:, 0]
		gt_discrete = np.expand_dims(gt_discrete, 0)

		return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(gt_discrete.copy()).float()

import os
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
from datetime import datetime

from datasets.crowd import Crowd
from losses.ot_loss import OT_Loss
from models import vgg19
from losses.mt_loss import MT_Loss
from utils.pytorch_utils import Save_Handle, AverageMeter
import utils.log_utils as log_utils


def train_collate(batch):
	transposed_batch = list(zip(*batch))
	images = torch.stack(transposed_batch[0], 0)
	points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
	gt_discretes = torch.stack(transposed_batch[2], 0)
	return images, points, gt_discretes


class Trainer(object):
	def __init__(self, args):
		self.args = args

	def setup(self):
		args = self.args
		sub_dir = f'input-{args.crop_size}_wot-{args.wot}_wtv-{args.wtv}_reg-{args.reg}_nIter-{args.num_of_iter_in_ot}_normCood-{args.norm_cood}'

		self.save_dir = os.path.join(args.save_dir, sub_dir)
		if not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)

		time_str = datetime.strftime(datetime.now(), '%m%d-%H%M%S')
		self.logger = log_utils.get_logger(os.path.join(self.save_dir, f'train-{time_str:s}.log'))
		log_utils.print_config(vars(args), self.logger)

		if torch.cuda.is_available():
			self.device = torch.device("cuda")
			self.device_count = torch.cuda.device_count()
			assert self.device_count == 1
			self.logger.info('using {self.device_count} gpus')
		else:
			raise Exception("gpu is not available")

		downsample_ratio = 8
		train_subdir = 'train' if args.dataset.lower() in ('qnrf', 'nwpu', 'synth', 'gcc') else 'train_data'
		test_subdir = 'val' if args.dataset.lower() in ('qnrf', 'nwpu', 'synth', 'gcc') else 'test_data'
		self.datasets = {
		    'train': Crowd(args.dataset.lower(), os.path.join(args.data_dir, train_subdir), args.crop_size, downsample_ratio, 'train', args.mixed, args.synth_path),
		    'val': Crowd(args.dataset.lower(), os.path.join(args.data_dir, test_subdir), args.crop_size, downsample_ratio, 'val', args.mixed, args.synth_path),
		}

		self.dataloaders = {x: DataLoader(self.datasets[x], collate_fn=(train_collate if x == 'train' else default_collate), batch_size=(args.batch_size if x == 'train' else 1), shuffle=(True if x == 'train' else False), num_workers=args.num_workers * self.device_count, pin_memory=(True if x == 'train' else False)) for x in ['train', 'val']}
		self.model = vgg19()
		self.model.to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

		self.start_epoch = 0
		if args.resume:
			self.logger.info('loading pretrained model from ' + args.resume)
			suf = args.resume.rsplit('.', 1)[-1]
			if suf == 'tar':
				checkpoint = torch.load(args.resume, self.device)
				self.model.load_state_dict(checkpoint['model_state_dict'])
				self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
				self.start_epoch = checkpoint['epoch'] + 1
			elif suf == 'pth':
				self.model.load_state_dict(torch.load(args.resume, self.device))
		else:
			self.logger.info('random initialization')

		self.ot_loss = MT_Loss(args.crop_size, downsample_ratio, args.norm_cood, self.device, args.num_of_iter_in_ot, args.reg)
		self.tv_loss = nn.L1Loss(reduction='none').to(self.device)
		self.mse = nn.MSELoss().to(self.device)
		self.mae = nn.L1Loss().to(self.device)
		self.save_list = Save_Handle(max_num=1)
		self.best_mae = np.inf
		self.best_mse = np.inf
		self.best_count = 0

	def train(self):
		"""training process"""
		args = self.args
		for epoch in range(self.start_epoch, args.max_epoch + 1):
			self.logger.info(f'----- Epoch {epoch}/{args.max_epoch} -----')
			self.epoch = epoch
			self.train_epoch()
			if epoch % args.val_epoch == 0 and epoch >= args.val_start:
				self.val_epoch()

	def train_epoch(self):
		epoch_ot_loss = AverageMeter()
		epoch_ot_obj_value = AverageMeter()
		epoch_wd = AverageMeter()
		epoch_count_loss = AverageMeter()
		epoch_tv_loss = AverageMeter()
		epoch_loss = AverageMeter()
		epoch_mae = AverageMeter()
		epoch_mse = AverageMeter()
		epoch_start = time.time()
		epoch_loss_cost = 0
		self.model.train()  # Set model to training mode

		for step, (inputs, points, gt_discrete) in enumerate(self.dataloaders['train']):
			inputs = inputs.to(self.device)
			gd_count = np.array([len(p) for p in points], dtype=np.float32)
			points = [p.to(self.device) for p in points]
			gt_discrete = gt_discrete.to(self.device)
			N = inputs.size(0)

			with torch.set_grad_enabled(True):
				outputs, outputs_normed = self.model(inputs)

				# Compute OT loss.
				ot_loss, wd, ot_obj_value, cost = self.ot_loss(outputs_normed, outputs, points)
				epoch_loss_cost += cost
				ot_loss = ot_loss * self.args.wot
				ot_obj_value = ot_obj_value * self.args.wot
				epoch_ot_loss.update(ot_loss.item(), N)
				epoch_ot_obj_value.update(ot_obj_value.item(), N)
				epoch_wd.update(wd, N)

				# Compute counting loss.
				count_loss = self.mae(outputs.sum(1).sum(1).sum(1), torch.from_numpy(gd_count).float().to(self.device))
				epoch_count_loss.update(count_loss.item(), N)

				# Compute TV loss.
				gd_count_tensor = torch.from_numpy(gd_count).float().to(self.device).unsqueeze(1).unsqueeze(2).unsqueeze(3)
				gt_discrete_normed = gt_discrete / (gd_count_tensor + 1e-6)
				tv_loss = (self.tv_loss(outputs_normed, gt_discrete_normed).sum(1).sum(1).sum(1) * torch.from_numpy(gd_count).float().to(self.device)).mean(0) * self.args.wtv
				epoch_tv_loss.update(tv_loss.item(), N)

				loss = ot_loss + count_loss + tv_loss

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				pred_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
				pred_err = pred_count - gd_count
				epoch_loss.update(loss.item(), N)
				epoch_mse.update(np.mean(pred_err * pred_err), N)
				epoch_mae.update(np.mean(abs(pred_err)), N)

		self.logger.info(f'Epoch {self.epoch} Train, Loss: {epoch_loss.get_avg():.2f}, OT Loss: {epoch_ot_loss.get_avg():.2e}, Wass Distance: {epoch_wd.get_avg():.2f}, OT obj value: {epoch_ot_obj_value.get_avg():.2f}, Count Loss: {epoch_count_loss.get_avg():.2f}, TV Loss: {epoch_tv_loss.get_avg():.2f}, MSE: {np.sqrt(epoch_mse.get_avg()):.2f} MAE: {epoch_mae.get_avg():.2f}, Cost {time.time() - epoch_start:.1f} ({epoch_loss_cost:.1f}) sec')
		model_state_dic = self.model.state_dict()
		save_path = os.path.join(self.save_dir, f'{self.epoch}_ckpt.tar')
		torch.save({'epoch': self.epoch, 'optimizer_state_dict': self.optimizer.state_dict(), 'model_state_dict': model_state_dic}, save_path)
		self.save_list.append(save_path)

	def val_epoch(self):
		args = self.args
		epoch_start = time.time()
		epoch_res = []
		self.model.eval()  # Set model to evaluate mode
		with torch.no_grad():
			for inputs, count, name in self.dataloaders['val']:
				print(f"Validating {name}")
				inputs = inputs.to(self.device)
				assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
				outputs, _ = self.model(inputs)
				del inputs
				res = count[0].item() - torch.sum(outputs).item()
				epoch_res.append(res)

		epoch_res = np.array(epoch_res)
		mse = np.sqrt(np.mean(np.square(epoch_res)))
		mae = np.mean(np.abs(epoch_res))
		self.logger.info(f'Epoch {self.epoch} Val, MSE: {mse:.2f} MAE: {mae:.2f}, Cost {time.time() - epoch_start:.1f} sec')

		model_state_dic = self.model.state_dict()
		if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
			self.best_mse = mse
			self.best_mae = mae
			self.logger.info(f"NEW BEST: mse {self.best_mse:.2f} mae {self.best_mae:.2f} model epoch {self.epoch}")
			torch.save(model_state_dic, os.path.join(self.save_dir, f'best_model_{self.best_count}.pth'))
			self.best_count += 1

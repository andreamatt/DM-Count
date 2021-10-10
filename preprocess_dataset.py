# Preprocess images in QNRF and NWPU dataset.

import argparse

def str2bool(v):
	return str(v).lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Preprocess')
parser.add_argument('--dataset', default='qnrf', help='dataset name, only support qnrf and nwpu')
parser.add_argument('--input-path', default='data/QNRF', help='original data directory')
parser.add_argument('--output-path', default='data/QNRF-Train-Val-Test', help='processed data directory')
parser.add_argument('--augmentation', type=str2bool, default=False, help='processed data directory')
args = parser.parse_args()

if args.dataset.lower() == 'qnrf':
	from preprocess.preprocess_dataset_qnrf import main
	main(args.input_path, args.output_path, 512, 2048)
elif args.dataset.lower() == 'nwpu':
	from preprocess.preprocess_dataset_nwpu import main
	main(args.input_path, args.output_path, 384, 1920)
elif args.dataset.lower() == 'synth':
	from preprocess.preprocess_dataset_synth import main
	main(args.input_path, args.output_path, args.augmentation)
elif args.dataset.lower() == 'gcc':
	from preprocess.preprocess_dataset_gcc import main
	main(args.input_path, args.output_path)
else:
	raise NotImplementedError

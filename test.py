import argparse
import torch
import os
import numpy as np
import datasets.crowd as crowd
from models import vgg19


def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("--device", default="0", help="assign device")
    parser.add_argument(
        "--crop-size", type=int, default=512, help="the crop size of the train image"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="pretrained_models/model_qnrf.pth",
        help="saved model path",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/QNRF-Train-Val-Test",
        help="data path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="qnrf",
        help="dataset name: qnrf, nwpu, synth, gcc",
    )
    parser.add_argument(
        "--mixed", type=str2bool, default=False, help="mix dataset with synth"
    )
    parser.add_argument(
        "--synth-path", default="DATA/processed/SynthAug", help="synth path for mixing"
    )

    args = parser.parse_args()
    return args


def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # set vis gpu
    device = torch.device("cuda")

    model_path = args.model_path
    model = vgg19()
    model.to(device)
    model.load_state_dict(torch.load(model_path, device))
    model.eval()
    image_errs = []

    crop_size = args.crop_size
    data_path = args.data_path
    if args.dataset.lower() in ["qnrf", "nwpu", "gcc", "synth"]:
        dataset = crowd.Crowd(
            args.dataset.lower(),
            os.path.join(data_path, "test"),
            crop_size,
            8,
            method="test",
            mixed=args.mixed,
            mix_val=True,
            synth_path=args.synth_path,
        )
    else:
        raise NotImplementedError

    dataloader = torch.utils.data.DataLoader(
        dataset, 1, shuffle=False, num_workers=1, pin_memory=True
    )

    for inputs, count, name in dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, "the batch size should equal to 1"
        with torch.set_grad_enabled(False):
            outputs, _ = model(inputs)
        img_err = count[0].item() - torch.sum(outputs).item()

        # print(name, img_err, count[0].item(), torch.sum(outputs).item())
        image_errs.append(img_err)

    image_errs = np.array(image_errs)
    mse = np.sqrt(np.mean(np.square(image_errs)))
    mae = np.mean(np.abs(image_errs))
    print(f"{model_path}: mae {mae}, mse {mse}\n")


if __name__ == "__main__":
    main(parse_args())

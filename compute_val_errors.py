import os
import glob
import numpy as np
import argparse
from dataset import Dataset
from network import Network as CnnModel
from evaluate import read_transform_file, calculate_eRE, calculate_eTE

import torch


def make_cnn_model(model_file, backbone):
    model = CnnModel(backbone=backbone).cuda()
    model.load_state_dict(torch.load(model_file))
    return model.eval()


def eval_cnn(pred, entry, dataset_dir):
    pred_zs, pred_ys, pred_ts = pred
    index = 0

    z = pred_zs[index].cpu().numpy()
    z /= np.linalg.norm(z)

    y = pred_ys[index].cpu().numpy()
    y = y - np.dot(z, y)*z
    y /= np.linalg.norm(y)

    x = np.cross(y, z)

    pr_R = np.zeros([3,3])
    pr_R[0] = x
    pr_R[1] = y
    pr_R[2] = z

    pr_R = pr_R.T

    pr_t = pred_ts[index].cpu().numpy()

    gt_R1, gt_t = read_transform_file(os.path.join(dataset_dir, entry['txt_path']))
    gt_R2 = np.matrix.copy(gt_R1)
    gt_R2[:, :2] *= -1

    eTE = calculate_eTE(gt_t, pr_t)
    eRE = min(calculate_eRE(gt_R1, pr_R), calculate_eRE(gt_R2, pr_R))
    return eTE, eRE


def eval_model(model, dataset, dataset_dir):
    eTEs = []
    eREs = []
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            bind_input = torch.tensor(np.array([sample['xyz']])).cuda()
            cnn_output = bind_model(bind_input)
            bind_eTE, bind_eRE = eval_cnn(cnn_output, sample, dataset_dir)

            eREs.append(bind_eRE)
            eTEs.append(bind_eTE)
    return np.array(eTEs).mean(), np.array(eREs).mean()

parser = argparse.ArgumentParser()
parser.add_argument('-bb', '--backbone', type=str, default='resnet34', help='which backbone to use: resnet18/34/50')
parser.add_argument('-m', '--models_dir', type=str, default=None, help='Directory with models with .pth extension')
parser.add_argument('-d', '--dataset', type=str, default=None, help='dataset json file')
parser.add_argument('-iw', '--input_width', type=int, default=516, help='resize samples to specified width')
parser.add_argument('-ih', '--input_height', type=int, default=384, help='resize samples to specified height')
args = parser.parse_args()

bins_width = args.input_width
bins_height = args.input_height
bin_dataset_json = args.dataset
val_dataset_dir = os.path.dirname(bin_dataset_json)
bind_dataset = Dataset(bin_dataset_json, 'val', bins_width, bins_height, preload=False)

models = glob.iglob(args.models_dir + '/*.pth', recursive=False)
for model_file in models:
    print(model_file)
    bind_model = make_cnn_model(model_file=model_file, backbone=args.backbone)
    eTE, eRE = eval_model(bind_model, bind_dataset, val_dataset_dir)
    print('\tval means: eTE:', eTE, 'eRE:', eRE)

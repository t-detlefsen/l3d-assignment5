import numpy as np
import argparse

import torch
from models import cls_model
from utils import create_dir, viz_seg

from tqdm import tqdm

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')
    parser.add_argument('--noise', type=float, default=0.0, help='The variance of the gaussian noise to be added to the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    model = cls_model(args.num_cls_class).to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy(np.load(args.test_label))

    test_data += args.noise * torch.randn_like(test_data)

    # ------ Make Prediction ------
    pred_label = []
    test_data = test_data.to(args.device)
    for i in tqdm(range(len(test_data))):
        pred_label.append(torch.argmax(model(test_data[i].unsqueeze(0)), 1))

    pred_label = torch.cat(pred_label).cpu()

    # Compute Accuracy
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    print ("test accuracy: {}".format(test_accuracy))

    # Correct for each class
    for i in range(args.num_cls_class):
        ind = np.where((pred_label == test_label) * (pred_label == i))[0][0]
        viz_seg(test_data[ind].cpu(), torch.ones(len(test_data[ind])), "{}/cls_{}_{}.gif".format(args.output_dir, pred_label[ind], int(test_label[ind])), args.device)

    # Inorrect for each class
    for i in range(args.num_cls_class):
        try:
            ind = np.where((pred_label != test_label) * (test_label == i))[0][0]
            viz_seg(test_data[ind].cpu(), torch.ones(len(test_data[ind])), "{}/cls_{}_{}.gif".format(args.output_dir, pred_label[ind], int(test_label[ind])), args.device)
        except:
            continue
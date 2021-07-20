import os
import sys
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import darmo
import pooraka as prk

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='/home/mllab/proj/ILSVRC2015/Data/CLS-LOC', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--arch', type=str, default='pdarts', help='which architecture to use')
args = parser.parse_args()

def main():
  if not torch.cuda.is_available():
    sys.exit(1)

  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  cudnn.enabled=True

  model = darmo.create_model(args.arch, num_classes=1000, pretrained=True, auxiliary=True)

  model = model.cuda()

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  validdir = os.path.join(args.data, 'val')
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  valid_data = dset.ImageFolder(
    validdir,
    transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
    ]))

  valid_queue = torch.utils.data.DataLoader(
    valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

  #model.drop_path_prob = args.drop_path_prob
  valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
  print("Top 1 Acc :", valid_acc_top1)
  print("Top 5 Acc :", valid_acc_top5)

def infer(valid_queue, model, criterion):
  objs = prk.utils.AverageMeter()
  top1 = prk.utils.AverageMeter()
  top5 = prk.utils.AverageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    target = target.cuda(non_blocking=True)
    input = input.cuda(non_blocking=True)
    logits, _ = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = prk.utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

  return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
  main() 


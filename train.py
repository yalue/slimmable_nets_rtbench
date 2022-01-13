import argparse
import importlib
import os
import time
import random
import math

import torch
from torch import multiprocessing
from torchvision import datasets, transforms
import numpy as np

from config import FLAGS
from data_utils import PreloadDataset

def get_model():
    """get model"""
    model_lib = importlib.import_module(FLAGS.model)
    model = model_lib.Model(FLAGS.num_classes, input_size=FLAGS.image_size)
    model_wrapper = torch.nn.DataParallel(model).cuda()
    return model, model_wrapper

def data_transforms():
    """get transform of dataset"""
    assert(FLAGS.data_transforms == "imagenet1k_mobile")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return val_transforms


def data_loader(val_set):
    """get data loader"""
    batch_size = int(FLAGS.batch_size)
    is_cpu = str(val_set.get_device()) == "cpu"
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        pin_memory=is_cpu,
        num_workers=0,
        drop_last=getattr(FLAGS, 'drop_last', False))
    return val_loader


def set_random_seed(seed=None):
    """set random seed"""
    if seed is None:
        seed = getattr(FLAGS, 'random_seed', 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def forward_loss(model, input, target):
    """ Forward model and return the number of top-k correct results. """
    output = model(input)
    # topk
    _, pred = output.topk(max(FLAGS.topk))
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = []
    for k in FLAGS.topk:
        correct_k.append(float(correct[:k].float().sum()))
    return correct_k

def run_one_epoch(loader, model, width_mult):
    """run one epoch for train/val/test/cal"""
    model.eval()
    t_start = time.time()

    total_batches = 0
    total_processed = 0
    total_correct_k = []
    for k in FLAGS.topk:
        total_correct_k.append(0.0)

    # TODO (next): Continue making this more efficient (esp. w.r.t. keeping
    # track of # correct.) Also remove more unused code, now that criterion-
    # related stuff has been removed.
    for batch_idx, (input, target) in enumerate(loader):
        total_processed += FLAGS.batch_size
        print("Running batch %d / 5" % (total_batches + 1,))
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        correct = forward_loss(model, input, target)
        for i in range(len(FLAGS.topk)):
            total_correct_k[i] += correct[i]
        total_batches += 1
        if total_batches >= 5:
            break

    topk_correct_string = ""
    for i in range(len(FLAGS.topk)):
        k = FLAGS.topk[i]
        correct_rate = total_correct_k[i] / float(total_processed)
        topk_correct_string += "top_%d: %.03f" % (k, correct_rate)
        if i == (len(FLAGS.topk) - 1):
            break
        topk_correct_string += ", "
    print("%.03fs, width_mult %.04f: %s" % (time.time() - t_start,
        width_mult, topk_correct_string))

    return True


def train_val_test():
    """train and val"""
    torch.backends.cudnn.benchmark = True
    # seed
    set_random_seed()

    # model
    model, model_wrapper = get_model()
    if getattr(FLAGS, 'label_smoothing', 0):
        print("label_smoothing setting isn't supported")
        exit()
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
    if getattr(FLAGS, 'inplace_distill', False):
        print("inplace_distill isn't supported")
        exit()
    else:
        soft_criterion = None

    # Load pretrained weights.
    assert(getattr(FLAGS, 'pretrained', "") != "")
    checkpoint = torch.load(FLAGS.pretrained, map_location=lambda storage,
        loc: storage)
    # update keys from external models
    if type(checkpoint) == dict and 'model' in checkpoint:
        checkpoint = checkpoint['model']
    assert(not getattr(FLAGS, 'pretrained_model_remap_keys', False))
    model_wrapper.load_state_dict(checkpoint)
    print('Loaded model {}.'.format(FLAGS.pretrained))

    # if start from scratch, print model and do profiling
    print(model_wrapper)

    # data
    val_transforms = data_transforms()
    val_set = datasets.ImageFolder(os.path.join(FLAGS.dataset_dir, 'val'),
        transform=val_transforms)
    val_set = PreloadDataset(val_set, 1000, torch.device("cuda"))
    val_loader = data_loader(val_set)

    print('Start testing.')
    with torch.no_grad():
        for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
            model_wrapper.apply(
                lambda m: setattr(m, 'width_mult', width_mult))
            run_one_epoch(val_loader, model_wrapper, width_mult)
    return


def init_multiprocessing():
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="The batch size to use.",
        type=int, default=64)
    parser.add_argument("--width_mult", type=float, default=1.0,
        help="The neural network width to use. 1.0 = full width.")
    parser.add_argument("--output_file", default="", help="The name of a " +
        "JSON file to which results will be written.")
    parser.add_argument("--data_limit", default=1000, type=int,
        help="Limit on the number of data samples to load.")
    # TODO (next, 2): Use the data_limit argument here.
    init_multiprocessing()
    train_val_test()


if __name__ == "__main__":
    main()


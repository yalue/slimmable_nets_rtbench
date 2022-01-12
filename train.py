import importlib
import os
import time
import random
import math

import torch
from torch import multiprocessing
from torchvision import datasets, transforms
import numpy as np

#from utils.config import FLAGS
from config import FLAGS

def get_model():
    """get model"""
    model_lib = importlib.import_module(FLAGS.model)
    model = model_lib.Model(FLAGS.num_classes, input_size=FLAGS.image_size)
    model_wrapper = torch.nn.DataParallel(model).cuda()
    return model, model_wrapper

def data_transforms():
    """get transform of dataset"""
    if FLAGS.data_transforms in [
            'imagenet1k_basic', 'imagenet1k_inception', 'imagenet1k_mobile']:
        if FLAGS.data_transforms == 'imagenet1k_inception':
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            crop_scale = 0.08
            jitter_param = 0.4
            lighting_param = 0.1
        elif FLAGS.data_transforms == 'imagenet1k_basic':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            crop_scale = 0.08
            jitter_param = 0.4
            lighting_param = 0.1
        elif FLAGS.data_transforms == 'imagenet1k_mobile':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            crop_scale = 0.25
            jitter_param = 0.4
            lighting_param = 0.1
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transforms = val_transforms
    else:
        try:
            transforms_lib = importlib.import_module(FLAGS.data_transforms)
            return transforms_lib.data_transforms()
        except ImportError:
            raise NotImplementedError(
                'Data transform {} is not yet implemented.'.format(
                    FLAGS.data_transforms))
    return None, val_transforms, test_transforms


def data_loader(val_set):
    """get data loader"""
    batch_size = int(FLAGS.batch_size)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        pin_memory=True,
        num_workers=1,
        drop_last=getattr(FLAGS, 'drop_last', False))
    return val_loader

def get_optimizer(model):
    """get optimizer"""
    if FLAGS.optimizer == 'sgd':
        # all depthwise convolution (N, 1, x, x) has no weight decay
        # weight decay only on normal conv and fc
        model_params = []
        for params in model.parameters():
            ps = list(params.size())
            if len(ps) == 4 and ps[1] != 1:
                weight_decay = FLAGS.weight_decay
            elif len(ps) == 2:
                weight_decay = FLAGS.weight_decay
            else:
                weight_decay = 0
            item = {'params': params, 'weight_decay': weight_decay,
                    'lr': FLAGS.lr, 'momentum': FLAGS.momentum,
                    'nesterov': FLAGS.nesterov}
            model_params.append(item)
        optimizer = torch.optim.SGD(model_params)
    else:
        try:
            optimizer_lib = importlib.import_module(FLAGS.optimizer)
            return optimizer_lib.get_optimizer(model)
        except ImportError:
            raise NotImplementedError(
                'Optimizer {} is not yet implemented.'.format(FLAGS.optimizer))
    return optimizer


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


def get_conv_layers(m):
    layers = []
    if (isinstance(m, torch.nn.Conv2d) and hasattr(m, 'width_mult') and
            getattr(m, 'us', [False, False])[1] and
            not getattr(m, 'depthwise', False) and
            not getattr(m, 'linked', False)):
        layers.append(m)
    for child in m.children():
        layers += get_conv_layers(child)
    return layers


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
    checkpoint = torch.load(FLAGS.pretrained, map_location=lambda storage,
        loc: storage)
    # update keys from external models
    if type(checkpoint) == dict and 'model' in checkpoint:
        checkpoint = checkpoint['model']
    if getattr(FLAGS, 'pretrained_model_remap_keys', False):
        new_checkpoint = {}
        new_keys = list(model_wrapper.state_dict().keys())
        old_keys = list(checkpoint.keys())
        for key_new, key_old in zip(new_keys, old_keys):
            new_checkpoint[key_new] = checkpoint[key_old]
            print('remap {} to {}'.format(key_new, key_old))
        checkpoint = new_checkpoint
    model_wrapper.load_state_dict(checkpoint)
    print('Loaded model {}.'.format(FLAGS.pretrained))

    # if start from scratch, print model and do profiling
    print(model_wrapper)

    # data
    train_transforms, val_transforms, test_transforms = data_transforms()
    val_set = datasets.ImageFolder(os.path.join(FLAGS.dataset_dir, 'val'),
        transform=val_transforms)
    val_loader = data_loader(val_set)

    print('Start testing.')
    with torch.no_grad():
        for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
            model_wrapper.apply(
                lambda m: setattr(m, 'width_mult', width_mult))
            run_one_epoch(val_loader, model_wrapper, width_mult)
    return


def init_multiprocessing():
    # print(multiprocessing.get_start_method())
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass


def validate_flags():
    """ Asserts that a few flags are sane before doing anything else. """
    # TODO: Remove some of these asserts after all references to the config
    # fields have been removed.
    # We only support the imagenet1k dataset.
    assert(FLAGS.dataset == 'imagenet1k')
    assert(FLAGS.data_loader == 'imagenet1k_basic')
    # We require testing only.
    assert(gettattr(FLAGS, 'pretrained', "") != "")
    assert(FLAGS.test_only)
    # We only support a single GPU.
    assert(FLAGS.num_gpus_per_job == 1)
    # We don't support autoslim.
    assert(getattr(FLAGS, 'autoslim', False) == False)
    # We only support universally slimmable
    assert(FLAGS.universally_slimmable_training)
    assert(FLAGS.slimmable_training)

def main():
    """train and eval model"""
    init_multiprocessing()
    train_val_test()


if __name__ == "__main__":
    main()

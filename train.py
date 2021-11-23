import importlib
import os
import time
import random
import math

import torch
from torch import multiprocessing
from torchvision import datasets, transforms
import numpy as np

from utils.model_profiling import model_profiling
from utils.transforms import Lighting
from utils.distributed import init_dist, master_only, is_master
from utils.distributed import get_rank
from utils.distributed import dist_all_reduce_tensor
from utils.distributed import master_only_print as print
from utils.distributed import AllReduceDistributedDataParallel, allreduce_grads
from utils.loss_ops import CrossEntropyLossSoft, CrossEntropyLossSmooth
from models.slimmable_ops import bn_calibration_init
from utils.config import FLAGS
from utils.meters import ScalarMeter, flush_scalar_meters


def get_model():
    """get model"""
    model_lib = importlib.import_module(FLAGS.model)
    model = model_lib.Model(FLAGS.num_classes, input_size=FLAGS.image_size)
    if getattr(FLAGS, 'distributed', False):
        gpu_id = init_dist()
        if getattr(FLAGS, 'distributed_all_reduce', False):
            # seems faster
            model_wrapper = AllReduceDistributedDataParallel(model.cuda())
        else:
            model_wrapper = torch.nn.parallel.DistributedDataParallel(
                model.cuda(), [gpu_id], gpu_id)
    else:
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
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
            transforms.ColorJitter(
                brightness=jitter_param, contrast=jitter_param,
                saturation=jitter_param),
            Lighting(lighting_param),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
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
    return train_transforms, val_transforms, test_transforms


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


@master_only
def get_meters(phase):
    """util function for meters"""
    def get_single_meter(phase, suffix=''):
        meters = {}
        meters['loss'] = ScalarMeter('{}_loss/{}'.format(phase, suffix))
        for k in FLAGS.topk:
            meters['top{}_error'.format(k)] = ScalarMeter(
                '{}_top{}_error/{}'.format(phase, k, suffix))
        if phase == 'train':
            meters['lr'] = ScalarMeter('learning_rate')
        return meters

    assert phase in ['train', 'val', 'test', 'cal'], 'Invalid phase.'
    if getattr(FLAGS, 'slimmable_training', False):
        meters = {}
        for width_mult in FLAGS.width_mult_list:
            meters[str(width_mult)] = get_single_meter(phase, str(width_mult))
    else:
        meters = get_single_meter(phase)
    if phase == 'val':
        meters['best_val'] = ScalarMeter('best_val')
    return meters


def forward_loss(
        model, criterion, input, target, meter, soft_target=None,
        soft_criterion=None, return_soft_target=False, return_acc=False):
    """forward model and return loss"""
    output = model(input)
    if soft_target is not None:
        loss = torch.mean(soft_criterion(output, soft_target))
    else:
        loss = torch.mean(criterion(output, target))
    # topk
    _, pred = output.topk(max(FLAGS.topk))
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = []
    for k in FLAGS.topk:
        correct_k.append(correct[:k].float().sum(0))
    tensor = torch.cat([loss.view(1)] + correct_k, dim=0)
    # allreduce
    tensor = dist_all_reduce_tensor(tensor)
    # cache to meter
    tensor = tensor.cpu().detach().numpy()
    bs = (tensor.size-1)//2
    for i, k in enumerate(FLAGS.topk):
        error_list = list(1.-tensor[1+i*bs:1+(i+1)*bs])
        if return_acc and k == 1:
            top1_error = sum(error_list) / len(error_list)
            return loss, top1_error
        if meter is not None:
            meter['top{}_error'.format(k)].cache_list(error_list)
    if meter is not None:
        meter['loss'].cache(tensor[0])
    if return_soft_target:
        return loss, torch.nn.functional.softmax(output, dim=1)
    return loss


def run_one_epoch(
        loader, model, criterion, meters, phase='train',
        soft_criterion=None):
    """run one epoch for train/val/test/cal"""
    t_start = time.time()
    assert phase in ['val', 'test'], 'Invalid phase.'
    model.eval()
    total_batches = 0

    for batch_idx, (input, target) in enumerate(loader):
        print("Running batch %d / 5" % (total_batches + 1,))
        target = target.cuda(non_blocking=True)
        # TODO (next): Don't set width_mult here, since it is set in train_val_test
        for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
            model.apply(
                lambda m: setattr(m, 'width_mult', width_mult))
            if is_master():
                meter = meters[str(width_mult)]
            else:
                meter = None
            forward_loss(model, criterion, input, target, meter)
        total_batches += 1
        if total_batches >= 5:
            break

    if is_master() and getattr(FLAGS, 'slimmable_training', False):
        for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
            results = flush_scalar_meters(meters[str(width_mult)])
            print('{:.1f}s\t{}\t{}: '.format(time.time() - t_start, phase,
                str(width_mult)) +
                ', '.join('{}: {:.3f}'.format(
                    k, v) for k, v in results.items()))
    else:
        results = None
    return results


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
        criterion = CrossEntropyLossSmooth(reduction='none')
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
    if getattr(FLAGS, 'inplace_distill', False):
        soft_criterion = CrossEntropyLossSoft(reduction='none')
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
    test_meters = get_meters('test')
    with torch.no_grad():
        # TODO: This is stupid for testing, this width_mult is ignored in
        # run_one_epoch.
        for width_mult in sorted(FLAGS.width_mult_list, reverse=True):
            model_wrapper.apply(
                lambda m: setattr(m, 'width_mult', width_mult))
            run_one_epoch(
                val_loader, model_wrapper, criterion,
                test_meters, phase='test')
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

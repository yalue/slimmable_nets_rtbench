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
from data_utils import PreloadDataset, SimpleLoader

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
    val_loader = SimpleLoader(val_set, batch_size)
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

def single_job(input, target, model):
    """ Requires the input batch and target labels, as well as the model to
    evaluate. Returns an array of correct classifications. The i'th entry of
    the returned array corresponds to the i'th value in FLAGS.topk. Expects the
    input and target data to be on the CPU, but the model to be on the GPU. """
    input = input.cuda(non_blocking = True)
    target = target.cuda(non_blocking = True)
    correct = forward_loss(model, input, target)
    return correct

def run_test(loader, model, args):
    """ Runs the number of batches specified in the args. Returns a tuple:
        ([correct_k values], total_processed). """
    model.eval()

    jobs_completed = 0
    total_processed = 0
    total_correct_k = []
    job_times = np.full((args.job_count,), 100.0)
    for k in FLAGS.topk:
        total_correct_k.append(0.0)

    # Run two batches as a warmup
    for batch_idx, (input, target) in enumerate(loader):
        time_1 = time.perf_counter()
        print("Running warmup batch %d" % (batch_idx + 1,))
        single_job(input, target, model)
        time_2 = time.perf_counter()
        print("Running warmup batch %d took %f seconds" % (batch_idx + 1,
            time_2 - time_1))
        if batch_idx >= 1:
            break
    print("Warmup done")

    start_time = time.perf_counter()
    # TODO: Several things
    #  - Wait for task system release
    #  - Wait for a job at the end of everything
    #  - Set cost estimate based on warmup time. (Increase to 4 warmup batches;
    #    take average of final 3?)
    #  - Use a stream.
    #  - Check and update correctness after each job.
    batch_index = 0
    batch_count = len(loader)
    batch_enumerator = enumerate(loader)
    print("Number of available batches: " + str(batch_count))
    while jobs_completed < args.job_count:
        # Reset the enumerator if we're out of batches.
        if batch_index == batch_count:
            batch_enumerator = enumerate(loader)
            batch_index = 0
        batch_index, (input, target) = next(batch_enumerator)
        total_processed += FLAGS.batch_size
        print("Running job %d / %d" % (jobs_completed + 1, args.job_count))
        job_start_time = time.perf_counter()

        correct = single_job(input, target, model)
        for i in range(len(FLAGS.topk)):
            total_correct_k[i] += correct[i]

        job_end_time = time.perf_counter()
        job_times[jobs_completed] = job_end_time - job_start_time
        jobs_completed += 1
        elapsed = job_end_time - start_time
        if (args.time_limit > 0) and (elapsed > args.time_limit):
            print("Time limit exceeded.")
            break
        if jobs_completed >= args.job_count:
            print("All jobs completed.")
            break

    return (total_correct_k, total_processed)

def correct_k_string(results):
    """ Takes the results returned by run_test and returns the top-k
    correctness formatted as a human-readable string. """
    to_return = ""
    for i in range(len(FLAGS.topk)):
        k = FLAGS.topk[i]
        correct_rate = results[0][i] / float(results[1])
        to_return += "top_%d: %.03f" % (k, correct_rate)
        if i < (len(FLAGS.topk) - 1):
            to_return += ","
    return to_return

def train_val_test(args):
    assert(not getattr(FLAGS, 'label_smoothing', False))
    assert(not getattr(FLAGS, 'inplace_distill', False))
    assert(not getattr(FLAGS, 'pretrained_model_remap_keys', False))
    assert(args.width_mult in FLAGS.width_mult_list)
    torch.backends.cudnn.benchmark = True
    set_random_seed()
    model, model_wrapper = get_model()
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    # Load pretrained weights.
    assert(getattr(FLAGS, 'pretrained', "") != "")
    checkpoint = torch.load(FLAGS.pretrained, map_location=lambda storage,
        loc: storage)
    # update keys from external models
    if type(checkpoint) == dict and 'model' in checkpoint:
        checkpoint = checkpoint['model']
    model_wrapper.load_state_dict(checkpoint)
    print('Loaded model {}.'.format(FLAGS.pretrained))

    # data
    val_transforms = data_transforms()
    val_set = datasets.ImageFolder(os.path.join(FLAGS.dataset_dir, 'val'),
        transform=val_transforms)
    val_set = PreloadDataset(val_set, args.data_limit, torch.device("cpu"))
    val_loader = data_loader(val_set)

    print("Running test using width mult %f" % (args.width_mult,))
    with torch.no_grad():
        model_wrapper.apply(lambda m: setattr(m, "width_mult", args.width_mult))
        start_time = time.perf_counter()
        results = run_test(val_loader, model_wrapper, args)
        end_time = time.perf_counter()
        topk_string = correct_k_string(results)
        print("Width mult %.04f took %.03fs. %s" % (args.width_mult,
            end_time - start_time, topk_string))
    return None


def init_multiprocessing():
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="The batch size to use.",
        type=int, default=64)
    parser.add_argument("--job_count", help="The number of jobs to launch.",
        type=int, default=100)
    parser.add_argument("--time_limit", type=float, default=-1,
        help="A limit on the number of seconds to run. Negative = unlimited.")
    parser.add_argument("--width_mult", type=float, default=1.0,
        help="The neural network width to use. 1.0 = full width.")
    parser.add_argument("--output_file", default="", help="The name of a " +
        "JSON file to which results will be written.")
    parser.add_argument("--data_limit", default=1000, type=int,
        help="Limit on the number of data samples to load.")
    args = parser.parse_args()
    FLAGS.batch_size = args.batch_size
    init_multiprocessing()
    train_val_test(args)

if __name__ == "__main__":
    main()


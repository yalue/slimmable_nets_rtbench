# Attempt to prevent numpy multithreading.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MIOPEN_COMPILE_PARALLEL_LEVEL"] = "1"
import threadpoolctl
threadpoolctl.threadpool_limits(1)

import argparse
import importlib
import json
import time

import kfmlp_control
import rocm_helper
import partitioned_streams
import data_utils
from config import FLAGS

import torch
import numpy as np

def get_mmapped_ndarray(filename, shape, dtype):
    """ Returns a numpy ndarray with the content of the named file and the
    given shape. """
    import mmap
    f = open(filename, "r+b")
    prot = mmap.PROT_READ | mmap.PROT_WRITE
    mm = mmap.mmap(f.fileno(), 0, flags=mmap.MAP_SHARED, prot=prot)
    f.close()
    a = np.frombuffer(mm, dtype=dtype)
    return a.reshape(shape)

def get_model():
    """get model"""
    model_lib = importlib.import_module(FLAGS.model)
    model = model_lib.Model(FLAGS.num_classes, input_size=FLAGS.image_size)
    model_wrapper = torch.nn.DataParallel(model).cuda()
    return model, model_wrapper

def data_transforms():
    from torchvision import transforms
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
    val_loader = data_utils.SimpleLoader(val_set, batch_size)
    return val_loader

def forward_loss(model, input, target, correct_k):
    """ Forward model and fills in the correct-k results. """
    output = model(input)
    # topk
    _, pred = output.topk(max(FLAGS.topk))
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    for i in range(len(FLAGS.topk)):
        k = FLAGS.topk[i]
        correct_k[i] = float(correct[:k].float().sum())
    return None

def single_job(input, target, model, correct_k):
    """ Requires the input batch and target labels, as well as the model to
    evaluate. Fills in the correct_k array so that the i'th entry of
    corresponds to the i'th value in FLAGS.topk. Expects the input and target
    data to be on the CPU, but the model to be on the GPU. """
    input = input.cuda(non_blocking = True)
    target = target.cuda(non_blocking = True)
    correct = forward_loss(model, input, target, correct_k)
    return None

class TaskStatistics:
    """ Used to keep track of statistics (correctness, etc) for jobs of a
    single task. """

    def __init__(self, args, topk):
        self.args = args
        self.topk = topk
        self.total_jobs_complete = 0
        self.jobs_completed_on_time = 0
        self.total_job_time = 0.0
        self.images_analyzed = 0
        self.images_analyzed_on_time = 0
        self.job_start_time = 0.0
        self.min_job_time = 1.0e10
        self.max_job_time = -1.0e10
        self.mean_job_time = 0.0
        self.median_job_time = 0.0
        self.job_time_std_dev = 0.0
        self.total_correct_k = np.full((len(topk),), 0.0, dtype="float32")
        self.correct_k_no_late = np.full((len(topk),), 0.0, dtype="float32")
        job_times_count = args.job_count
        if (job_times_count <= 0) or (job_times_count > args.max_job_times):
            job_times_count = args.max_job_times
        self.job_times = np.full((job_times_count,), 100.0, dtype="float32")
        self.last_job_duration = 0.0

    def starting_job(self):
        """ To be called prior to starting a job's computation. """
        self.job_start_time = time.perf_counter()

    def finished_job(self, correct_k):
        """ To be called when a job completes. Requires the topk array returned
        by forward_loss. """
        end_time = time.perf_counter()
        duration = end_time - self.job_start_time
        if self.total_jobs_complete < len(self.job_times):
            self.job_times[self.total_jobs_complete] = duration
        self.total_jobs_complete += 1
        self.total_job_time += duration
        self.images_analyzed += self.args.batch_size
        self.total_correct_k += correct_k
        self.last_job_duration = duration

        # Record some info depending on whether we completed on time.
        # Everything's "on time" if no deadline was specified.
        relative_dl = self.args.relative_deadline
        if (relative_dl <= 0) or (duration <= relative_dl):
            self.images_analyzed_on_time += self.args.batch_size
            self.correct_k_no_late += correct_k
            self.jobs_completed_on_time += 1

    def add_missed_deadlines(self, missed_count):
        """ To be called with the number of missed deadlines every time a job
        overran one or more deadlines. (Does nothing if missed_count is 0.) """
        self.total_jobs_complete += missed_count

    def get_last_job_duration(self):
        """ Returns the amount of time required by the previous job. """
        return self.last_job_duration

    def all_jobs_completed(self):
        """ Returns true if the number of jobs specified in the args has been
        completed. """
        if self.args.job_count <= 0:
            # We don't have a limit on jobs
            return False
        return self.total_jobs_complete >= self.args.job_count

    def average_job_time(self):
        """ Returns the average amount of time a job has taken so far. """
        return self.total_job_time / float(self.total_jobs_complete)

    def correct_k_string(self):
        """ Returns a human-readable string of the top-k correctness results.
        """
        to_return = ""
        analyzed = float(self.images_analyzed)
        for i in range(len(self.topk)):
            k = self.topk[i]
            analyzed = self.images_analyzed
            correct_rate = self.total_correct_k[i] / analyzed
            to_return += "top_%d: %.03f" % (k, correct_rate)
            if i < (len(self.topk) - 1):
                to_return += ", "
        return to_return

    def compute_stats(self):
        """ Sets the min, max, mean, and std. dev. of job times. """
        if len(self.job_times) == 0:
            return
        # Cast these to floats to avoid issues with JSON serialization of numpy
        # float types.
        self.min_job_time = float(min(self.job_times))
        self.max_job_time = float(max(self.job_times))
        self.median_job_time = float(np.median(self.job_times))
        self.mean_job_time = float(np.mean(self.job_times))
        self.job_time_std_dev = float(np.std(self.job_times))

    def write_to_file(self):
        """ Writes the data contained in this object to the named JSON file
        in args.output_file. Does nothing if no output file was specified. """

        def dumper(obj):
            """ Small helper so we can JSON serialize numpy arrays. """
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return vars(obj)

        if self.args.output_file == "":
            print("No output file specified.")
            return
        print("Writing results to " + str(self.args.output_file))
        # Temporarily truncate the job_times array to the number of jobs
        # actually completed (if necessary).
        old_job_times = self.job_times
        if self.total_jobs_complete < len(self.job_times):
            self.job_times = self.job_times[0:self.total_jobs_complete]
        self.compute_stats()
        with open(self.args.output_file, "w") as f:
            json.dump(vars(self), f, indent="  ", default=dumper)
        print("Wrote output to " + self.args.output_file)
        self.job_times = old_job_times

def sleep_next_period(last_cost, relative_deadline):
    """ Sleeps until the next period boundary, assuming relative deadlines
    equal periods. Returns the number of missed deadlines, if any. """
    missed_deadlines = 0
    while last_cost > relative_deadline:
        missed_deadlines += 1
        last_cost -= relative_deadline
    time.sleep(relative_deadline - last_cost)
    return missed_deadlines

def run_test(loader, model, args):
    """ Runs the number of batches specified in the args. Returns a
    TaskStatistics object. """
    model.eval()

    statistics = TaskStatistics(args, FLAGS.topk)
    correct_k = np.full((len(FLAGS.topk),), 0.0)

    # Run two batches as a warmup
    for batch_idx, (input, target) in enumerate(loader):
        time_1 = time.perf_counter()
        print("Running warmup batch %d" % (batch_idx + 1,))
        single_job(input, target, model, correct_k)
        time_2 = time.perf_counter()
        print("Running warmup batch %d took %f seconds" % (batch_idx + 1,
            time_2 - time_1))
        if batch_idx >= 1:
            break

    # Make sure we occasionally suspend after this!
    #kfmlp_control.start_sched_fifo()

    lock_od = None
    streams = None
    stream = torch.cuda.default_stream()
    if args.use_partitioned_streams:
        k = kfmlp_control.get_k()
        streams = partitioned_streams.streams_for_partitions(k)
    batch_index = 0
    batch_count = len(loader)
    batch_enumerator = enumerate(loader)
    print("Number of available batches: " + str(batch_count))
    torch.cuda.synchronize()

    if args.wait_for_ts_release:
        print("Waiting to be released.")
        kfmlp_control.wait_for_ts_release()

    start_time = time.perf_counter()
    elapsed_time = 0.0
    jobs_missed = 0

    while True:
        # Reset the enumerator if we're out of batches.
        if batch_index == (batch_count - 1):
            batch_enumerator = enumerate(loader)
            batch_index = 0
        batch_index, (input, target) = next(batch_enumerator)
        statistics.add_missed_deadlines(jobs_missed)

        elapsed_time = time.perf_counter() - start_time
        print("%.02f/%.02fs: Running job %d / %d" % (elapsed_time,
            args.time_limit, statistics.total_jobs_complete + 1,
            args.job_count))
        lock_slot = 0
        if args.use_locking:
            kfmlp_control.acquire_lock()
        if args.use_partitioned_streams:
            stream = streams[lock_slot]
        statistics.starting_job()
        with torch.cuda.stream(stream):
            single_job(input, target, model, correct_k)
        stream.synchronize()
        if args.use_locking:
            kfmlp_control.release_lock()
        statistics.finished_job(correct_k)

        if (args.time_limit > 0) and (elapsed_time > args.time_limit):
            print("Time limit reached.")
            break
        if statistics.all_jobs_completed():
            print("Job limit reached.")
            break
        if args.relative_deadline > 0:
            jobs_missed = sleep_next_period(statistics.get_last_job_duration(),
                args.relative_deadline)

    #kfmlp_control.end_sched_fifo()
    return statistics

def validate_args(args):
    """ Exits and prints a message if any of the given args are incompatible.
    """
    if args.max_job_times <= 0:
        print("Must be able to store a positive number of job times.")
        exit(1)
    if args.use_partitioned_streams and not args.use_locking:
        print("Partitioned streams require k-exclusion locking.")
        exit(1)

def train_val_test(args, input_ndarray=None, result_ndarray=None):
    """ This takes the command-line args object (or similar), and possibly two
    numpy ndarrays. If provided, the ndarrays are used in lieu of loading files
    from disk for the testing dataset. """
    assert(not getattr(FLAGS, 'label_smoothing', False))
    assert(not getattr(FLAGS, 'inplace_distill', False))
    assert(not getattr(FLAGS, 'pretrained_model_remap_keys', False))
    assert(args.width_mult in FLAGS.width_mult_list)
    # Disable pytorch multithreading, hopefully.
    torch.set_num_threads(1)
    threadpoolctl.threadpool_limits(1)
    # If train_val_test was called in a child process, we need to open a new
    # file handle to the KFMLP kernel module's chardev.
    kfmlp_control.reset_module_handle()
    validate_args(args)
    FLAGS.batch_size = args.batch_size
    torch.backends.cudnn.benchmark = True
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
    val_loader = None
    if input_ndarray is None:
        val_transforms = data_transforms()
        from torchvision import datasets
        val_set = datasets.ImageFolder(os.path.join(FLAGS.dataset_dir, 'val'),
            transform=val_transforms)
        val_set = data_utils.PreloadDataset(val_set, args.data_limit,
            torch.device("cpu"))
        val_loader = data_loader(val_set)
    else:
        assert(result_ndarray is not None)
        val_set = data_utils.BufferDataset(input_ndarray, result_ndarray)
        val_loader = data_loader(val_set)

    print("Running test using width mult %f" % (args.width_mult,))
    results = None
    with torch.no_grad():
        model_wrapper.apply(lambda m: setattr(m, "width_mult", args.width_mult))
        start_time = time.perf_counter()
        results = run_test(val_loader, model_wrapper, args)
        end_time = time.perf_counter()
        topk_string = results.correct_k_string()
        print("Width mult %.04f took %.03fs. %s" % (args.width_mult,
            end_time - start_time, topk_string))
    results.write_to_file()
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="The batch size to use.",
        type=int, default=64)
    parser.add_argument("--job_count", help="The number of jobs to launch. " +
        "0 = unlimited.", type=int, default=100)
    parser.add_argument("--time_limit", type=float, default=-1,
        help="A limit on the number of seconds to run. Negative = unlimited.")
    parser.add_argument("--width_mult", type=float, default=1.0,
        help="The neural network width to use. 1.0 = full width.")
    parser.add_argument("--output_file", default="", help="The name of a " +
        "JSON file to which results will be written.")
    parser.add_argument("--data_limit", default=1000, type=int,
        help="Limit on the number of data samples to load. Ignored if "+
            "--use_data_blobs is set.")
    parser.add_argument("--use_data_blobs", action="store_true",
        help="If set, use input_data_raw.bin and result_data_raw.bin instead" +
            " of the torchvision dataset. Only used when running " +
            "rtbenchmark.py directly.")
    parser.add_argument("--num_competitors", type=int, default=1,
        help="The total number of competing tasks including this one.")
    parser.add_argument("--task_index", type=int, default=0,
        help="This task's ID if multiple competitors are running.")
    parser.add_argument("--experiment_name", type=str, default="",
        help="A string identifying this experiment. Basically just to be " +
            "copied to the output file.")
    parser.add_argument("--task_system_index", type=int, default=0,
        help="The number of this task's task system in this experiment.")

    parser.add_argument("--max_job_times", type=int, default=10000,
        help="The maximum number of job times to record.")
    parser.add_argument("--wait_for_ts_release", action="store_true",
        help="If set, wait for tasks to be released. (Uses the KFMLP module.)")
    parser.add_argument("--relative_deadline", default=-1.0, type=float,
        help="Each real-time job's relative deadline, in seconds.")
    parser.add_argument("--use_locking", action="store_true",
        help="If set, use k-exclusion locking. Some other process must have " +
            "already set the max K value with the KFMLP module.")
    parser.add_argument("--use_partitioned_streams", action="store_true",
        help="If set, and k-exclusion locking is used, then require each " +
            "task to run GPU work on the stream corresponding to its lock " +
            "slot.")

    args = parser.parse_args()
    data_blob = None
    result_blob = None
    if args.use_data_blobs:
        print("Using pre-computed data blobs.")
        data_blob = get_mmapped_ndarray("input_data_raw.bin",
            (-1, 3, 224, 224), "float32")
        result_blob = get_mmapped_ndarray("result_data_raw.bin",
            (-1,), "int64")
    train_val_test(args, input_ndarray=data_blob, result_ndarray=result_blob)

if __name__ == "__main__":
    main()


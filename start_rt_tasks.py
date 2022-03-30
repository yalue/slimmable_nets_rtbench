# Attempt to disable numpy multithreading.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MIOPEN_COMPILE_PARALLEL_LEVEL"] = "1"
import threadpoolctl
threadpoolctl.threadpool_limits(1)

import copy
import mmap
import multiprocessing
import numpy
import random
import time

import kfmlp_control

def width_mults_and_batch_sizes():
    width_mult_list = [0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45,
        0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725,
        0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0]
    batch_size_list = [1, 2, 4, 8, 16, 32, 64, 128]
    return width_mult_list, batch_size_list

def get_mmapped_ndarray(filename, shape, dtype):
    """ Returns a numpy ndarray with the content of the named file and the
    given shape. """
    f = open(filename, "r+b")
    prot = mmap.PROT_READ | mmap.PROT_WRITE
    mm = mmap.mmap(f.fileno(), 0, flags=mmap.MAP_SHARED, prot=prot)
    f.close()
    a = numpy.frombuffer(mm, dtype=dtype)
    return a.reshape(shape)

def page_in_ndarray(array):
    """ Reads every 512 entries from the given numpy array. The return value
    is arbitrary and can be ignored. Intended to be used to page an entire
    mmapped file into memory.
    """
    flattened = array.flatten()
    i = 0
    junk = 0.0
    while i < len(flattened):
        junk += flattened[i]
        i += 512
    return junk

def load_dataset():
    """ Loads the existing dataset blobs from disk. The files must already be
    present for this to succeed. Returns the input and result numpy ndarrays,
    respectively. (The child processes should convert them to torch Tensors.)
    """
    print("Loading input dataset.")
    input_numpy = get_mmapped_ndarray("input_data_raw_small.bin", (-1, 3, 224, 224),
        "float32")
    page_in_ndarray(input_numpy)
    print("Loading result dataset.")
    result_numpy = get_mmapped_ndarray("result_data_raw_small.bin", (-1,), "int64")
    page_in_ndarray(result_numpy)
    return (input_numpy, result_numpy)

class FakeArgs:
    """ The rtbenchmark.py functionality expects command-line args for plenty
    of things, so this class mimics the attributes it may expect from the
    command-line args. """
    def __init__(self, config):
        self.batch_size = 64
        self.job_count = 100
        self.time_limit = -1
        self.width_mult = 1.0
        self.output_file = ""
        self.data_limit = 1000
        self.max_job_times = 10000
        self.wait_for_ts_release = True
        self.use_locking = False
        self.use_partitioned_streams = False
        self.num_competitors = 1
        self.task_index = 0
        self.experiment_name = ""
        # No deadline by default.
        self.relative_deadline = -1.0
        for key in config:
            setattr(self, key, config[key])

def launch_single_task(input_data, result_data, config):
    """ Intended to be launched in a new child process. Requires the input and
    output dataset numpy tensors, and the specific config for the task. """
    # We must NOT import rtbenchmark in the parent process; it imports pytorch
    # which does not play well with child processes. We want to initialize
    # pytorch ONLY in the child processes.
    import rtbenchmark
    args = FakeArgs(config)
    rtbenchmark.train_val_test(args, input_ndarray = input_data,
        result_ndarray = result_data)

def wait_for_ready_tasks(count, timeout):
    """ Waits until the number of task-sysytem release waiters is at least the
    given count. Returns True when this happens, or False if the timeout occurs
    first. """
    waiting = kfmlp_control.get_ts_waiter_count()
    start_time = time.perf_counter()
    while waiting < count:
        cur_time = time.perf_counter()
        if (cur_time - start_time) > timeout:
            return False
        time.sleep(0.1)
        waiting = kfmlp_control.get_ts_waiter_count()
    return True

def run_all_kernel_configs(input_dataset, result_dataset):
    """ Runs one task with each possible batch size and width multiplier.
    Necessary for generating cached kernel code used by AMD's software. Make
    sure to run a program with this function before actually doing any tests.
    """
    width_mult_list, batch_size_list = width_mults_and_batch_sizes()
    i = 0
    for bs in batch_size_list:
        for wm in width_mult_list:
            result_filename = "results/isolated_all_%d.json" % (i,)
            config = {
                "batch_size": bs,
                "width_mult": wm,
                "job_count": 100,
                "output_file": result_filename,
                "relative_deadline": -1.0,
            }
            print("Running test %d: with width mult %.03f, batch size %d" % (i,
                wm, bs))
            i += 1
            start_time = 0.0
            p = None
            while True:
                start_time = time.perf_counter()
                p = multiprocessing.Process(target=launch_single_task,
                    args=(input_dataset, result_dataset, config))
                p.start()
                if wait_for_ready_tasks(1, 120.0):
                    break
                print("Timed out waiting for task to be ready.")
                p.terminate()
                time.sleep(5.0)
                p.close()
                p = None
                continue
            kfmlp_control.release_ts()
            p.join()
            end_time = time.perf_counter()
            print("Test with width_mult %.03f, batch size %d took %.03fs" % (
                wm, bs, end_time - start_time))
    return True

def estimate_cost(width_mult, batch_size, competitor_count):
    # TODO (next, 2): Implement estimate_cost a bit better. Use real data.
    return 0.1

def deadline_from_cost(cost, width_mult, batch_size, competitor_count):
    """ Returns a random period/deadline given a cost and task params. """
    # TODO: Randomize the deadline a bit better. For now it just returns a
    # multiple of cost between 2.0 and (2.0 + competitor_count)
    cost_multiplier = 2.0 + (float(competitor_count) * random.random())
    return cost * cost_multiplier

def random_config(experiment_name, task_system_index, task_index,
    num_competitors):
    """ Returns an args object for a single task with a randomly selected width
    mult, batch size, and cost/period estimate. """
    width_mult_list, batch_size_list = width_mults_and_batch_sizes()
    width_mult = random.choice(width_mult_list)
    batch_size = random.choice(batch_size_list)
    cost = estimate_cost(width_mult, batch_size, num_competitors)
    deadline = deadline_from_cost(cost, width_mult, batch_size,
        num_competitors)
    output_filename = "results/%s_%d_%d.json" % (experiment_name,
        task_system_index, task_index)
    config = {
        "num_competitors": num_competitors,
        "task_index": task_index,
        "experiment_name": experiment_name,
        "task_system_index": task_system_index,
        "output_file": output_filename,
        "relative_deadline": deadline,
        "time_limit": 120.0,
        "job_count": 0,
        "batch_size": batch_size,
        "width_mult": width_mult
    }
    return config

def unpartitioned_competitors(competitor_count, task_system_count):
    """ Runs task systems where all tasks share the same GPU with no
    restrictions. Returns a list of lists of task configs. """
    experiment_name = "unpartitioned" + str(competitor_count)
    to_return = []
    for i in range(task_system_count):
        task_system = []
        for j in range(competitor_count):
            task = random_config(experiment_name, i, j, competitor_count)
            task_system.append(task)
        to_return.append(task_system)
    return to_return

def compare_6way_sharing():
    """ Returns a set of scenarios, each of which allows 6 tasks to share the
    GPU. """
    base_task = {
        "num_competitors": 6,
        "task_system_index": 0,
        "relative_deadline": -1.0,
        "time_limit": 60.0 * 5.0,
        "job_count": 0,
        "batch_size": 16,
    }
    task1 = copy.deepcopy(base_task)
    task1["width_mult"] = 1.0
    task2 = copy.deepcopy(base_task)
    task2["width_mult"] = 0.75
    task3 = copy.deepcopy(base_task)
    task3["width_mult"] = 0.5
    task4 = copy.deepcopy(base_task)
    task4["batch_size"] = 8
    task4["width_mult"] = 1.0
    task5 = copy.deepcopy(base_task)
    task5["batch_size"] = 8
    task5["width_mult"] = 0.75
    task6 = copy.deepcopy(base_task)
    task6["batch_size"] = 8
    task6["width_mult"] = 0.5
    base_task_system = [task1, task2, task3, task4, task5, task6]

    def make_experiment(name, use_locking, use_partitioned_streams):
        scenario = copy.deepcopy(base_task_system)
        for i in range(len(scenario)):
            t = scenario[i]
            t["experiment_name"] = name
            t["task_index"] = i
            t["output_file"] = "results/%s_%d.json" % (name, i)
            t["use_locking"] = use_locking
            t["use_partitioned_streams"] = use_partitioned_streams
        return scenario

    to_return = [
        make_experiment("6way_unmanaged", False, False),
        make_experiment("6way_locked_exclusive", True, False),
        make_experiment("6way_locked_2shared", True, False),
        make_experiment("6way_locked_2partitioned", True, True),
    ]
    return to_return


# TODO: Use a shared buffer to allow run_task_system to monitor child tasks
# for activity after they've started up.
def run_task_system(tasks, input_data, result_data):
    ready_count = 0
    children = []
    i = 0
    experiment_name = tasks[0]["experiment_name"]
    while ready_count < len(tasks):
        task = tasks[i]
        p = multiprocessing.Process(target=launch_single_task,
            args=(input_data, result_data, task))
        p.start()
        if not wait_for_ready_tasks(ready_count + 1, 120.0):
            print(("Timed out waiting for task %d/%d of experiment %s to " +
                "start. Retrying.") % (task["task_index"] + 1,
                    task["num_competitors"], experiment_name))
            p.terminate()
            time.sleep(5.0)
            p.close()
            p = None
            continue
        ready_count += 1
        children.append(p)
        i += 1
    print("All child tasks for %s ready. Releasing." % (experiment_name,))
    kfmlp_control.release_ts()
    for p in children:
        p.join()

def run_6way_sharing_experiment(input_data, result_data):
    """ Runs the 6-way sharing experiment. """
    # I wrote this while too tired; it probably should be refactored.
    scenarios = compare_6way_sharing()
    run_task_system(scenarios[0], input_data, result_data)
    kfmlp_control.set_k(1)
    run_task_system(scenarios[1], input_data, result_data)
    kfmlp_control.set_k(2)
    run_task_system(scenarios[2], input_data, result_data)
    run_task_system(scenarios[3], input_data, result_data)

def main():
    kfmlp_control.reset_module_handle()
    input_data, result_data = load_dataset()
    print("Input shape: %s, result shape: %s" % (str(input_data.shape),
        str(result_data.shape)))
    random.seed(1337)
    run_6way_sharing_experiment(input_data, result_data)

if __name__ == "__main__":
    main()


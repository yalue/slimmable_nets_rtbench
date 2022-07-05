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
    import gc
    import mmap
    f = open(filename, "r+b")
    prot = mmap.PROT_READ | mmap.PROT_WRITE
    mm = mmap.mmap(f.fileno(), 0, flags=mmap.MAP_SHARED, prot=prot)
    f.close()
    # Copy the entire file into a separate memory mapping.
    mm2 = mmap.mmap(-1, mm.size(), flags=mmap.MAP_SHARED, prot=prot)
    mm2.write(mm.read())
    mm2.seek(0)
    a = numpy.frombuffer(mm2, dtype=dtype)
    gc.collect()
    return a.reshape(shape)

def load_dataset():
    """ Loads the existing dataset blobs from disk. The files must already be
    present for this to succeed. Returns the input and result numpy ndarrays,
    respectively. (The child processes should convert them to torch Tensors.)
    """
    print("Loading input dataset.")
    input_numpy = get_mmapped_ndarray("input_data_raw_small.bin", (-1, 3, 224, 224),
        "float32")
    print("Loading result dataset.")
    result_numpy = get_mmapped_ndarray("result_data_raw_small.bin", (-1,), "int64")
    return (input_numpy, result_numpy)

class FakeArgs:
    """ The rtbenchmark.py functionality expects command-line args for plenty
    of things, so this class mimics the attributes it may expect from the
    command-line args. """
    def __init__(self, config):
        self.batch_size = 32
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
        self.cu_mask = ""
        self.experiment_name = ""
        self.scenario_name = ""
        self.do_kutrace = False
        self.no_preload_dataset = False
        self.preload_gpu_memory = False
        self.insert_trace_marks = False
        self.use_data_blobs = True
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

def compare_sharing_methods(num_competitors, batch_size=32, width_mult=1.0):
    """ Returns a set of scenarios, in which systems of four tasks contend for
    the GPU, using different possible management scenarios. """
    width_int = int(width_mult * 100.0)
    experiment_name = "%d-Way Sharing Management Comparison (Batch = %d, Width = %d)" % (
        num_competitors, batch_size, width_int)
    base_task = {
        "num_competitors": num_competitors,
        "task_system_index": 0,
        "relative_deadline": -1.0,
        "time_limit": 60.0,
        "experiment_name": experiment_name,
        "job_count": 0,
        "batch_size": batch_size,
        "width_mult": width_mult,
    }
    base_task_system = []
    for i in range(num_competitors):
        t = copy.deepcopy(base_task)
        t["task_index"] = i
        base_task_system.append(t)

    def make_scenario(k, use_partitions, task_system_index):
        """ Returns a task system for a given scenario, based on
        base_task_system."""
        scenario_name = ""
        file_name = "%d_competitors_" % (num_competitors,)
        if k == 0:
            scenario_name = "Unmanaged"
            file_name += "unmanaged"
        elif k == 1:
            scenario_name = "Exclusive access"
            file_name += "exclusive"
        else:
            tmp = "unpartitioned"
            if use_partitions:
                tmp = "partitioned"
            scenario_name = "%d-way sharing (%s)" % (k, tmp)
            file_name += "%dway_%s" % (k, tmp)
        file_name += "_%d_batch_%d_width" % (batch_size, width_int)
        to_return = copy.deepcopy(base_task_system)

        # Hack for 4 way partitioning
        cu_masks = None
        if k == 4:
            cu_masks = [
                "1" * 15,
                "2" * 15,
                "4" * 15,
                "8" * 15,
            ]

        for i in range(len(to_return)):
            t = to_return[i]
            # It won't matter that k is put in the args; rtbenchmark.py should
            # just ignore it, but it's helpful when we're setting partition
            # sizes.
            t["k"] = k
            t["output_file"] = "results/%s_task%d.json" % (file_name, i)
            t["scenario_name"] = scenario_name
            t["task_system_index"] = task_system_index
            if k == 4:
                # No need to lock or partition when we manually specify a CU
                # mask.
                t["cu_mask"] = cu_masks[i]
                t["use_locking"] = False
                t["use_partitioned_streams"] = False
            else:
                if k != 0:
                    t["use_locking"] = True
                if use_partitions:
                    t["use_partitioned_streams"] = True
        return to_return

    ts_index = 0
    task_systems = []
    # Unamanaged, full GPU
    task_systems.append(make_scenario(0, False, 0))
    # Locked, full GPU
    task_systems.append(make_scenario(1, False, 1))
    # 2-exclusion locking, no partitioning
    task_systems.append(make_scenario(2, False, 2))
    # 2-exclusion locking, partitioning
    task_systems.append(make_scenario(2, True, 3))
    # All four tasks are partitioned
    task_systems.append(make_scenario(4, True, 4))
    return task_systems

def cu_masks_with_size(n):
    """ Returns a mask containing the given number of CUs along with a mask
    containing all of the other CUs. """
    # Full GPU, for both this and the competitor(s).
    if (n == 60) or (n == 0):
        return "f" * 15, "f" * 15
    # SE-packed is better for the even split
    if n == 30:
        return "3" * 15, "c" * 15
    # SE-distributed is better for the 1/3rd split
    if n == 20:
        x = "fffff"
        y = "00000"
        return x + y + y, y + x + x
    if n == 40:
        return x + x + y, y + y + x
    if n == 8:
        return "ff" + "0" * 13, "00" + "f" * 13
    # SE-packed is better for 15 CUs.
    if n == 15:
        return "1" * 15, "e" * 15
    if n == 45:
        return "e" * 15, "1" * 15
    print("Unsupported number of CUs: " + str(n))
    exit(1)
    return None


def test_cu_masking_isolation(partition_size, num_competitors,
        competitor_bs=32, competitor_wm=1.0):
    """ Takes a partition size and the number of competitors to assign to the
    other partition, and returns a task system intended to contain one measured
    task in one partition, with all the competitors in the rest. """
    wm_pct = int(competitor_wm * 100.0)
    experiment_name = "Partitioning Efficacy (%d CUs) With %d batch, %d%% width Competitors" % (
        partition_size, competitor_bs, wm_pct)
    measured_mask, competitor_mask = cu_masks_with_size(partition_size)
    scenario_name = "vs. %d competitors" % (num_competitors,)
    base_filename = "%d_cus_%d_competitors_%d_cbs_%d_wm.json" % (
        partition_size, num_competitors, competitor_bs, wm_pct)
    measured_config = {
        "task_index": 0,
        "num_competitors": num_competitors + 1,
        "time_limit": 60.0,
        "job_count": 0,
        "experiment_name": experiment_name,
        "scenario_name": scenario_name,
        "output_file": "results/measured_" + base_filename,
        "batch_size": 32,
        "width_mult": 1.0,
        "cu_mask": measured_mask,
    }
    tasks = [measured_config]
    for i in range(num_competitors):
        competitor_filename = "results/competitor_%d_%s" % (i, base_filename)
        # All of the competitors will be in a different "scenario" so they
        # ought to be excluded from the results we want.
        competitor_config = {
            "task_index": i + 1,
            "num_competitors": num_competitors + 1,
            "time_limit": 60.0,
            "job_count": 0,
            "experiment_name": experiment_name,
            "scenario_name": scenario_name + " (competitor)",
            "output_file": competitor_filename,
            "batch_size": competitor_bs,
            "width_mult": competitor_wm,
            "cu_mask": competitor_mask,
        }
        tasks.append(competitor_config)
    return tasks

def get_competitors(n, width_mult, batch_size, exp_name, file_basename,
        cu_mask = ""):
    """ Returns a list of n identical "competitor" configs for use in
    experiments. The first in the list will have a task_index of 1, though
    honestly I don't even remember if I use these for anything. """
    wm_pct = int(width_mult * 100.0)
    # Scenario name shouldn't be important, as we will be ignoring competitor
    # results like this.
    scenario_name = "%s (competitor, bs=%d, wm=%d)" % (exp_name, batch_size,
        wm_pct)
    to_return = []
    for i in range(n):
        output_filename = "results/competitor%d_%s_width%d_batch%d.json" % (
            i + 1, file_basename, wm_pct, batch_size)
        config = {
            "task_index": i + 1,
            "experiment_name": exp_name,
            "num_competitors": n + 1,
            "scenario_name": scenario_name,
            "output_file": output_filename,
            "batch_size": batch_size,
            "width_mult": width_mult,
            "job_count": 0,
            "time_limit": 60.0,
        }
        if cu_mask != "":
            config["cu_mask"] = cu_mask
        to_return.append(config)
    return to_return

def gen_partitioning_experiment(n_competitors, measured_size, competitor_size):
    """ n_competitors is self explanatory. measured_size determines the
    parameters we'll assign to the measured task. This size must be a string:
    "small", "med", or "large". (competitor_size is the same, but for the
    competitors) """

    sizes = {
        "small": (8, 0.25),
        "med": (32, 0.5),
        "large": (64, 1.0),
    }
    partition_sizes = [15, 20, 30, 60]
    experiment_name = "Performance of a %s task vs. %d %s competitors" % (
        measured_size, n_competitors, competitor_size)
    partial_filename = "%s_" % (measured_size,)
    if n_competitors == 0:
        partial_filename += "isolated"
    else:
        partial_filename += "vs_%d_%s" % (n_competitors, competitor_size)
    to_return = []
    for cu_count in partition_sizes:
        measured_mask, comp_mask = cu_masks_with_size(cu_count)
        scenario_name = ""
        management_str = ""
        if cu_count == 60:
            scenario_name = "Unpartitioned"
            management_str = "_unpartitioned"
        else:
            scenario_name = "Partitioned (%d CUs)" % (cu_count,)
            management_str = "_%d_cus" % (cu_count,)
        output_filename = "results/measured_%s%s.json" % (partial_filename,
            management_str)
        measured_config = {
            "task_index": 0,
            "num_competitors": n_competitors + 1,
            "scenario_name": scenario_name,
            "experiment_name": experiment_name,
            "output_file": output_filename,
            "cu_mask": measured_mask,
            "batch_size": sizes[measured_size][0],
            "width_mult": sizes[measured_size][1],
            "job_count": 0,
            "time_limit": 60.0,
        }
        task_system = []
        task_system.append(measured_config)
        if n_competitors != 0:
            c_size = sizes[competitor_size]
            competitors = get_competitors(n_competitors, c_size[1], c_size[0],
                experiment_name, partial_filename + management_str,
                cu_mask=comp_mask)
            task_system.extend(competitors)
        to_return.append(task_system)
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

def run_4way_sharing_experiment(input_data, result_data, batch_size,
        width_mult):
    task_systems = compare_sharing_methods(4, batch_size=batch_size,
        width_mult=width_mult)
    kfmlp_control.set_k(1)
    for ts in task_systems:
        if ts[0]["k"] > 1:
            kfmlp_control.set_k(ts[0]["k"])
        run_task_system(ts, input_data, result_data)

def generate_4way_sharing_data(input_data, result_data):
    """ Run this to generate the data that's now in the 4way_sharing_results
    directory. """
    batch_sizes = [8, 32, 64]
    width_mults = [1.0, 0.5, 0.25]
    for bs in batch_sizes:
        for wm in width_mults:
            run_4way_sharing_experiment(input_data, result_data, bs, wm)

def generate_cu_mask_efficacy_data(input_data, result_data):
    competitor_batch_sizes = [8, 64]
    competitor_width_mults = [0.25, 1.0]
    measured_partition_sizes = [8, 15, 20, 30]
    for cbs in competitor_batch_sizes:
        for cwm in competitor_width_mults:
            for cu_count in measured_partition_sizes:
                for num_competitors in range(4):
                    task_system = test_cu_masking_isolation(cu_count,
                        num_competitors, cbs, cwm)
                    run_task_system(task_system, input_data, result_data)

def gen_partitioning_data(input_data, result_data):
    # Generating data for three tables and/or CDFs:
    #  - Small tasks vs nothing or small, med, and large competitors w/
    #    different partitioning.
    #  - Same but for med. tasks
    #  - Same but for large tasks.
    #
    # Should result in 36 task systems: (3 * 3) size configs * 4 partitioning
    # types per size config.
    competitor_sizes = ["none", "small", "med", "large"]
    measured_sizes = ["small", "med", "large"]
    for ms in measured_sizes:
        for cs in competitor_sizes:
            if cs == "none":
                task_systems = gen_partitioning_experiment(0, ms, "junk")
            else:
                task_systems = gen_partitioning_experiment(3, ms, cs)
            for ts in task_systems:
                run_task_system(ts, input_data, result_data)

def run_job_vs_kernel_locking(input_data, result_data):
    """ Intended to test exclusive locking on a per-job or per-kernel
    granularity. """
    # Six things to run:
    #  - Single "medium" job, unmanaged.
    #  - Single "medium" job, per-kernel locking.
    #  - Single "medium" job, per-job locking.
    #  - Two "medium" jobs, unmanaged.
    #  - Two "medium" jobs, per-kernel locking.
    #  - Two "medium" jobs, per-job locking.
    envname = "USE_KFMLP_LOCKING"
    task1 = {
        "experiment_name": "Job vs. Kernel Locking",
        "scenario_name": "1 Task, Unmanaged",
        "output_file": "results/isolated_unmanaged_med.json",
        "num_competitors": 1,
        "task_index": 0,
        "time_limit": 60.0,
        "job_count": 0,
        "batch_size": 32,
        "width_mult": 0.5,
    }
    # 1 task, unmanaged
    run_task_system([task1], input_data, result_data)

    # 1 task, per-kernel locking
    task1["scenario_name"] = "1 Task, Per-Kernel Locking"
    task1["output_file"] = "results/isolated_perkernel_med.json"
    os.environ[envname] = "1"
    run_task_system([task1], input_data, result_data)

    # 1 task, per-job locking
    del os.environ[envname]
    task1["scenario_name"] = "1 Task, Per-Job Locking"
    task1["output_file"] = "results/isolated_perjob_med.json"
    task1["use_locking"] = True
    run_task_system([task1], input_data, result_data)

    # 2 tasks, unmanaged
    task1["use_locking"] = False
    task1["num_competitors"] = 2
    task1["scenario_name"] = "2 Tasks, Unmanaged"
    task2 = copy.deepcopy(task1)
    task2["task_index"] = 1
    task1["output_file"] = "results/2tasks_unmanaged_med_task0.json"
    task2["output_file"] = "results/2tasks_unmanaged_med_task1.json"
    run_task_system([task1, task2], input_data, result_data)

    # 2 tasks, per-kernel locking
    task1["scenario_name"] = "2 Tasks, Per-Kernel Locking"
    task2["scenario_name"] = task1["scenario_name"]
    task1["output_file"] = "results/2tasks_perkernel_med_task0.json"
    task2["output_file"] = "results/2tasks_perkernel_med_task1.json"
    os.environ[envname] = "1"
    run_task_system([task1, task2], input_data, result_data)

    # 2 tasks, per-job locking
    del os.environ["USE_KFMLP_LOCKING"]
    task1["scenario_name"] = "2 Tasks, Per-Job Locking"
    task2["scenario_name"] = task1["scenario_name"]
    task1["output_file"] = "results/2tasks_perjob_med_task0.json"
    task2["output_file"] = "results/2tasks_perjob_med_task1.json"
    task1["use_locking"] = True
    task2["use_locking"] = True
    run_task_system([task1, task2], input_data, result_data)

def main():
    kfmlp_control.reset_module_handle()
    input_data, result_data = load_dataset()
    generate_4way_sharing_data(input_data, result_data)

if __name__ == "__main__":
    main()


import mmap
import multiprocessing
import numpy
import time

import liblitmus_helper as liblitmus

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
    input_numpy = get_mmapped_ndarray("input_data_raw.bin", (-1, 3, 224, 224),
        "float32")
    page_in_ndarray(input_numpy)
    print("Loading result dataset.")
    result_numpy = get_mmapped_ndarray("result_data_raw.bin", (-1,), "int64")
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
        self.use_litmus = True
        self.k_exclusion_value = -1
        self.use_partitioned_streams = False
        # Default relative deadline = 2 Hz. Arbitrary and ought to be
        # overridden in typical uses.
        self.relative_deadline = 0.5
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
    """ Waits until the number of LITMUS task-sysytem release waiters is at
    least the given count. Returns True when this happens, or False if the
    timeout occurs first. """
    waiting = liblitmus.get_nr_ts_release_waiters()
    start_time = time.perf_counter()
    while waiting < count:
        cur_time = time.perf_counter()
        if (cur_time - start_time) > timeout:
            return False
        time.sleep(0.1)
        waiting = liblitmus.get_nr_ts_release_waiters()
    return True

def run_all_kernel_configs(input_dataset, result_dataset):
    """ Runs one task with each possible batch size and width multiplier.
    Necessary for generating cached kernel code used by AMD's software. Make
    sure to run a program with this function before actually doing any tests.
    """
    width_mult_list = [0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45,
        0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725,
        0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0]
    batch_size_list = [1, 2, 4, 8, 16, 32, 64, 128]
    for bs in batch_size_list:
        for wm in width_mult_list:
            config = {
                "batch_size": bs,
                "width_mult": wm,
                "job_count": 2,
                "relative_deadline": 2.0,
            }
            print("Running test with width mult %.03f, batch size %d" % (wm,
                bs))
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
            liblitmus.release_ts(liblitmus.litmus_clock())
            p.join()
            end_time = time.perf_counter()
            print("Test with width_mult %.03f, batch size %d took %.03fs" % (
                wm, bs, end_time - start_time))
    return True

def main():
    input_data, result_data = load_dataset()
    print("Input shape: %s, result shape: %s" % (str(input_data.shape),
        str(result_data.shape)))
    # TODO: Actually generate configs for multiple RT tasks.
    tasks = []
    ready_count = 0
    config = {}
    # Kick off two tasks as a test
    while ready_count < 2:
        p = multiprocessing.Process(target=launch_single_task,
            args=(input_data, result_data, config))
        p.start()
        # ROCm is pretty flaky about tasks hanging during warmup and
        # initialization. So, kill any tasks that seem to be taking too long,
        # and try starting them up again.
        if not wait_for_ready_tasks(ready_count + 1, 120.0):
            print("Timed out waiting for RT task to start. Retrying.")
            p.terminate()
            time.sleep(5.0)
            p.close()
            p = None
            continue
        ready_count += 1
        tasks.append(p)
    print("All child tasks launched. Releasing and waiting.")
    liblitmus.release_ts(liblitmus.litmus_clock())
    for p in tasks:
        p.join()

if __name__ == "__main__":
    main()


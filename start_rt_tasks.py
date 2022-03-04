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

def load_dataset():
    """ Loads the existing dataset blobs from disk. The files must already be
    present for this to succeed. Returns the input and result numpy ndarrays,
    respectively. (The child processes should convert them to torch Tensors.)
    """
    input_numpy = get_mmapped_ndarray("input_data_raw.bin", (-1, 3, 224, 224),
        "float32")
    result_numpy = get_mmapped_ndarray("result_data_raw.bin", (-1,), "int64")
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
        self.wait_for_ts_release = True
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
        if not wait_for_ready_tasks(ready_count + 1, 60.0):
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


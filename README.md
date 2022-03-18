Real-time Benchmark Based on "Slimmable Networks"
=================================================

This repository is a modification of
[the original](https://github.com/JiahuiYu/slimmable_networks) by Yu et. al.

I have made, or plan to make, the following changes to make the code a
benchmark for real-time applications:

 - Removed the ability to train; only running inference on a pretrained
   network.

 - Added more command-line flags, making it easier for other scripts to
   configure parameters.

 - Changed the way the data is loaded to avoid reading from disk.

 - Added code for measuring times and rate-limiting jobs.

 - Various other things to make it more suitable to my personal preferencecs.


Usage
-----

This project, being primarily used by my own research, requires _many_ specific
modified dependencies. You will likely need to install all of the following
projects on your system in order for this to function correctly, if at all.
They should be installed in the following order:

 1. [My modifications to ROCm.](https://github.com/yalue/rocm_mega_repo). Note
    that this requires a specific ROCm version to be installed. It adds
    userspace support for per-kernel GPU locking and subdividing large memory-
    transfer requests.

 2. The `rocm_helper` python library. This is part of the same
    `rocm_mega_repo`, but I'm listing it separately so it won't be overlooked.
    Follow the instructions in the `rocm_helper_python` directory in the above
    repository. It's used to create streams with CU masks within PyTorch
    scripts.

 3. [My modified version of PyTorch.](https://github.com/yalue/rocm_pytorch).
    Install this from source, using the same instructions you would when
    installing vanilla PyTorch from source. (It has been run through `hipify`
    already, so you don't need to do so again.) This fixes support for using
    external CUDA/HIP streams that ought to be present in vanilla PyTorch, but
    was apparently never fully integrated. Maybe I'll submit a patch to
    upstream about this some day.

 4. [My GPU-locking kernel module.](https://github.com/yalue/gpu_locking_module).
    While not used for locking any more (I use my newer KFMLP module instead),
    this still provides a user-accessible interface for evicting running tasks
    off of an AMD GPU.

 5. [My KFMLP locking support.](https://github.com/yalue/kfmlp_locking_module).
    This consists of a Linux kernel module and a python library for interacting
    with it. The kernel module provides a k-exclusion lock and a few other
    convenience features, such as a barrier allowing a task system to be
    released only when all tasks are ready, and an API for switching a process
    to use the SCHED_FIFO scheduler, bypassing kernel permission checks.

 6. The pre-computed data blobs. This will, in turn, require the imagenet
    dataset on-disk in the same layout expected by the
    [original repo](https://github.com/JiahuiYu/slimmable_networks). Once you
    have the imagenet dataset in the correct layout, change the `dataset_dir`
    field in `config.py` to point to your dataset location. (On my system,
    the `dataset_dir` contains three directories: `test`, `train`, and `val`.
    The `test` directory is full of JPG images, while the other two dirs are
    full of directories containing JPG images. I'm not sure specifically what
    is required by the original `slimmable_networks` code responsible for
    parsing this structure.) Anyway, once you have this directory set up,
    run `python generate_data_blobs.py` to generate `input_data_raw.bin` and
    `result_data_raw.bin`. Note that `input_data_raw.bin` is about 6 GB. This
    entire file is buffered into memory at runtime to simplify loading logic.

After setting up *all* of the above dependencies, you should be able to run
`rtbenchmark.py` using Python 3. Run `python rtbenchmark.py --help` for usage
information.


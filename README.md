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

 1. [My modified LITMUS^RT kernel.](https://github.com/yalue/litmus-rt/tree/add_kfmlp).
    (Note the branch.) This adds kernel support for k-FMLP locking, and is
    simply a version of LITMUS that is new enough to support ROCm. This
    requires some specic config options discussed in my [porting notes](https://gist.github.com/yalue/6852e9b88abbc60beba9c855a0045271).
    A working `.config` file for my Ubuntu 18.04-based system can be found
    [here](https://gist.github.com/yalue/f22e28165f518b37497155db662af027).

 2. [My modifications to ROCm.](https://github.com/yalue/rocm_mega_repo). Note
    that this requires a specific ROCm version to be installed. It adds
    userspace support for per-kernel GPU locking and subdividing large memory-
    transfer requests.

 3. The `rocm_helper` python library. This is part of the same
    `rocm_mega_repo`, but I'm listing it separately so it won't be overlooked.
    Follow the instructions in the `rocm_helper_python` directory in the above
    repository. It's used to create streams with CU masks within PyTorch
    scripts.

 4. [My modified version of PyTorch.](https://github.com/yalue/rocm_pytorch).
    Install this from source, using the same instructions you would when
    installing vanilla PyTorch from source. (It has been run through `hipify`
    already, so you don't need to do so again.) This fixes support for using
    external CUDA/HIP streams that ought to be present in vanilla PyTorch, but
    was apparently never fully integrated. Maybe I'll submit a patch to
    upstream about this some day.

 5. [My GPU-locking kernel module.](https://github.com/yalue/gpu_locking_module).
    While not used for locking any more (due to LITMUS support), this still
    provides a user-accessible interface for evicting running tasks off of an
    AMD GPU.

 6. [My python library for interacting with LITMUS.](https://github.com/yalue/liblitmus_python)
    This provides the ability to configure Python scripts as real-time tasks,
    run jobs, acquire locks, etc.

After setting up *all* of the above dependencies, you should be able to run
`rtbenchmark.py` using Python 3. Run `python rtbenchmark.py --help` for usage
information.


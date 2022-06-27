# This file contains some code for generating PyTorch streams for various GPU
# partition configurations.

try:
    import rocm_helper
except:
    print("rocm_helper not available. Expect errors if using partitioning.")

import torch

def split_halves(mask):
    """ Takes a 64-bit integer and returns a tuple (low_cus, high_cus), that
    can be passed to create_stream_with_cu_mask. """
    upper = mask >> 32
    lower = mask & 0xffffffff
    return (lower, upper)

def hex_list_to_mask(a):
    """ Takes an array of hex strings and returns an array of tuples
    corresponding to the upper and lower halves of each. """
    to_return = []
    for s in a:
        to_return.append(split_halves(int(s, 16)))
    return to_return

def cu_masks_for_partitions(k):
    """ Takes a number of partitions, k, and returns a list of CU masks for
    k streams: one mask for each of the k partitions. This is done manually
    to ensure partitions are distributed evenly across shader engines. This
    assumes a Radeon VII GPU with 4 SEs and 60 CUs. Note that not all values of
    k are supported, since not all can be evenly distributed across 4 SEs. """
    # Remember that CU masks are "striped" across the four SEs. So, a mask of
    # 0xf is 1 CU on each SE, while a mask of 0x1111 is four CUs on 1 SE.
    if k == 1:
        # All 60 CUs set.
        return hex_list_to_mask(["f" * 15])
    if k == 2:
        # 1st mask = CUs 0 and 1, 2nd mask = CUs 2 and 3.
        return hex_list_to_mask(["3" * 15, "c" * 15])
    if k == 3:
        # Each of 3 masks = 20 CUs, 5 per SE.
        x = "fffff"
        y = "00000"
        return hex_list_to_mask([x, x + y, x + y + y])
    if k == 4:
        # Each mask = all CUs on one SE.
        return hex_list_to_mask(["1" * 15, "2" * 15, "4" * 15, "8" * 15])
    if k == 5:
        # Each mask gets 3 CUs per SE.
        x = "fff"
        y = "000"
        return hex_list_to_mask([x, x + y, x + 2 * y, x + 3 * y, x + 4 * y])
    if k == 15:
        # Each mask gets one CU on each SE.
        hex_list = []
        for i in range(15):
            hex_list.append("f" + i * "0")
        return hex_list_to_mask(hex_list)
    print("Unsupported k value for partitioning (%d)." % (k,))
    exit(1)

def streams_for_partitions(k):
    """ Takes a number of partitions, k, and returns a list of k PyTorch
    streams with non-overlapping CU masks. """
    masks = cu_masks_for_partitions(k)
    to_return = []
    for i in range(k):
        s = rocm_helper.create_stream_with_cu_mask(masks[i][0], masks[i][1])
        stream = torch.cuda.streams.ExternalStream(s)
        to_return.append(stream)
    return to_return


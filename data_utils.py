import torch
import numpy as np
import time

class PreloadDataset(torch.utils.data.Dataset):
    """ A dataset class that loads another dataset and buffers it all in
    memory. Expects the result format to be a plain python int. """

    def __init__(self, data, data_limit, device):
        """ Requires an existing dataset and a device on which to create the
        buffer. data_limit is a maximum number of items to load. """
        self.device = device
        self.data_count = len(data)
        # 20000 results in ~11 GB usage for ImageNet data.
        #data_limit = 10000
        data_limit = 1000
        if self.data_count > data_limit:
            self.data_count = data_limit

        # Shuffle the input data now.
        print("Pre-generating shuffled data order.")
        shuffled_order = np.arange(len(data))
        np.random.shuffle(shuffled_order)

        a, b = data[0]
        print("Input (%d values): shape %s, dtype %s" % (self.data_count,
            str(a.size()), str(a.dtype)))
        if not isinstance(b, int):
            print("Expected a dataset with a plain int result.")
            exit(1)

        data_tensor_shape = torch.Size([self.data_count] + list(a.size()))
        self.result_tensor = torch.zeros(self.data_count, dtype=int)
        self.input_tensor = torch.zeros(data_tensor_shape, dtype=a.dtype)

        tmp = time.perf_counter()
        for i in range(self.data_count):
            if (i % 1000) == 999:
                print("Preloading data: %d/%d" % (i, self.data_count))
            # This is where we do the shuffling, otherwise imagenet data is
            # returned class-by-class.
            input_data, result = data[shuffled_order[i]]
            self.input_tensor[i] = input_data
            self.result_tensor[i] = result
        self.result_tensor = self.result_tensor.to(device)
        self.input_tensor = self.input_tensor.to(device)
        print("Preloading data: done. Took %f seconds." % (time.perf_counter() - tmp,))

        #print(str(torch.cuda.memory_summary()))
        return

    def __getitem__(self, index):
        return self.input_tensor[index], self.result_tensor[index]

    def __len__(self):
        return self.data_count

    def get_device(self):
        return self.device

# TODO (next): Add the SimpleLoader code here too.

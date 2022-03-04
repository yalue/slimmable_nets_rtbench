import torch
import numpy as np
import gc
import time

def dump_to_disk(dataset, sample_count, input_data_filename,
    result_data_filename):
    """ Selects a random set of samples from across the entire given dataset.
    Saves the data into flat raw binary blobs with the given filenames. """
    print("Determining which %d/%d random samples to save to disk." % (
        sample_count, len(dataset)))
    # Sample across the entire dataset.
    shuffled_samples = np.arange(len(dataset))
    np.random.shuffle(shuffled_samples)

    a, b = dataset[0]
    data_tensor_shape = torch.Size([sample_count] + list(a.size()))
    result_tensor = torch.zeros(sample_count, dtype=int)
    input_tensor = torch.zeros(data_tensor_shape, dtype=a.dtype)
    print("Reading %d random samples..." % (sample_count,))
    for i in range(sample_count):
        input_data, result = dataset[shuffled_samples[i]]
        input_tensor[i] = input_data
        result_tensor[i] = result
    gc.collect()
    print("Saving data to disk...")
    tmp = input_tensor.numpy().tobytes()
    print("Input data size: %d bytes uncompressed." % len(tmp))
    with open(input_data_filename, "wb") as f:
        f.write(tmp)
    tmp = result_tensor.numpy().tobytes()
    print("Result data size: %d bytes uncompressed." % len(tmp))
    with open(result_data_filename, "wb") as f:
        f.write(tmp)
    print("Saved data to %s and %s." % (input_data_filename,
        result_data_filename))

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
        if self.data_count > data_limit:
            print("Using %d/%d available input data points." % (data_limit,
                len(data)))
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
        return

    def __getitem__(self, index):
        return self.input_tensor[index], self.result_tensor[index]

    def __len__(self):
        return self.data_count

    def get_device(self):
        return self.device

class SimpleLoader(object):
    """ This class replaces PyTorch's loader, and simply wraps our
    PreloadDataset to return subsequent slices. Always drops an unevenly sized
    last batch, and doesn't shuffle the data. """

    def __init__(self, dataset, batch_size):
        """ Expects a PreloadDataset. """
        self.dataset = dataset
        self.current_index = 0
        self.batch_size = batch_size

    def __next__(self):
        if (self.current_index + self.batch_size) >= len(self.dataset):
            raise StopIteration
        i = self.current_index
        data_slice = self.dataset.input_tensor[i : (i + self.batch_size)]
        result_slice = self.dataset.result_tensor[i : (i + self.batch_size)]
        self.current_index += self.batch_size
        return (data_slice, result_slice)

    def __iter__(self):
        self.current_index = 0
        return self

    def __len__(self):
        return int(int(len(self.dataset)) / int(self.batch_size))


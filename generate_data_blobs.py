from torchvision import datasets, transforms
from config import FLAGS
from data_utils import dump_to_disk
import os

def data_transforms():
    """ Taken from the original repo. """
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

def main():
    val_transforms = data_transforms()
    val_set = datasets.ImageFolder(os.path.join(FLAGS.dataset_dir, 'val'),
        transform=val_transforms)
    dump_to_disk(val_set, 6000, "input_data_raw.bin", "result_data_raw.bin")

if __name__ == "__main__":
    main()


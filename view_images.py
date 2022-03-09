# A quick script for viewing images in the dataset. Requires
# generate_data_blobs.py to have run. Used as a sanity check so I can see
# visually that my dataset blob has reasonable stuff in it.
import mmap
import numpy
import cv2
import json

def get_label_info():
    """ Basically loads imagenet_class_index.json from disk. """
    data = None
    with open("imagenet_class_index.json", "r") as f:
        data = json.load(f)
    return data

def get_mmapped_ndarray(filename, shape, dtype):
    """ Returns a numpy ndarray with the content of the named file and the
    given shape. """
    f = open(filename, "r+b")
    prot = mmap.PROT_READ | mmap.PROT_WRITE
    mm = mmap.mmap(f.fileno(), 0, flags=mmap.MAP_SHARED, prot=prot)
    f.close()
    a = numpy.frombuffer(mm, dtype=dtype)
    return a.reshape(shape)

def convert_format(pic):
    """ Returns a picture with the size required by imshow. The input pic must
    be 3x224x224. Returns one that is 224x224x3. """
    to_return = numpy.zeros((224, 224, 3))
    for chan in range(3):
        for y in range(224):
            for x in range(224):
                to_return[x, y, chan] = pic[chan, x, y]
    return to_return


def main():
    input_data = get_mmapped_ndarray("input_data_raw.bin", (-1, 3, 224, 224),
        "float32")
    label_data = get_mmapped_ndarray("result_data_raw.bin", (-1,), "int64")
    label_info = get_label_info()
    for i in range(len(input_data)):
        pic = convert_format(input_data[i])
        label = label_data[i]
        label_name = label_info[str(label)][1]
        print("Showing image %d/%d. Label = %d" % (i + 1, len(input_data),
            label))
        cv2.imshow("Image class %s (%d)" % (label_name, label,), pic)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


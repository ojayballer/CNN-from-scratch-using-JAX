import numpy as np
import struct
import os

def load(path, kind='train'):
    here = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(here, path)

    if kind == 'train':
        img_path = os.path.join(folder, 'train-images-idx3-ubyte')
        lbl_path = os.path.join(folder, 'train-labels-idx1-ubyte')
    else:
        img_path = os.path.join(folder, 't10k-images-idx3-ubyte')
        lbl_path = os.path.join(folder, 't10k-labels-idx1-ubyte')

    with open(lbl_path, 'rb') as f:
        f.read(8)
        labels = np.fromfile(f, dtype=np.uint8)

    with open(img_path, 'rb') as f:
        header_data = f.read(16)
        magic, num_images, rows, cols = struct.unpack('>IIII', header_data)
        raw_data = np.fromfile(f, dtype=np.uint8)
        images = raw_data.reshape(num_images, 1, rows, cols)
        images = images.astype('float32') / 255.0

    return images, labels

def one_hot_encode(y, num_classes=10):
    encoded = np.zeros((len(y), num_classes))
    for i in range(len(y)):
        encoded[i, y[i]] = 1
    return encoded
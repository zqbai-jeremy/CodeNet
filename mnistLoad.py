import numpy as np
import struct
import matplotlib.pyplot as plt

def readImages(filename, read_size):
    binfile = open(filename , 'rb')
    buf = binfile.read()
     
    index = 0
    magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , buf , index)
    index += struct.calcsize('>IIII')

    images = []
    for i in range(read_size):
        im = struct.unpack_from('>784B' ,buf, index)
        images.append(im)
        index += struct.calcsize('>784B')
    images = np.array(images, dtype='d')
    return images


def readLabels(filename, read_size):
    binfile = open(filename , 'rb')
    buf = binfile.read()
     
    index = 0
    magic, numImages = struct.unpack_from('>II' , buf , index)
    index += struct.calcsize('>II')

    labels = []
    for i in range(read_size):
        label = struct.unpack_from('>B' ,buf, index)
        labels += label
        index += struct.calcsize('>B')
    labels = np.array(labels)
    return labels

if __name__ == '__main__':
    #images = readImages('train-images.idx3-ubyte')
    #labels = readLabels('train-labels.idx1-ubyte')
    images = readImages('t10k-images.idx3-ubyte', 1000)
    labels = readLabels('t10k-labels.idx1-ubyte', 1000)

    im = images[200].reshape(28,28)
    print(labels[200], type(labels[200]))
    print(im)
     
    fig = plt.figure()
    plotwindow = fig.add_subplot(111)
    plt.imshow(im , cmap='gray')
    plt.show()
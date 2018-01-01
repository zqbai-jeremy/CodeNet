import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def readImagesLabels(file, read_size):
    dict = unpickle(file)
    X = np.array(dict['data'], dtype='d')[:read_size]
    Y = np.array(dict['labels'], dtype='i')[:read_size]
    return X, Y

    
if __name__ == '__main__':
    images, labels = readImagesLabels("test_batch", 2000)
    for i in range(100):
        tmp_im = images[i].reshape(3, 32, 32)
        im = np.empty((32, 32, 3), dtype='d')
        im[:, :, 0] = 255.0 - tmp_im[0, :, :]
        im[:, :, 1] = 255.0 - tmp_im[1, :, :]
        im[:, :, 2] = 255.0 - tmp_im[2, :, :]
        print(labels[i], type(labels[i]))
        #print(im)
     
    fig = plt.figure()
    plotwindow = fig.add_subplot(111)
    plt.imshow(im)
    plt.show()
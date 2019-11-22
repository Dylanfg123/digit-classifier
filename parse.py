import struct as st
import numpy as np
import matplotlib.pyplot as plt

def main():
    filename = {'training_images': 'train-images-idx3-ubyte', 'training_labels': 'train-labels-idx1-ubyte', 'testing_images': 't10k-images-idx3-ubyte', 'testing_labels': 't10k-labels-idx1-ubyte'}

    images_array = image_parse(filename['testing_images'])
    labels_array = label_parse(filename['testing_labels'])

    ax = []
    fig = plt.figure(figsize=(10,10))
    for num in range(25):
        img = images_array[num]
        ax.append(fig.add_subplot(5, 5, num + 1))
        ax[-1].set_title(str(labels_array[num]))
        plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
    fig.tight_layout()
    plt.show()
    
def image_parse(filename):
    train_imagesfile = open(filename, 'rb')
    train_imagesfile.seek(0)
    magic = st.unpack('>4B', train_imagesfile.read(4))
    nimgs = st.unpack('>I', train_imagesfile.read(4))[0]
    nrows = st.unpack('>I', train_imagesfile.read(4))[0]
    ncolumns = st.unpack('>I', train_imagesfile.read(4))[0]
    nbytestotal = nimgs * ncolumns * nrows
    images_array = np.asarray(st.unpack('>' + 'B' * nbytestotal, train_imagesfile.read(nbytestotal))).reshape((nimgs, nrows, ncolumns))
    return images_array

def label_parse(filename):
    train_labelfile = open(filename, 'rb')
    train_labelfile.seek(0)
    label_magic = st.unpack('>I', train_labelfile.read(4))
    nlabels = st.unpack('>I', train_labelfile.read(4))[0]
    nbytestotal = nlabels
    labels_array = np.asarray(st.unpack('>' + 'B' * nbytestotal, train_labelfile.read(nbytestotal))).reshape((nlabels))
    return labels_array


if __name__ == "__main__":
    main()
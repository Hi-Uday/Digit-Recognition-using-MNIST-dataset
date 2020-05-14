from mlxtend.data import loadlocal_mnist
#doenload dataset from "http://yann.lecun.com/exdb/mnist/"

X, y = loadlocal_mnist(
        images_path='path of train-images-idx3-ubyte',
        labels_path='path for train-labels-idx1-ubyte')
np.savetxt(fname='/Users/Sebastian/Desktop/Train_images.csv',
           X=X, delimiter=',', fmt='%d')
np.savetxt(fname='/Users/Sebastian/Desktop/Train_labels.csv',
           X=y, delimiter=',', fmt='%d')

#Repeat above steps to generate other two csv file for
#Test_images.csv and Test_images.csv"
# you simply have to change the path in above file and give other two file name by comment above code
w, z = loadlocal_mnist(
        images_path='path for t10k-images-idx3-ubyte',
        labels_path='path for t10k-labels-idx1-ubyte')
np.savetxt(fname='/Users/Sebastian/Desktop/Train_images.csv',
           X=w, delimiter=',', fmt='%d')
np.savetxt(fname='/Users/Sebastian/Desktop/Train_labels.csv',
           X=z, delimiter=',', fmt='%d')



#Plese see coversion.py before this
import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

#reading testing and training database
xtrain=pd.read_csv(r"C:\Users\Administrator\PycharmProjects\image\Temp\database_digit\Train_images.csv").to_numpy()
train_label=pd.read_csv(r"C:\Users\Administrator\PycharmProjects\image\Temp\database_digit\Train_labels.csv").to_numpy()
xtest=pd.read_csv(r"C:\Users\Administrator\PycharmProjects\image\Temp\database_digit\Test_images.csv").to_numpy()
actual_label=pd.read_csv(r"C:\Users\Administrator\PycharmProjects\image\Temp\database_digit\Test_images.csv").to_numpy()

#calling DecisionTreeClassifier
clf=DecisionTreeClassifier()

#fitting of training data into classifier
clf.fit(xtrain,train_label)

#this code is ok but now we try to find out accuracy by givind all the tesing image to prodiction at a time for
#that we hide this code for time
d=xtest[16]
d.shape=(28,28)
pt.imshow(255-d,cmap=None)
print(clf.predict([xtest[16]]))
pt.show()

#this code is not working and issue is not getting
#actually this code is for calculating accuracy
# p=clf.predict(xtest)
# count=0
# for i in range(np.shape(xtest)):
#     count+=1 if p[i] == actual_label[i] else 0
# print("Accuracy=", (count/np.shape(xtest))*100)

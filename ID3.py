import pandas as pd 
import numpy as np
from sklearn.tree import export_text
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import graphviz

#import the dataset
def importdata(url,header,drop=None):
    balance_data = pd.read_csv(url,sep=",",header=header)
    if drop == None:
        pass
    else:
        dataset = dataset.drop(drop)
    print ("Dataset Lenght: ", len(balance_data)) 
    print ("Dataset Shape: ", balance_data.shape) 
    print ("Dataset: ",balance_data.head()) 
    print("Dataset values : ",balance_data.sample)
    return balance_data
#splitting the dataset, taking their slicing opt as parameters
def splitdatset(dataset,testsize,randomstate):
    x = dataset.values[:,1:5]
    y = dataset.values[:, 0]
    x_train,x_test,y_train,y_test = train_test_split(x,y,
    test_size=testsize,random_state=randomstate)
    return x,y,x_test,x_train,y_train,y_test
#planting the ID3 tree
def enttree(x_train,y_train,randomstate,minleaf,maxdepth):
    clf = DecisionTreeClassifier(criterion="entropy",random_state=randomstate,max_depth=maxdepth,min_samples_leaf=minleaf)
    clf.fit(x_train,y_train)
    return clf
#predict for new classes
def predict(clf_obj,x_test):
    prediction = clf_obj.predict(x_test)
    print("predicted values: ")
    print(prediction)
    return prediction
#checking the tree's accuracy
def checkAccuracy(clf_obj,x_test,y_test):
    accur = clf_obj.score(x_test,y_test)*100
    print("This is the Accuracy:",
    accur)
def fetchgraph(clf):
    r = export_text(clf)
    print(r)

#driver code
def main():  
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data'
    dataset = importdata(url = url,header=None,drop=None)
    x,y,x_test,x_train,y_train,y_test = splitdatset(dataset=dataset,testsize=0.3,randomstate=100)
    clf = enttree(x_train,y_train,randomstate=100,minleaf=5,maxdepth=3)
    print("results of entropy of this dataset!")
    y_pred = predict(clf,x_test)
    checkAccuracy(clf,x_test,y_test)
    fetchgraph(clf)

if __name__ == "__main__":
    main()

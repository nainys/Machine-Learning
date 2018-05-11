import numpy as np
import random
import  csv


orig = []

avg_acc = []
epochs = [10,15,20,25,30,35,40,45,50]

with open("ionosphere.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        if '?' in row:
            continue

        if row[-1] == 'b':
            row[-1] = 1
        else:
            row[-1] = -1
        row = map(float,row)
        orig.append(row)

# print len(orig)
no_of_folds = 10
kf = len(orig)//no_of_folds

############################## split data to training  and testing ##################################

def split():
    newdata = []
    copy = list(orig)
    for i in range(no_of_folds):
        curr = []
        while(len(curr) < kf):
            i = random.randrange(0,len(copy))
            curr.append(copy.pop(i))
        newdata.append(curr)

    return newdata


############################# train perceptron using training data ###################################
def train_percep(x,y,e):
    train_output = []
    w = np.zeros(len(x[0]))
    b = 0.0
    c = 0
    epochs = 0
    misclassify = True
    while misclassify and epochs < e:
        misclassify = False
        for i in xrange(len(x)):
            if (np.dot(x[i],w) + b)*y[i] <=0:
                tup = (w,b,c)
                train_output.append(tup)
                w = w + y[i]*x[i]
                b = b+y[i]
                c = 1
                misclassify = True
            else:
                c  = c+1
        epochs = epochs+1
    tup = (w,b,c)
    train_output.append(tup)

    return train_output



############################# predict on test data ########################################################
def predict(test,train_output):
    correct = 0

    for i in range(len(test)):
        temp = 0
        for j in range(len(train_output)):

            if (np.dot(test[i][0:-1],train_output[j][0]) + train_output[j][1]) >0:
                temp = temp + train_output[j][2]
            else:
                temp = temp - train_output[j][2]

        if temp > 0:
            lab = 1
        else:
            lab = -1

        if lab == test[i][-1]:
            correct = correct + 1

    return correct



###########################################################################################################
for e in epochs:
    scores = []
    for i in range(no_of_folds):
        newdata = split()
        test = newdata[i]
        train = []
        for j in range(no_of_folds):
            if j!=i:
                for z in newdata[j]:
                    train.append(z)

        x = []
        y = []
        for i in range(len(train)):
            x.append(train[i][0:-1])
            y.append(train[i][-1])

        x = np.array(x,dtype=float)
        y = np.array(y,dtype=float)
        test = np.array(test,dtype = float)


        train_output = train_percep(x,y,e)

        correct = predict(test,train_output)
        score = float(correct)/len(test)
        score = round(score,2)
        scores.append(score)
    avg_acc.append(round(float(sum(scores))/len(scores),2)*100)
    print scores

print avg_acc

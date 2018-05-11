import numpy as np
import  random
import csv

epoch = input("Enter no of epochs\n")
neurons = input("Enter no of nodes in hidden layer\n")
activation = input("Enter activation function for hidden layer\n")
activation = activation.lower()
eta = input("Enter learning rate\n") #learning rate


def read_input():
    inp = []
    with open("dermatology.data") as f:
        reader = csv.reader(f)
        for row in reader:
            if '?' in row:
                continue
            label = row[-1].strip()
            if label == '1' or label == '2' or label == '3':
                row = map(int,row)
                inp.append(row)

    return inp

inp = read_input()
no_of_folds = 5
kf = len(inp)//no_of_folds

# Split data into training and testing set
def split():
    newdata = []
    copy = list(inp)
    for i in range(5):
        curr = []
        while(len(curr) < kf):
            i = random.randrange(0,len(copy))
            curr.append(copy.pop(i))
        newdata.append(curr)

    return newdata


def init_weights(n_input,n_hidden,n_output):
    """initialize input weights uniformly randomly with small values"""
    w1 = np.random.uniform(-1.0,1.0,
    size=(n_input,n_hidden))
    bias_hidden = np.random.uniform(-1.0,1.0,
    size=(1,n_hidden))
    w2 = np.random.uniform(-1.0,1.0,
    size=(n_hidden,n_output))
    bias_output = np.random.uniform(-1.0,1.0,
    size=(1,n_output))
    return w1,bias_hidden,w2,bias_output





def encode_labels(y, num_labels):
        """ Encode labels into a one-hot representation"""
        # print "y shape = ",y.shape[0]
        onehot = np.zeros((y.shape[0],num_labels))
        # print onehot
        for i in range(y.shape[0]):
            onehot[i,y[i]-1] = 1.0
        return onehot



def get_cost(y_enc, output):
        """ Compute the cost function."""
        cost = - np.sum(y_enc*np.log(output))
        cost = cost
        return cost/y_enc.shape[0] #average cost


def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def dsigmoid(x):
    return x * (1. - x)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1. - x * x

def ReLU(x):
    return np.maximum(x,0)

def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:
        return e / np.array([np.sum(e, axis=1)]).T


def forward_pass(X,w1,bh,w2,bo):
    hidden_layer_input = np.dot(X,w1)+bh
    if activation == "sigmoid":
        hidden_output = sigmoid(hidden_layer_input)
    elif activation == "relu":
        hidden_output = ReLU(hidden_layer_input)
    elif activation == "tanh":
        hidden_output = tanh(hidden_layer_input)
    output_layer_input = np.dot(hidden_output,w2)+bo
    output = softmax(output_layer_input)
    return hidden_output,output

def back_propagation(hidden_output,output,one_hot,w1,bh,w2,bo):

    # update output layer weights
    error_output = (output-one_hot)/output.shape[0]

    update_out = np.dot(hidden_output.T,error_output)
    update_out_bias = np.sum(error_output,axis = 0,keepdims = True)
    w2 -= eta*update_out
    bo -= eta*update_out_bias

    # update hidden layer weights
    if activation == "sigmoid":
        error_hidden = np.multiply(dsigmoid(hidden_output),error_output.dot(w2.T))
    elif activation == "tanh":
        error_hidden = np.multiply(dtanh(hidden_output),error_output.dot(w2.T))
    elif activation == "relu":
        error_hidden = np.multiply(1,error_output.dot(w2.T))

    # update_hidden = np.dot(X.T,error_hidden)
    update_hidden = X.T.dot(error_hidden)
    update_hidden_bias = np.sum(error_hidden,axis = 0,keepdims=True)
    w1 -= eta*update_hidden
    bh -= eta*update_hidden_bias
    return w1,bh,w2,bo

# Predict label on testing data
def predict(w1,bh,w2,bo,test):
    x = []
    for i in range(len(test)):
        x.append(test[i][:-1])
    x = np.array(x,dtype=int)
    net1 = np.dot(x,w1)+bh
    if activation == "sigmoid":
        out1 = sigmoid(net1)
    elif activation == "relu":
        out1 = ReLU(net1)
    elif activation == "tanh":
        out1 = tanh(net1)

    net2 = np.dot(out1,w2)+bo
    out2 = softmax(net2)
    return out2


# Calculate accuracy for predicted data
def accuracy(output,test):
    output = output.tolist()
    acc = 0
    for i in range(len(output)):
        l = output[i]
        index = l.index(max(l))
        # print "index of max = ",index

        if index+1 == test[i][-1]:
            acc += 1
    # print "acc = ",acc
    return ((acc*1.0)/len(test))*100

scores = []
for i in range(no_of_folds):

    newdata = split()
    test = newdata[i]
    train = []
    for j in range(no_of_folds):
        if j!=i:
            for z in newdata[j]:
                train.append(z)

    X = []
    Y = []
    for k in range(len(train)):
        X.append(train[k][0:-1])
        Y.append(train[k][-1])

    X = np.array(X,dtype=int)
    Y = np.array(Y,dtype=int)

    w1,bh,w2,bo = init_weights(X.shape[1],neurons,3)
    one_hot = encode_labels(Y,3)

    for j in range(epoch):

        hidden_output,output = forward_pass(X,w1,bh,w2,bo)
        w1,bh,w2,bo = back_propagation(hidden_output,output,one_hot,w1,bh,w2,bo)

    result = predict(w1,bh,w2,bo,test)
    acc = accuracy(result,test)
    # print "Accuracy = ",acc
    scores.append(acc)

print "K fold accuracy = ",scores
print "Average accuracy = ",sum(scores)/len(scores)

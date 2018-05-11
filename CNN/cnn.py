import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve

img = Image.open('car.png').resize((32,32))
data = np.asarray(img)
print "Input image shape = ",data.shape


filter_size = 5
stride = 1

def relu(x):
    return np.maximum(x,0)

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:
        return e / np.array([np.sum(e, axis=1)]).T

# Convolution operation
def convolve(data,no_of_filters):
    dim = data.shape[2]
    h_range = int((data.shape[1] - filter_size) / stride) + 1
    # print "hr == ",h_range
    v_range = int((data.shape[0] - filter_size) / stride) + 1
    # print "vr == ",v_range
    output = np.ndarray((h_range,v_range,no_of_filters),dtype=int)
    filter = np.random.uniform(-1.0,1.0,size=(filter_size,filter_size,no_of_filters))
    for k in range(no_of_filters):
        for i in range(v_range):
            for j in range(h_range):
                x = data[i:i+filter_size,j:j+filter_size,k%dim]
                # print x.shape
                y = filter[:,:,k]
                # print y.shape
                output[i][j][k] = np.sum(np.multiply(x,y))

    return output
con1 = convolve(data,6)
con1 = relu(con1)
# print "Output after first convolution = \n",con1
print "Shape after first convolution = \n",con1.shape

img1 = Image.fromarray(con1,'RGB')
img1 = img1.resize((312,312))
img1.save("con1.jpg")

pool_size = 2

# Max pooling
def maxpool(output,no_of_filters):
    new = np.ndarray((output.shape[0]//2,output.shape[1]//2,no_of_filters),dtype=float)
    for i in range(no_of_filters):
        new[:,:,i] = output[:,:,i].reshape(output.shape[0]//2,pool_size,output.shape[1]//2,pool_size).max(axis=(1,3))
    return new


pool1 = maxpool(con1,6)
# print "Output after first subsampling = \n",pool1
print "Shape after first subsampling = ",pool1.shape

img1 = Image.fromarray(pool1,'RGB')
img1 = img1.resize((312,312))
img1.save("pool1.jpg")

con2 = convolve(pool1,16)
con2 = relu(con2)
# print "Output after second convolution = \n",con2
print "Shape after second convolution = ",con2.shape

img1 = Image.fromarray(con2,'RGB')
img1 = img1.resize((312,312))
img1.save("con2.jpg")

pool2 = maxpool(con2,16)
# print "Output after second subsampling = \n",pool2
print "Shape after second subsampling = ",pool2.shape

img1 = Image.fromarray(pool2,'RGB')
img1 = img1.resize((312,312))
img1.save("pool2.jpg")

output = pool2.reshape(pool2.shape[0]*pool2.shape[1]*pool2.shape[2],1)

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

def forward_pass(X,w1,bh,w2,bo):
    hidden_layer_input = np.dot(X.T,w1)+bh

    # hidden_output = sigmoid(hidden_layer_input)
    # hidden_output = relu(hidden_layer_input)
    hidden_output = tanh(hidden_layer_input)
    output_layer_input = np.dot(hidden_output,w2)+bo
    # output = sigmoid(output_layer_input)
    # output = relu(output_layer_input)
    output = tanh(output_layer_input)
    return output

# Fully connected layer
def fc(output):
    n_hidden = 84
    n_output = 10
    w1,bh,w2,bo = init_weights(output.shape[0],n_hidden,n_output)
    result = forward_pass(output,w1,bh,w2,bo)
    return result

filt = np.random.uniform(-1.0,1.0,size=(pool2.shape[0]*pool2.shape[1]*pool2.shape[2],120))
output = np.dot(filt.T,output)
# print "output shape == ",output.shape

result = fc(output)

res = softmax(result)
print "Result = \n",res
print "Result shape = ",res.shape


# img = Image.fromarray(output)
# img.show()

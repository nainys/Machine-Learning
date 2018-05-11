import numpy as np
import matplotlib.pyplot as plt

class1 = [[3,3,1],[3,0,1],[2,1,1],[0,2,1]]
class2 = [[-1,1,-1],[0,0,-1],[-1,-1,-1],[1,0,-1]]

x = [[3,3],[3,0],[2,1],[0,2],[-1,1],[0,0],[-1,-1],[1,0]]
y = [1,1,1,1,-1,-1,-1,-1]

# Augmented Dataset
for i in range(len(x)):
    x[i].append(1)


x = np.array(x,dtype=float)
y = np.array(y,dtype=float)

x_matrix = np.matrix(x)
y_matrix = np.matrix(y)

xt = x_matrix.getT()
yt = y_matrix.getT()

# weight vector
weight = xt*x
weight = weight.getI()
weight = weight*xt*yt
weight  = np.array(weight)
weight = weight.round(2)
print weight

# Elements of weight are coefficients of classifier line ax + by + c = 0
# a = weight[0], b = weight[1], c = weight[2]

c1_x = []
for i in class1:
    c1_x.append(i[0])

c1_y = []
for i in class1:
    c1_y.append(i[1])

c2_x = []
for i in class2:
    c2_x.append(i[0])

c2_y = []
for i in class2:
    c2_y.append(i[1])

# plot dataset
c1 = plt.scatter(c1_x, c1_y, c = "red", label = "class1")
c2 = plt.scatter(c2_x, c2_y, c = "blue", label = "class2")
plt.suptitle("Least squares approach - C1 vs C2 - Dataset 1", fontsize = 12)
plt.xlabel("x1", fontsize = 10)
plt.ylabel("x2", fontsize = 10)

# Plot classifier
xx = np.array(range(-2, 5))
# From ax + by + c = 0, y = (-ax - c)/b
yy = eval("xx * (-weight[0] / weight[1]) - (weight[2] / weight[1])")
plt.plot(xx, yy, label = "least square classifier")

plt.legend(loc = 'upper left')
plt.show()

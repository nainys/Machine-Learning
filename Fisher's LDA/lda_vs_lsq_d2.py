import numpy as np
import matplotlib.pyplot as plt
import fisher_lda_data2

# Dataset 1
class1 = [[3,3],[3,0],[2,1],[0,1.5]]
class1 = np.array(class1)
# print class1
class2 = [[-1,1],[0,0],[-1,-1],[1,0]]
class2 = np.array(class2)

# Mean vectors
c1_mean = np.mean(class1,axis = 0)
c2_mean = np.mean(class2,axis = 0)

# Within class scatter
c1_wcs = np.dot((class1-c1_mean).T,(class1-c1_mean))
c2_wcs = np.dot((class2-c2_mean).T,(class2-c2_mean))
total_wcs = c1_wcs + c2_wcs
total_wcs.round(2)


# Fisher's linear discriminant
w = np.dot(np.linalg.inv(total_wcs),(c1_mean-c2_mean))
w = np.array(w)

# Unit vector
wu = w/np.linalg.norm(w)
print "Unit vector in direction of discriminant - ",wu

slope = wu[1]/wu[0]


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

# print c1_x
# print c1_y
# print c2_x
# print c2_y

# Plot dataset
c1 = plt.scatter(c1_x, c1_y, c = "red", label = "C1")
c2 = plt.scatter(c2_x, c2_y, c = "blue", label = "C2")
plt.suptitle("Fisher's linear discriminant - C1 vs C2 - Dataset 2", fontsize = 12)
plt.xlabel("x1", fontsize = 10)
plt.ylabel("x2", fontsize = 10)


# Fisher's linear discriminant plot

# Class 1 plot
projection1 = np.dot(class1, wu)
projections1 = []
for value in projection1:
	projections1.append(np.dot(value, wu))

projections1 = np.array(projections1)


projections_x = []
for i in projections1:
    projections_x.append(i[0])

projections_y = []
for i in projections1:
    projections_y.append(i[1])

plt.scatter(projections_x,projections_y,marker = 's', c="red", s = 20)
plt.plot(projections_x,projections_y, "rx", marker = 's',label = "C1 projection")

for i in range(len(projections1)):
	plt.plot([class1[i][0], projections1[i][0]], [class1[i][1], projections1[i][1]], 'r--')


# Class 2 plot
projection2 = np.dot(class2, wu)
projections2 = []
for value in projection2:
	projections2.append(np.dot(value, wu))

projections2 = np.array(projections2)


projections_x = []
for i in projections2:
    projections_x.append(i[0])

projections_y = []
for i in projections2:
    projections_y.append(i[1])


plt.scatter(projections_x,projections_y,marker = 's',c="blue",s = 20)
plt.plot(projections_x,projections_y, "bx", marker = 's', label = "C2 projection")

for i in range(len(projections2)):
	plt.plot([class2[i][0], projections2[i][0]], [class2[i][1], projections2[i][1]], 'b--')


x = np.array(range(-2, 5))
y = eval("(x - projections2[0][0]) * slope + projections2[0][1]")
plt.plot(x, y, label = 'Fisher\'s discriminant', c = "green")



# Classifier using perceptron

def train_percep(x,y,e):
    w = np.zeros(len(x[0]))
    b = 0.0
    epochs = 0
    misclassify = True
    while misclassify and epochs < e:
        misclassify = False
        for i in xrange(len(x)):
            if (np.dot(x[i],w) + b)*y[i] <=0:
                w = w + y[i]*x[i]
                b = b+y[i]
                misclassify = True
        epochs = epochs+1
    return w,b

x = []
y = []
for p in projections1:
    x.append(p)
    y.append(1)
for p in projections2:
    x.append(p)
    y.append(-1)



weight,b = train_percep(x,y,10000)

# Plot the classifier
x = np.array(range(-2, 5))
y = eval("x * (-weight[0] / weight[1]) - (b / weight[1])")
plt.plot(x, y, 'c--', label = 'Fisher\'s classifier')

plt.legend(loc = 'upper left')
plt.show()

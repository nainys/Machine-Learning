import matplotlib.pyplot as plt
import vanilla_percep_ionosphere
import voted_percep_ionosphere

voted = voted_percep_ionosphere.avg_acc
vanilla = vanilla_percep_ionosphere.avg_acc

epochs = [10,15,20,25,30,35,40,45,50]

plt.scatter(epochs,voted,c="red",label = "voted")
plt.scatter(epochs,vanilla,c="blue",label = "vanilla")

plt.plot(epochs,voted,c="red")
plt.plot(epochs,vanilla,c="blue")

plt.suptitle("Vanilla perceptron vs Voted perceptron - Ionosphere Dataset",fontsize = 12)
plt.xlabel("Epochs",fontsize = 10)
plt.ylabel("Accuracy %",fontsize = 10)
plt.legend(loc = 'upper right')
plt.show()

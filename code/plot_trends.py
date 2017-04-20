import pandas as pd
import matplotlib.pyplot as plt


# plots the data from in progess training algorithms for different
# numbers of trees (75, 100, 125, 150) in the random forests algo.

n = ['1','2','3']
trees75 = pd.read_csv("trees75.txt", delimiter="|",names=n)
trees100 = pd.read_csv("trees100.txt", delimiter="|",names=n)
trees125 = pd.read_csv("trees125.txt", delimiter="|",names=n)
trees150 = pd.read_csv("trees150.txt", delimiter="|",names=n)

plt.figure()
trees75['2'].plot()
trees100['2'].plot()
trees125['2'].plot()
trees150['2'].plot()
plt.legend(['75 trees','100 trees','125 trees','150 trees'])
plt.xlabel('iteration number')
plt.ylabel('metric')
plt.show()

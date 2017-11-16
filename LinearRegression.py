import pandas as pd
import numpy as np
from matplotlib import pyplot as pl
data = pd.read_excel(open('./data/Quangdohoi.xlsx','rb'))
data = np.array(data)

n = np.shape(data)[0]
m = np.shape(data)[1]

X = data[:, 0:m -1]
y = data[:, m-1:m]

mu = np.mean(X)
sigma = np.std(X)
X = (X - mu) / sigma
X = np.insert(X,0, 1, axis=1)

alpha = 0.01
num_iters = 500

theta = np.zeros((m,1))

for iter in range(1, num_iters):
      h = np.dot(X,theta)
      theta = theta - alpha * np.dot(X.T,(h - y))/n
print theta

pl.plot(X[:,1], y, '+r')
pl.plot(X, theta[1]*X + theta[0])
pl.show()
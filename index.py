import perceptronS as ps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def getDataOfColumnsCsv(df,column):
    lista = []
    for i in range(len(df)):
        aux = df[i]
        lista.append(aux[column])
    
    return np.asarray(lista)

df = pd.read_csv('XOR_tst.csv',header=0)
test = df.values
aux = []
x1 = getDataOfColumnsCsv(test,0)
x2 = getDataOfColumnsCsv(test,1)
Y = getDataOfColumnsCsv(test,2)
for i in range(len(Y)):
    p = int(Y[i])
    aux.append(p)

Y = np.array(aux)

X = np.concatenate((x1.reshape(-1,1).T,x2.reshape(-1,1).T),axis=0)
Y = Y.reshape(1,-1)

model = ps.perceptronSimple(2, 0.5)
model.fit(X, Y,100)
print(model.predict(X))

# dibujo
p = X.shape[1]
for i in range(p):
    if(Y[:,i] <= 0):
        plt.plot(X[0, i], X[1, i], 'or')
    else:
        plt.plot(X[0, i], X[1, i], 'og')


plt.title('test perceptron')
plt.grid('on')
# plt.xlim([-2, 2])
# plt.ylim([-2, 2])
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
model.drawPerceptron2d(model)
plt.show()

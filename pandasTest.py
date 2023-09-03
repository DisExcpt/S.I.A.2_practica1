import pandas as pd
import numpy as np

def getDataOfColumnsCsv(df,column):
    lista = []
    for i in range(len(df)):
        aux = df[i]
        lista.append(aux[column])
    lista = np.asarray(lista)
    return lista.reshape(-1,1)


df = pd.read_csv('OR_tst.csv',header=0)


test = df.values
aux = []
x1 = getDataOfColumnsCsv(test,0)
x2 = getDataOfColumnsCsv(test,1)
Y = getDataOfColumnsCsv(test,2)
for i in range(len(Y)):
    p = int(Y[i])
    aux.append(p)

Y = np.array(aux)
print(Y)
# X = np.concatenate((x1.T,x2.T),axis=0)
# print(X.shape)
# print(len(x1))
# print(len(x2))
# print(len(Y))

X = np.concatenate((x1.T,x2.T),axis=0)

Y = Y.reshape(1,-1)
print(X.shape)
print(Y.shape)
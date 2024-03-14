import numpy as np

def activateFunction(x):
    return 0 if x < 0.5 else 1

def network(house, genre, attr):
    x = np.array([house, genre, attr])
    w11 = [0.3, 0.3, 0]
    w12 = [0.4, -0.5, 1]
    weight1 = np.array([w11, w12]) #матрица 2x3
    weight2 = np.array([-1, 1]) #матрица 1х3

    sum_hidden = np.dot(weight1, x)
    print('Значение сумм на нейронах скрытого слоя:' + str(sum_hidden))

    out_hidden = np.array([activateFunction(x) for x in sum_hidden])
    print('Значение на выходах нейронов скрытого слоя:' + str(out_hidden))

    sum_end = np.dot(weight2, out_hidden)
    y = activateFunction(sum_end)
    print("Выходное значение НС:" +str(y))

    return y

house = 1
genre = 1
attr = 0

res = network(house, genre, attr)

if (res):
    print("Yes")
else:
    print("No")

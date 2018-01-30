import numpy as np
import scipy.io

from model import Calculation
import matplotlib.pyplot as plt

fileName = 'data/reuters.mat'
mat = scipy.io.loadmat(fileName)

# attributes = mat['TOPICS_COLUMN_NAMES'][0][0:2500]
# data = mat['TOPICS'][0:2500]

attributes = np.array(['x1', 'x2', 'x3', 'x4', 'x5', 'x6'])
data = np.array([
    [0, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 0, 0],
    [1, 1, 1, 0, 1, 1],
    [1, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 1, 1],
    [0, 1, 1, 1, 1, 1],
    [0, 1, 0, 1, 1, 1],
    [0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1],
    [0, 0, 1, 1, 0, 0],
])


rules, fis = Calculation.apriori(data)
aprioriLine = Calculation.findPareto(rules)

for value in rules:
    plt.plot(value['support'], value['confidence'], 'k*')

tmp = np.array(list(aprioriLine.keys()))
tmp2 = np.array(list(aprioriLine.values()))
sort_index = np.argsort(np.array(tmp))
plt.plot(tmp[sort_index], tmp2[sort_index])

plt.show()

r = []
for i in rules:
    a, b = i['att1'], i['att2']
    s = ""
    for x in a:
        s += attributes[x][0] + " "
    s += "=> "
    for x in b:
        s += attributes[x][0] + " "

    a = {'rule': s,
         'confidence': i['confidence'],
         'support': i['support'], }
    r.append(a)

print(r)

test_arr = []
for value in rules:
    test_arr.append((value['support'], value['confidence']))

print("Unique points: %d" % len(set(test_arr)))

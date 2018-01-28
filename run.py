import scipy
import numpy as np
from model import Calculation

fileName = 'data/reuters.mat'
mat = scipy.io.loadmat(fileName)

# attributes = mat['TOPICS_COLUMN_NAMES'][0][0:1500]
# data = mat['TOPICS'][0:1500]

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
])  # apriori.png
# 7, 0.5

# data = np.array([
#    [1, 0, 1, 1, 0],
#    [0, 1, 1, 0, 1],
#    [1, 1, 1, 0, 1],
#    [0, 1, 0, 0, 1],
#    [1, 0, 1, 0, 1],
# ])
# 2, 0.6

# data = np.array([
#    [1, 1, 0, 0, 1],
#    [0, 1, 0, 1, 0],
#    [0, 1, 1, 0, 0],
#    [1, 1, 0, 1, 0],
#    [1, 0, 1, 0, 0],
#    [0, 1, 1, 0, 0],
#    [1, 0, 1, 0, 0],
#    [1, 1, 1, 0, 1],
#    [1, 1, 1, 0, 0],
# ])
# 2, 0.7

minimumSupportCount = 7
minimumConfidence = 0.5

rules, fis = Calculation.apriori(data, minimumSupportCount, minimumConfidence)
aprioriLine = Calculation.findPareto(rules)

for value in rules:
    plt.plot(value['x'], value['confidence'], '*')

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
        s += attributes[x] + " "
    s += "=> "
    for x in b:
        s += attributes[x] + " "

    a = {'rule': s,
         'confidence': i['confidence'],
         'support': i['support'], }
    r.append(a)

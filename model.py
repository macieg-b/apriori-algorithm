import numpy as np
import itertools


class Calculation:
    minimumSupportCount = 7
    minimumConfidence = 0.5

    def __init__(self):
        pass

    @staticmethod
    def scaleByOperatorAND(data):
        newData = []
        for row in data:
            tmpValue = 1
            for i in row:
                tmpValue &= i
            newData.append(tmpValue)

        return np.array(newData).transpose()

    @staticmethod
    def createNewCandidates(candidates, keys, k):
        listOfKeys = list(keys)
        if type(candidates[0]) is tuple:
            listOfKeys = list(set(itertools.chain(*list(listOfKeys))))

        return list(itertools.combinations(listOfKeys, k + 1))

    @staticmethod
    def apriori(binTab):
        iteration = 0
        candidates = range(len(binTab[0]))
        frequentItemSets = []
        lenBinTab = len(binTab)

        while True:
            iteration += 1

            frequentItems = dict()
            for index in candidates:
                column = binTab[:, index]

                if type(index) is tuple:
                    column = Calculation.scaleByOperatorAND(column)

                sumValue = sum(column)
                if sumValue >= Calculation.minimumSupportCount:
                    frequentItems[index] = sumValue

            candidates = Calculation.createNewCandidates(candidates, frequentItems.keys(), iteration)

            if not candidates:
                break

            frequentItemSets.append(frequentItems.copy())

        rules = []
        new_frequentItemSets = {}

        for fis in frequentItemSets:
            for key, value in fis.items():
                if type(key) is tuple:
                    new_frequentItemSets[key] = value
                else:
                    new_frequentItemSets[tuple([key])] = value

        for key1, value1 in new_frequentItemSets.items():
            for key2, value2 in new_frequentItemSets.items():

                var = key1 == key2
                for x in key1:
                    if x in key2:
                        var = True
                        break

                if var:
                    continue

                tmpJoin = tuple(set(key1) | set(key2))
                abcde = new_frequentItemSets.get(tmpJoin)
                if abcde != None:
                    rule = str(key1) + '=>' + str(key2)
                    conf = new_frequentItemSets[tmpJoin] / new_frequentItemSets[key1]
                    supp = new_frequentItemSets[tmpJoin] / lenBinTab

                    r = {
                        'rule': rule,
                        'confidence': conf,
                        'x': new_frequentItemSets[key1],
                        'att1': tuple(key1),
                        'att2': tuple(key2),
                        'support': supp,
                    }

                    if r['confidence'] >= Calculation.minimumConfidence:
                        rules.append(r)

        return rules, new_frequentItemSets

    @staticmethod
    def findPareto(rules):
        d = dict()

        for value in rules:
            k = value['support']
            v = value['confidence']
            tmp = d.get(k, v)

            if tmp <= v:
                d[k] = v

        tmp = np.array(list(d.keys()))
        tmp2 = np.array(list(d.values()))
        sort_index = np.argsort(np.array(tmp))

        for i in range(1, len(sort_index) - 1):
            if tmp2[sort_index[i + 1]] >= tmp2[sort_index[i]]:
                d[tmp[sort_index[i]]] = tmp2[sort_index[2]]

        if d[tmp[sort_index[0]]] < d[tmp[sort_index[1]]]:
            d[tmp[sort_index[0]]] = None

        return d

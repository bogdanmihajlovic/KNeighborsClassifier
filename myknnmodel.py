import math
class KNN:
    def __init__(self, k):
        self.k = k
        self.train = None
        self.label = None

    def fit(self, train, label):
        self.train = train
        self.label = label

    def predict(self, test):
        train_len = len(self.train)
        prediction = []
        for i in range(len(test)):
            all_distance = []
            val1 = test.iloc[i].tolist()
            for j in range(train_len):
                val2 = self.train.iloc[j].tolist()
                d = math.sqrt(sum([pow(el1 - el2, 2) for(el1, el2) in zip(val1, val2)]))
                all_distance.append([d, self.label.iloc[j]])

            all_distance.sort(key = lambda x: x[0])
            all_distance = all_distance[:self.k]
            count_types = dict()
            for elem in all_distance:
                d = elem[0]
                type = elem[1]
                if type in count_types:
                    count_types[type] += 1
                else:
                    count_types[type] = 1

            prediction.append(max(count_types, key=count_types.get))
        return prediction

    def score(self, x, y):
        pred = self.predict(x)
        succ=0
        y_list = y.tolist()
        for (i, j) in zip(pred, y_list):
            if i == j:
                succ += 1

        score = (succ*1.0)/len(pred)
        return score


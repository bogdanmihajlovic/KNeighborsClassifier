import math
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from myknnmodel import KNN


def train_model(model, input_train, output_train, input_test, output_test, txt):
    model.fit(input_train, output_train)
    my_pred = model.predict(input_test)
    ser_pred = pd.Series(data=my_pred, name="Predicted", index=input_test.index)
    my_res = pd.concat([input_test, output_test, ser_pred], axis=1)
    print(txt)
    print(my_res.head())
    print("Score: ", model.score(input_test, output_test), end="\n\n")


def show_diagram(data, names):
    for name in names:
        plt.scatter(data[name], data['type'])
        plt.title("title")
        plt.xlabel("x-" + name)
        plt.ylabel("y-type")
        plt.show()


def main():

    # 1. ucitavanje podataka i prikaz prvih pet redova
    data = pd.read_csv('cakes.csv')
    pd.set_option('display.width', None)
    print(data.head(), end="\n\n")

    # 2. prikaz informacija
    print(data.info(), end="\n\n")
    print(data.describe(), end="\n\n")
    print(data.describe(include=object), end="\n\n")

    # 3. eliminacija nevalidnih
    data = data.dropna()
    # data.where(data.flour.notnull(), inplace=True)

    # 4. korelaciona matrica
    plt.figure()
    sb.heatmap(data.drop(columns=["type"]).corr(), annot=True, fmt=".2f")
    plt.show()

    # 5. grafici TODO atribut tip
    show_diagram(data, ["flour", "eggs", "sugar", "milk", "butter", "baking_powder"])

    # 6. odabir atributa
    data_train = data.drop(columns=["type"])

    # 7. transofrmacija
    data_train = data_train.apply(lambda x: x*67 if x.name == 'eggs' else x)
    le = LabelEncoder()
    labels = pd.Series(le.fit_transform(data['type']))

    # 8. formiranje skupova za trening i testiranje
    input_train, input_test, output_train, output_test = train_test_split(data_train, labels, train_size=0.8,
                                                                          random_state=123, shuffle=True)

    # 9. realizacija treniranja i prikaz rezultata
    k = round(math.sqrt(len(input_train)))
    if k % 2 == 0:
        k = k + 1

    my_knn_model = KNN(k)
    knn_model = KNeighborsClassifier(n_neighbors=k)

    train_model(my_knn_model, input_train, output_train, input_test, output_test, "My model")
    train_model(knn_model, input_train, output_train, input_test, output_test, "KNeighborsClassifier")


if __name__ == "__main__":
    main()

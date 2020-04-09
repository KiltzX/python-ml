import pandas as pd
from sklearn import tree

df = pd.read_csv("objects.csv")

dados = df.filter(['peso','largura'])
nome = df.filter(['nome'])


clf = tree.DecisionTreeClassifier()
clf = clf.fit(dados, nome)
res1 = input('digite o peso : ')
res2 = input('Digite a largura : ')
resultado = clf.predict([[res1, res2]])

print(resultado)
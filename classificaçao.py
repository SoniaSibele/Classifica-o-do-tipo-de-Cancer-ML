
#Algoritmos de machine learning para classificar tipos de Câncer.

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
dados=load_breast_cancer()

#print(dados.DESCR) 

cancer=pd.DataFrame(data=dados.data, columns=dados.feature_names) 

cancer['Class']=dados.target 
print(cancer.head(4))
print(cancer.shape)

dist = cancer['Class'].value_counts() # 1- Benigno 0 = Máligno
print (dist)

colors=['#35b2de','#ffcb5a']
labels=cancer['Class'].value_counts().index
plt.pie(cancer['Class'].value_counts(),autopct='%1.1f%%',colors=colors) # ocorrências de cada classe em porcentagem
plt.legend(labels,bbox_to_anchor=(1.25,1),) 
plt.title('Porcentagem: Benignos x Malignos ')
plt.show()

nul = cancer.isnull().sum() 
#print(nul)
from sklearn.model_selection import train_test_split

# dividir a base de dados entre features e target
X= cancer.iloc[:,0:-1]
Y=cancer.iloc[:,-1] 

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.30,random_state=42) 

print('X treino',x_train.shape)
print('X test',x_test.shape)
print('Y treino',y_train.shape)
print('Y test',y_test.shape)

#Possíveis modelos

model = LogisticRegression() # Criar o modelo
#model = SVC()
#model = GaussianNB()
#model = DecisionTreeClassifier()
#model = RandomForestClassifier(n_estimators=100)
#model = knn=KNeighborsClassifier(n_neighbors=1) 

model.fit(x_train,y_train) # Treinar o modelo
y_pred= model.predict(x_test) # predizer
acc_model=round(metrics.accuracy_score(y_pred,y_test)*100,1) # avaliar a acurácia. previsões x resultados reais
print("{}% de acurácia".format(acc_model,))

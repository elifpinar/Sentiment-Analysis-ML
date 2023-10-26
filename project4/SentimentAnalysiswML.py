 # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

 #E-Ticaret Ürün Yorumları 
 # 0= Olumsuz Yorum 
 # 1= Olumlu Yorum 
 # 2= Nötr/Tarafsız Yorum

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#delimiter = veri setinde verilerin , yerine ; ile ayrıldığını belirtmek için 
df= pd.read_csv('C:/Users/PINAR/.spyder-py3/e-ticaret_urun_yorumlari.csv', delimiter=';') 
print(df.head(5))

#ön işleme
# X=df.iloc[:,0:1].values
# y=df.iloc[:,1].values
# X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=42)
# print("X'in ilk değeri :",X[0])
# print("y'nin ilk değeri :",y[0])

satir_sayisi = df.shape[0]
print("Satır sayısı:", satir_sayisi)
eksik_veriler=df.isnull()
eksik_veriler=eksik_veriler.sum()
print("Eksik Veri Sayısı:",  eksik_veriler)                     #metin:0 , durum:0 eksik veri yok
allowed_values = ["0", "1", "2"]                                # sadece bu değerlerin olması bekleniyor
column_name = "Durum"
unique_values = df[column_name].value_counts().to_dict()
print(unique_values)
# to_dict() yöntemi, DataFrame'deki her bir sütunu bir sözlük olarak döndürür.
# Bu sözlükte, her bir sütunun ismi, o sütunda bulunan tüm değerlerin bir listesi ile eşleştirilir.


for index, row in df.iterrows():                                 #satırlarını sırayla döndürür
    value = row[column_name]
    if str(value) not in allowed_values:
        print(f"Uyarı: {index+1}. satırda sütun '{column_name}' değeri '{value}' beklenen değerlerden farklıdır.")
        
sns.histplot(data=df, x="Durum", kde=True)
#kde=True  yoğunluğu gösteren bir yoğunluk eğrisi çizer





 
readComments=df.iloc[:,0:1].values                         #ilk sütunu aldım
commentList= readComments.tolist()                         #îlk sütunu bir listeye attım.
#print(commentList)
commentsArray=[]
for i in range(len(commentList)):
    review=re.sub(r'\W',' ',str(commentList[i]))           #noktalama işaretlerini boşluk ile değiştir
    review=review.lower()                                  #tüm harfleri küçük harfe çevir
    commentsArray.append(review)

#kelimeleri binary kodlara dönüştürmek için
# max_features=2500, en yüksek 2500 kelimeyi seçecektir.
#min_df, Bir kelimenin en az kaç belgede geçmesi gerektiğini belirleyen eşik değeridir
#max_df, Bir kelimenin en fazla kaç belgede geçebileceğini belirleyen eşik değeridir. Örneğin, max_df=0.6, bir kelimenin toplam belgelerin %60'ından fazlasında geçmemesi gerektiğini ifade eder


vectorizer= TfidfVectorizer(max_features=2500, min_df=1, max_df=0.6)
X=vectorizer.fit_transform(commentsArray).toarray() #x e attım
#print(X.shape)  (15170,2500)
y=df.iloc[:,1].values
#print(y)

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=42)    



#LOGISTIC REGRESSION 

classifier =LogisticRegression(max_iter=1000)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
print("LOGISTIC REGRESSION")
cm1=confusion_matrix(y_test,y_pred)
print(cm1)
print(classification_report(y_test,y_pred))
score=accuracy_score(y_test, y_pred)
print("accuracy score",score)




#DECISION TREE KARAR AĞAÇLARI
# gini  index= örnekteki eşitsizliğin ölçüsü 0= homojen, 1=elementler arasında maksimum eşitsizlik vardır
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(criterion='gini',max_depth=52, random_state=42)    #max_depth=4 için 0.59, 8için0.66, 0.80 
                                                                                 #50 den 100e kadar 0.01 değişim oluyor
tree.fit(X_train,y_train)
y_pred=tree.predict(X_test)
print("KARAR AĞAÇLARI")
cmforDC=confusion_matrix(y_test, y_pred)
print(cmforDC)
print(classification_report(y_test,y_pred))
scoreforDC=accuracy_score(y_test, y_pred)
print("accuracy score",scoreforDC)




#GAUSSIAN NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
acc_gaussian = gaussian.score(X_train, y_train)
print("Naive Bayes Eğitim Doğruluğu: %.2f " %acc_gaussian) #virgülden sonra yalnızca 2 basamağı göstermek için 
y_pred = gaussian.predict(X_test) 
acc_naiveBayes=accuracy_score(y_test,y_pred)
print("Naive Bayes Test Doğruluğu: %.2f " %acc_naiveBayes)

cmforNaiveBayes = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall =  recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print("GAUSSIAN NAIVE BAYES")
print('Confusion matrix\n ', cmforNaiveBayes)
print(classification_report(y_test, y_pred))
print("accuracy score",accuracy_score(y_test, y_pred))

print('Precision: %.2f ' %precision)
print('Recall: %.2f ' %recall)
print('f1-score: %.2f ' %f1)




#DESTEK VEKTÖR MAKİNELERİ (Support Vector Machines, SVM)
from sklearn.svm import SVC
print("DESTEK VEKTÖR MAKİNELERİ")
#Polynomial Kernel
svc_siniflandirici = SVC(kernel='poly', degree=8)
svc_siniflandirici.fit(X_train, y_train)
y_pred = svc_siniflandirici.predict(X_test)                 # Sınıflandırıcı test ediliyor
print("Polynomial Kernel")
print(confusion_matrix(y_test, y_pred))                     # Performans sonuçları yazdırılıyor
print(classification_report(y_test, y_pred))
print("----------------------------------")

#Gaussian Kernel
svc_siniflandirici= SVC(kernel='rbf')
svc_siniflandirici.fit(X_train, y_train)
y_pred = svc_siniflandirici.predict(X_test)
print("Gaussian Kernel")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("----------------------------------")

#Sigmoid Kernel
svc_siniflandirici = SVC(kernel='sigmoid')
svc_siniflandirici.fit(X_train, y_train)
y_pred = svc_siniflandirici.predict(X_test)
print("Sigmoid Kernel")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("-----------------------------------")






#RASTGELE ORMAN (Random Forest)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier as randomforest

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

num_trees = 100
max_features = 3
RF = randomforest (criterion='gini', bootstrap=True,
                                n_estimators=num_trees, max_features=max_features)

model =RF. fit(X_train, y_train)
y_pred = model.predict(X_test)
print("RASTGELE ORMAN")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("accuracy score:",accuracy_score(y_test, y_pred))



#KNN K EN YAKIN KOMŞU ALGORİTMASI
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)
y_pred=knn.predict(X_test)
print("KNN")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))









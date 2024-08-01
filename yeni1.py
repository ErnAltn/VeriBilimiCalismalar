import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Veri setini yükle
data = pd.read_csv("meteorite-landings.csv")

# Gerekli sütunları seç
X = data[['name', 'nametype', 'recclass', 'mass', 'fall', 'year', 'reclat', 'reclong']]
y = data['fall']

# Eksik değerleri doldur
imputer = SimpleImputer(strategy='most_frequent')
X = imputer.fit_transform(X)

# Etiketleri kodla
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Veri setini eğitim ve test olarak böle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# K-En Yakın Komşu (KNN) sınıflandırıcısı
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
knn_y_pred = knn_model.predict(X_test)

knn_cm = confusion_matrix(y_test, knn_y_pred)
knn_cr = classification_report(y_test, knn_y_pred)

print("K-En Yakın Komşu (KNN) Sınıflandırıcısı:")
print("Karmaşıklık Matrisi:")
print(knn_cm)
print("Sınıflandırma Raporu:")
print(knn_cr)

# Rastgele Orman sınıflandırıcısı
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

rf_cm = confusion_matrix(y_test, rf_y_pred)
rf_cr = classification_report(y_test, rf_y_pred)

print("Rastgele Orman Sınıflandırıcısı:")
print("Karmaşıklık Matrisi:")
print(rf_cm)
print("Sınıflandırma Raporu:")
print(rf_cr)

# Naive Bayes sınıflandırıcısı
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_y_pred = nb_model.predict(X_test)

nb_cm = confusion_matrix(y_test, nb_y_pred)
nb_cr = classification_report(y_test, nb_y_pred)

print("Naive Bayes Sınıflandırıcısı:")
print("Karmaşıklık Matrisi:")
print(nb_cm)
print("Sınıflandırma Raporu:")
print(nb_cr)

# Karar Ağaçları sınıflandırıcısı
dt_model = DecisionTreeClassifier(max_depth=2)
dt_model.fit(X_train, y_train)
dt_y_pred = dt_model.predict(X_test)

dt_cm = confusion_matrix(y_test, dt_y_pred)
dt_cr = classification_report(y_test, dt_y_pred)

print("Karar Ağaçları Sınıflandırıcısı:")
print("Karmaşıklık Matrisi:")
print(dt_cm)
print("Sınıflandırma Raporu:")
print(dt_cr)

# ROC eğrilerini çizdir
knn_probs = knn_model.predict_proba(X_test)[:, 1]
rf_probs = rf_model.predict_proba(X_test)[:, 1]
nb_probs = nb_model.predict_proba(X_test)[:, 1]
dt_probs = dt_model.predict_proba(X_test)[:, 1]

knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_probs)
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_probs)

plt.figure()
plt.plot(knn_fpr, knn_tpr, label='K-En Yakın Komşu (KNN)')
plt.plot(rf_fpr, rf_tpr, label='Rastgele Orman')
plt.plot(nb_fpr, nb_tpr, label='Naive Bayes')
plt.plot(dt_fpr, dt_tpr, label='Karar Ağaçları')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Oranı')
plt.ylabel('True Positive Oranı')
plt.title('ROC Eğrileri')
plt.legend()
plt.show()

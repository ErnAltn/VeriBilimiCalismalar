import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, accuracy_score

# Verileri oku
data = pd.read_csv('meteorite-landings.csv')

# Gerekli öznitelikleri ve hedef değişkeni seç
features = data[['mass', 'reclat', 'reclong']]
target = data['fall']

# Eksik değerleri doldur
imputer = SimpleImputer(strategy='mean')
features_filled = imputer.fit_transform(features)

# Eğitim ve test verilerine ayır
X_train, X_test, y_train, y_test = train_test_split(features_filled, target, test_size=0.2, random_state=42)

# Özellikleri ölçeklendir
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN sınıflandırıcısını oluştur
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Modeli eğit
knn_classifier.fit(X_train_scaled, y_train)

# Test verileri üzerinde tahmin yap
y_pred = knn_classifier.predict(X_test_scaled)

# Karmaşıklık matrisini oluştur
cm = confusion_matrix(y_test, y_pred)

# Başarı oranını hesapla
accuracy = accuracy_score(y_test, y_pred)

# Karmaşıklık matrisini görselleştir
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Karmaşıklık Matrisi')
plt.show()

# Başarı oranını göster
print(f'Başarı Oranı: {accuracy:.2f}')

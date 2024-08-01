import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Verileri oku
data = pd.read_csv('meteorite-landings.csv')

# Gerekli öznitelikleri seç
features = data[['mass', 'reclat', 'reclong']]

# Hedef değişkeni seç
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

# Destek Vektör Makineleri sınıflandırıcısını oluştur
svm_classifier = SVC()

# Modeli eğit
svm_classifier.fit(X_train_scaled, y_train)

# Test verileri üzerinde tahmin yap
y_pred = svm_classifier.predict(X_test_scaled)

# Sınıflandırma raporunu görüntüle
print(classification_report(y_test, y_pred))

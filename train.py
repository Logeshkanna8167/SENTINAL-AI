import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ===== LOAD DATASET =====
data = pd.read_csv("dataset.csv")

X = data[["vibration"]]
y = data["label"]

# ===== SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== RANDOM FOREST =====
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# ===== EVALUATE =====
pred = model.predict(X_test)

print(classification_report(y_test, pred))

# ===== SAVE =====
joblib.dump(model, "model.pkl")

print("✅ Random Forest model saved as model.pkl")
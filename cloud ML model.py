import joblib
import pandas as pd
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
file_path = r"C:\Users\kanka\OneDrive\Desktop\cloud\breast+cancer+wisconsin+diagnostic.zip"

with ZipFile(file_path) as z:
    with z.open("wdbc.data") as f:
        data = pd.read_csv(f, header=None)

# 2. Separate input and output
X = data.iloc[:, 2:]       # 30 input features
y = data.iloc[:, 1]        # Diagnosis: B or M

# 3. Split data (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 4. Train the model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
# Save trained model
joblib.dump(model, "breast_cancer_model.pkl")

print("Model saved successfully!")

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Display results
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Test one example
sample = X_test.iloc[[0]]
prediction = model.predict(sample)

if prediction[0] == "B":
    print("Prediction: Benign (non-cancerous)")
else:
    print("Prediction: Malignant (cancerous)")

import pandas as pd
from scipy.io.arff import loadarff
import numpy as np
# Load ARFF file
data = loadarff("Autism-Adult-Data.arff")  # Replace with your file path
df = pd.DataFrame(data[0])

# Convert byte strings to regular strings (if needed)
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.decode('utf-8')

# Check for missing values
print(df.isnull().sum())

# Check for infinite values
print(df.isin([np.inf, -np.inf]).sum())
# Option 1: Drop rows with missing values
df = df.dropna()

# Option 2: Fill missing values with a specific value (e.g., mean, median, or mode)
# Example: Fill numerical columns with the mean
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col].fillna(df[col].mean(), inplace=True)

# Example: Fill categorical columns with the mode
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Check again for missing values
print(df.isnull().sum())
import numpy as np

# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN values (if needed)
df = df.dropna()

# Check again for infinite values
print(df.isin([np.inf, -np.inf]).sum())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Encode categorical variables
label_encoder = LabelEncoder()
categorical_columns = ['gender', 'ethnicity', 'jundice', 'austim', 'contry_of_res', 'used_app_before', 'age_desc', 'relation', 'Class/ASD']
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Separate features and target
X = df.drop('Class/ASD', axis=1)  # Features
y = df['Class/ASD']  # Target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save preprocessed data (optional)
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Train SVM model
svm_model = SVC(kernel='linear', random_state=42)  # Linear kernel for simplicity
svm_model.fit(X_train, y_train)

# Predict on test data
y_pred = svm_model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the trained model (optional)
import joblib
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")
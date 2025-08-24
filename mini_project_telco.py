import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix 

"""Task 1: Perform EDA and Preprocessing"""

# Load Telco Customer-Churn Dataset
df_telco = pd.read_csv('Telco-Customer-Churn.csv') 

# Inspect the data
print("Dataset Info:")
print(df_telco.info())
print("\nDataset Description:")
print(df_telco.describe())

# Check for missing values
print("\nMissing values:")
print(df_telco.isnull().sum())

# Visualize churn distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df_telco)
plt.title("Churn Distribution")
plt.show()

# Fix TotalCharges column (it's often stored as object type with spaces)
print("\nFixing TotalCharges column...")
df_telco['TotalCharges'] = pd.to_numeric(df_telco['TotalCharges'], errors='coerce')

# Handle missing values in TotalCharges
df_telco['TotalCharges'].fillna(df_telco['TotalCharges'].median(), inplace=True)

# Encode target variable
le_target = LabelEncoder()
df_telco['Churn'] = le_target.fit_transform(df_telco['Churn'])

# Define features and target
X = df_telco.drop(columns=['Churn'])
y = df_telco['Churn']

print("\nFeature data types:")
print(X.dtypes)

# Encode categorical variables in features
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns to encode: {categorical_columns}")

le = LabelEncoder()
for column in categorical_columns:
    X[column] = le.fit_transform(X[column].astype(str))

print("\nAfter encoding:")
print(X.dtypes)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset (FIXED: test_size should be 0.42 or 0.2, not 42)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"\nDataset split:")
print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Train the Logistic Regression model
log_model = LogisticRegression(max_iter=1000)  # Increased iterations
log_model.fit(X_train, y_train)

# Train K-NN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Make predictions
log_pred = log_model.predict(X_test)
knn_pred = knn_model.predict(X_test)

# Evaluate models
print("\n" + "="*50)
print("LOGISTIC REGRESSION RESULTS")
print("="*50)
print(classification_report(y_test, log_pred))

print("\n" + "="*50)
print("K-NEAREST NEIGHBORS RESULTS")
print("="*50)
print(classification_report(y_test, knn_pred))

# Confusion Matrices
print("\n" + "="*50)
print("CONFUSION MATRICES")
print("="*50)

print("\nLogistic Regression Confusion Matrix:")
log_cm = confusion_matrix(y_test, log_pred)
print(log_cm)

print("\nK-NN Confusion Matrix:")
knn_cm = confusion_matrix(y_test, knn_pred)
print(knn_cm)

# Visualize confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Logistic Regression CM
sns.heatmap(log_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Logistic Regression\nConfusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# K-NN CM
sns.heatmap(knn_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('K-Nearest Neighbors\nConfusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Calculate and display accuracy scores
from sklearn.metrics import accuracy_score

log_accuracy = accuracy_score(y_test, log_pred)
knn_accuracy = accuracy_score(y_test, knn_pred)

print(f"\nModel Accuracies:")
print(f"Logistic Regression: {log_accuracy:.4f}")
print(f"K-Nearest Neighbors: {knn_accuracy:.4f}")

# Display feature importance for Logistic Regression
feature_names = X.columns if hasattr(X, 'columns') else [f'Feature_{i}' for i in range(X_train.shape[1])]
print(f"\nLogistic Regression Feature Coefficients (Top 10):")
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': log_model.coef_[0]
})
coef_df['Abs_Coefficient'] = abs(coef_df['Coefficient'])
print(coef_df.sort_values('Abs_Coefficient', ascending=False).head(10))
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

"""Task 1: Perform EDA and Preprocessing"""

# Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

print("Dataset Overview:")
print(f"Dataset shape: {df.shape}")
print(f"Target variable: MedHouseVal (Median House Value)")
print(f"Selected features: MedInc, HouseAge, AveRooms")

# Define features and target
X = df[['MedInc', 'HouseAge', 'AveRooms']]
y = df['MedHouseVal']

# Inspect data
print("\n" + "="*50)
print("DATA INFORMATION")
print("="*50)
print(df.info())

print("\n" + "="*50)
print("STATISTICAL SUMMARY")
print("="*50)
print(df.describe())

# Check for missing values
print("\n" + "="*50)
print("MISSING VALUES CHECK")
print("="*50)
missing_values = df.isnull().sum()
print(missing_values)
print(f"Total missing values: {missing_values.sum()}")

# Enhanced Visualizations
print("\n" + "="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

# 1. Distribution of target variable
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.hist(y, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribution of Median House Values')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')

# 2. Feature distributions
features = ['MedInc', 'HouseAge', 'AveRooms']
colors = ['lightcoral', 'lightgreen', 'lightsalmon']

for i, (feature, color) in enumerate(zip(features, colors)):
    plt.subplot(2, 3, i+2)
    plt.hist(X[feature], bins=30, alpha=0.7, color=color, edgecolor='black')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

# 3. Correlation heatmap
plt.subplot(2, 3, 5)
correlation_data = df[['MedInc', 'HouseAge', 'AveRooms', 'MedHouseVal']]
corr_matrix = correlation_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
plt.title('Correlation Matrix')

# 4. Scatter plots
plt.subplot(2, 3, 6)
plt.scatter(X['MedInc'], y, alpha=0.3, color='purple')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('Income vs House Value')

plt.tight_layout()
plt.show()

# Pairplot (uncomment if you want to see it)
print("Creating pairplot...")
plt.figure(figsize=(10, 8))
sns.pairplot(df[['MedInc', 'AveRooms', 'HouseAge', 'MedHouseVal']], 
             diag_kind='hist', plot_kws={'alpha': 0.6})
plt.suptitle('Pairplot of Selected Features', y=1.02)
plt.show()

# Feature scaling (optional but good practice)
print("\n" + "="*50)
print("FEATURE SCALING")
print("="*50)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

print("Original features statistics:")
print(X.describe())
print("\nScaled features statistics:")
print(X_scaled_df.describe())

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Comprehensive Model Evaluation
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

# Calculate multiple metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared Score: {r2:.4f}")

# Model coefficients and feature importance
print("\n" + "="*50)
print("MODEL COEFFICIENTS")
print("="*50)
feature_names = ['MedInc', 'HouseAge', 'AveRooms']
coefficients = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_,
    'Abs_Coefficient': np.abs(model.coef_)
})
coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)
print(coefficients)
print(f"\nIntercept: {model.intercept_:.4f}")

# Visualize predictions vs actual values
plt.figure(figsize=(15, 5))

# Subplot 1: Predicted vs Actual
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Subplot 2: Residuals plot
plt.subplot(1, 3, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6, color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals Plot')

# Subplot 3: Feature importance
plt.subplot(1, 3, 3)
bars = plt.bar(coefficients['Feature'], coefficients['Abs_Coefficient'], 
               color=['skyblue', 'lightcoral', 'lightgreen'])
plt.title('Feature Importance (Absolute Coefficients)')
plt.xlabel('Features')
plt.ylabel('Absolute Coefficient Value')
plt.xticks(rotation=45)

# Add value labels on bars
for bar, value in zip(bars, coefficients['Abs_Coefficient']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{value:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Model interpretation
print("\n" + "="*50)
print("MODEL INTERPRETATION")
print("="*50)
print("The linear regression equation is:")
equation = f"MedHouseVal = {model.intercept_:.4f}"
for feature, coef in zip(feature_names, model.coef_):
    equation += f" + ({coef:.4f} × {feature})"
print(equation)

print(f"\nModel Performance Summary:")
print(f"- The model explains {r2*100:.1f}% of the variance in house values")
print(f"- Average prediction error: ${mae*100000:.0f}")
print(f"- Root mean squared error: ${rmse*100000:.0f}")

# Feature interpretation
print(f"\nFeature Impact:")
for _, row in coefficients.iterrows():
    impact = "increases" if row['Coefficient'] > 0 else "decreases"
    print(f"- {row['Feature']}: A 1-unit increase {impact} house value by ${row['Coefficient']*100000:.0f}")
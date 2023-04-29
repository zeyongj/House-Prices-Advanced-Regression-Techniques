import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

# Load the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# EDA
plt.figure(figsize=(12, 6))
sns.histplot(train['SalePrice'], kde=True)
plt.title('Sale Price Distribution')
plt.show()

plt.figure(figsize=(12, 6))
sns.scatterplot(x=train['GrLivArea'], y=train['SalePrice'])
plt.title('Sale Price vs. GrLivArea')
plt.show()

correlation_matrix = train.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
plt.title('Correlation Matrix')
plt.show()

# Identify categorical and numerical columns
categorical_columns = train.select_dtypes(include=['object']).columns
numerical_columns = train.select_dtypes(exclude=['object']).drop(['Id', 'SalePrice'], axis=1).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_columns),
        ('cat', Pipeline(steps=[
            ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_columns)])

# Prepare the data
X_train = train.drop(['Id', 'SalePrice'], axis=1)
y_train = train['SalePrice']
X_test = test.drop(['Id'], axis=1)

# Preprocess the data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Feature scaling
scaler = MaxAbsScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define base models for stacking
base_models = [
    ('random_forest', RandomForestRegressor(n_jobs=-1, random_state=42)),
    ('xgb_regressor', XGBRegressor(n_jobs=-1, random_state=42)),
    ('lasso', Lasso(max_iter=10000, random_state=42))
]

# Set up stacking regressor
stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=Lasso(max_iter=10000, random_state=42), n_jobs=-1)

# Perform KFold cross-validation
kf = KFold(n_splits=5, random_state=42, shuffle=True)
rmse_scores = np.sqrt(-cross_val_score(stacking_regressor, X_train, y_train, scoring='neg_mean_squared_error', cv=kf))

# Print mean and standard deviation of RMSE scores
print(f"KFold Cross-Validation RMSE scores: {rmse_scores}")
print(f"Mean RMSE score: {rmse_scores.mean():.2f}")
print(f"Standard deviation of RMSE scores: {rmse_scores.std():.2f}")

# Train the stacking model
stacking_regressor.fit(X_train, y_train)

# Generate predictions
predictions = stacking_regressor.predict(X_test)

# Create the submission file
submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': predictions})
submission.to_csv('submission.csv', index=False)

print("Submission file created.")

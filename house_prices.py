import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Lasso
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

# Grid search for best alpha value
alphas = np.logspace(-5, 1, 100)
lasso = Lasso(max_iter=10000)
grid_search = GridSearchCV(estimator=lasso, param_grid=dict(alpha=alphas), scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

# Train a LASSO model with the best alpha value and increased max_iter
model = Lasso(alpha=grid_search.best_params_['alpha'], max_iter=10000)
model.fit(X_train, y_train)

# Perform KFold cross-validation
kf = KFold(n_splits=5, random_state=42, shuffle=True)
rmse_scores = np.sqrt(-cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf))

# Print mean and standard deviation of RMSE scores
print(f"KFold Cross-Validation RMSE scores: {rmse_scores}")
print(f"Mean RMSE score: {rmse_scores.mean():.2f}")
print(f"Standard deviation of RMSE scores: {rmse_scores.std():.2f}")

# Generate predictions
predictions = model.predict(X_test)

# Create the submission file
submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': predictions})
submission.to_csv('submission.csv', index=False)

print("Submission file created.")

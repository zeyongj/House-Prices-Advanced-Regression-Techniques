# House Prices: Advanced Regression Techniques
Kaggle Project: Predict sales prices and practice feature engineering, RFs, and gradient boosting

## Introduction

This project aims to predict house prices using the Ames Housing dataset. The goal is to preprocess the data, train a stacking model with multiple base models, and evaluate its performance using root mean squared error (RMSE) as the metric. The dataset contains both numerical and categorical features, which require different preprocessing techniques. A combination of RandomForestRegressor, XGBRegressor, and Lasso models are employed as base models, and a Lasso model is used as the final estimator.

## Dataset

The dataset used in this project is the Ames Housing dataset, which contains 79 explanatory variables describing various aspects of residential homes in Ames, Iowa.

## Libraries

The following libraries are used in the project:

- pandas
- numpy
- scikit-learn
- xgboost

## Model

A stacking model with RandomForestRegressor, XGBRegressor, and Lasso as base models and a Lasso model as the final estimator is used for this project. Grid search is employed to find the best alpha value for Lasso, and KFold cross-validation is used to evaluate the model's performance.

## Results

The model achieves a mean RMSE score of *0.13352* with a standard deviation of the RMSE scores across the KFold cross-validation. The code takes 5 minutes to execute on Google Colab.

## Execution

### Via Google Colab

The code in a Jupyter Notebook format can be run using Google Colab or Jupyter Notebook on your local machine. Make sure you have the required libraries installed and the dataset files available.

### Via Python Locally
1. Install the required libraries:
   - pandas
   - numpy
   - scikit-learn
   - xgboost
2. Load the datasets (`train.csv` and `test.csv`) in the same directory as the code.
3. Run the provided code to preprocess the data, train the stacking model, and make predictions on the test set.
4. The predictions will be saved in a `submission.csv` file in the correct format.


## Conclusions, Discussions, and Improvements

The stacking model achieved a relatively low RMSE score of *0.13352*, indicating decent performance in predicting house prices. The model's performance can be considered satisfactory given the complexity and variety of features in the dataset. However, there is still room for improvement.

Some potential improvements include:

1. **Feature engineering**: Investigating the dataset more thoroughly and creating new features or transforming existing ones to capture non-linear relationships between variables.

2. **Feature selection**: Applying feature selection techniques to reduce the number of features and retain the most important ones. This could potentially help the model generalize better and reduce overfitting.

3. **Model selection**: Trying different regression models such as Ridge regression, Elastic Net, or even tree-based models like Random Forest and Gradient Boosting to compare their performance with Lasso.

4. **Hyperparameter tuning**: Performing more extensive hyperparameter tuning for the chosen model to optimize its performance further.

By implementing these improvements, the model's performance in predicting house prices can potentially be enhanced, providing more accurate predictions and insights into the housing market.

## License

This project is licensed under the MIT License. The MIT License is a permissive open source license that allows for free use, copying, modification, and distribution of the software, as long as the copyright notice and permission notice are included in all copies or substantial portions of the software. This license is suitable for both academic and commercial projects.

## Reference

Anna Montoya, DataCanary. (2016). House Prices - Advanced Regression Techniques. Kaggle. Retrieved from https://kaggle.com/competitions/house-prices-advanced-regression-techniques .

## Author

Zeyong Jin

April 29th, 2023

#!/usr/bin/env python
# coding: utf-8


import pandas as pd 
import numpy as np 
import seaborn as sns 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle 
from sklearn import linear_model


# ### Load the train data 
# - Load the train data from the xlsx file from dataset folder
# - make timestamp as index


xlsx_path_train_data = "trainset.xlsx"
xlsx_path_test_data = "testset.xlsx"

# Read the CSV file into a pandas DataFrame
df_train = pd.read_excel(xlsx_path_train_data)
df_train["timestamp"] = pd.to_datetime(df_train["timestamp"], format="%Y-%m-%d %H:%M:%S")
df_train.set_index("timestamp", inplace=True)

# ### Make a  feature data set for the training data 
# - drop the unnecessary column 
# - drop the target column 

# Get the column index of the 114th column (since we want to keep the first 114 columns)
drop_cols = list(df_train.columns[114:])

# Drop the columns from the DataFrame
df_train = df_train.drop(drop_cols, axis=1)

df_train["grade"] = df_train["grade"].astype("category")


num_cols = df_train.select_dtypes(include=["int", "float"]).columns
cat_cols = df_train.select_dtypes(include=["category"]).columns

# Remove the target column "y_var" from num_cols
num_cols = num_cols.drop("y_var")

# Create a pipeline for numerical features
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("scaler", StandardScaler())
])

# Create a pipeline for categorical features
cat_transformer = OneHotEncoder()

# Combine the numerical and categorical transformers into a single ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ])
 
 

# # Fit the pipeline to Train dataset

# Fit the preprocessor to the data and transform it
X_preprocessed_train = preprocessor.fit_transform(df_train)

# Get the feature names from the preprocessor
cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)
num_feature_names = list(num_cols)
num_scaled_feature_names = [f"{name}_scaled" for name in num_cols]
feature_names = num_scaled_feature_names + list(cat_feature_names)
# Create a new DataFrame with the preprocessed features
X_preprocessed_train = pd.DataFrame(X_preprocessed_train, columns=feature_names)

# Set the timestamp as the index for the X_preprocessed_train DataFrame
X_preprocessed_train.index = df_train.index

# Add the "y_var" target column to the X_preprocessed_train DataFrame
X_preprocessed_train["y_var"] = df_train["y_var"]



# ### Test set Preprocessing for model fitting 

# Read the CSV file into a pandas DataFrame
df_test = pd.read_excel(xlsx_path_test_data)
df_test["timestamp"] = pd.to_datetime(df_test["timestamp"], format="%Y-%m-%d %H:%M:%S")
df_test.set_index("timestamp", inplace=True)

# Get the column index of the 114th column (since we want to keep the first 114 columns)
drop_cols = list(df_test.columns[114:])

# Drop the columns from the DataFrame
df_test = df_test.drop(drop_cols, axis=1)



df_test["grade"] = df_test["grade"].astype("category")


## Fitting the test data to the  Preprocessing pipeline

# Fit the preprocessor to the data and transform it
X_preprocessed_test = preprocessor.transform(df_test)

# Get the feature names from the preprocessor
cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)
num_feature_names = list(num_cols)
num_scaled_feature_names = [f"{name}_scaled" for name in num_cols]
feature_names = num_scaled_feature_names + list(cat_feature_names)
X_preprocessed_test = pd.DataFrame(X_preprocessed_test, columns=feature_names)
# Set the timestamp as the index for the X_preprocessed_train DataFrame
X_preprocessed_test.index = df_test.index


# Model Training 

# Define the name of the pickle file
filename = 'model.pkl'

# Try to load the model from the pickle file
try:
    with open(filename, 'rb') as file:
        model = pickle.load(file)
except:
    # If the file doesn't exist or there's an error loading it, define a new model
    model = linear_model.Ridge(alpha=2)

# Iterate over the indices of the test DataFrame
predictions = []
for i in range(len(df_test.index)):
    # Get the index of the current test data point
    test_index = df_test.index[i]

    # Get the subset of the train DataFrame that occurs between the previous test index and the current test index
    if i == 0:
        train_subset = X_preprocessed_train.loc[:test_index]
    else:
        prev_test_index = df_test.index[i-1]
        train_subset = X_preprocessed_train.loc[prev_test_index:test_index]

    # Check if the train_subset is not empty
    if not train_subset.empty:
        # Separate the features and target variable
        X_train = train_subset.drop('y_var', axis=1)
        y_train = train_subset['y_var']

        # Update the model with the new training data
        model.fit(X_train, y_train)

        # Save the updated model as a pickle file
        with open(filename, 'wb') as file:
            pickle.dump(model, file)

    # Use the updated model to make predictions on the current test data point
    X_test = X_preprocessed_test.iloc[[i]]
    y_pred = model.predict(X_test)

    # Add the predicted value to the list of predictions
    predictions.append(y_pred[0])



# Applying Domain Expertise
## Removing negative numbers and converting to nearest multiple of 5
num_list = predictions
for i in range(len(num_list)):
    if num_list[i] < 0:
        num_list[i] = 0
    else:
        num_list[i] = (num_list[i] + 2) // 5 * 5
predictions = num_list



# Developing Prediction Dataframe
test_timestamp = pd.read_excel(xlsx_path_test_data)['timestamp'].to_list()
testset_predictions = pd.DataFrame({'timestamp':test_timestamp,'y_var':predictions})


# Exporting into Excel file
testset_predictions.to_excel("pred_test_set.xlsx",index=False)

# Result
print("-----------------------\nDone\n Please check the file 'pred_test_set.xlsx' for predicted values.\n")
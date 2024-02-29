#module imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from evaluate import plot_residuals, regression_errors, baseline_mean_errors, better_than_baseline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from math import sqrt

#custom imports
import wrangle as w
import evaluate as ev


def eval_model(y_actual, y_hat):
    """
    Calculate the root mean squared error (RMSE) between the actual and predicted values.

    Args:
        y_actual (array-like): Actual target values.
        y_hat (array-like): Predicted target values.

    Returns:
        float: The RMSE between y_actual and y_hat.
    """
    
    return sqrt(mean_squared_error(y_actual, y_hat))




def scale_data(train, val, test):
    """
    Scales the numerical columns of the data using the specified scaler.

    Args:
        train (pd.DataFrame): Training data.
        val (pd.DataFrame): Validation data.
        test (pd.DataFrame): Test data.
        scaler (sklearn.preprocessing.Scaler): The scaler to use for data scaling.

    Returns:
        tuple: A tuple containing scaled versions of train, validation, and test DataFrames.
    """

    # Make copies for scaling
    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()

    columns_to_scale = ['fixed_acidity',
                         'volatile_acidity',
                         'citric_acid',
                         'residual_sugar',
                         'chlorides',
                         'free_sulfur_dioxide',
                         'total_sulfur_dioxide',
                         'density',
                         'ph',
                         'sulphates',
                         'alcohol',
                         'total_sulfur_ratio',
                         'acidity_level',
                         'sugar_alcohol_ratio',
                         'chlorides_ph_ratio',
                         'density_ph_ratio',
                         'sulfur_dioxide_level',
                         'sulfates_chlorides_ratio',
                         'total_acid',
                         'sulfur_dioxide_chlorides_ratio',
                         'residual_sugar_ph_ratio',
                         'acid_ratio',
                         'alcohol_ph_ratio',
                         'chlorides_density_ratio',
                         'total_sulfur_residual_sugar_ratio',
                         'sulfur_dioxide_percentage',
                         'ph_chlorides_ratio',
                         'alcohol_sugar_ratio',
                         'density_sulfates_ratio',
                         'chlorides_sulfates_ratio',
                         'residual_sugar_percentage',
                         'alcohol_chlorides_ratio',
                         'density_sulfur_dioxide_ratio',
                         'ph_sulfur_dioxide_ratio',
                         'sulfur_dioxide_sugar_ratio',
                       ]

    scaler = MinMaxScaler()  # Scaler object is created within the function
    
    # Fit the scaler on the training data for all of the columns
    scaler.fit(train[columns_to_scale])
    
    # Transform the data for each split
    train_scaled[columns_to_scale] = scaler.transform(train[columns_to_scale])
    val_scaled[columns_to_scale] = scaler.transform(val[columns_to_scale])
    test_scaled[columns_to_scale] = scaler.transform(test[columns_to_scale])

    scaled_col = [train_scaled, val_scaled, test_scaled]
    
    return train_scaled, val_scaled, test_scaled


def model_features(train_scaled, val_scaled, test_scaled, variable1, variable2, variable3, n_clusters=5):
    '''
    Inputs:
    train_scaled - the training dataset
    val_scaled - the validation dataset
    test_scaled - the test dataset
    variable1, variable2, variable3 - feature names as strings in search of potential clusters
    n_clusters - the number of clusters for KMeans (default is 5)
    
    Outputs:
    Modified train, validation, and test dataframes with additional 'cluster' and 'composite_cluster' columns.
    '''
    
    # create a subset of train with the specified variables
    X_train = train_scaled[[variable1, variable2, variable3]]
    X_val = val_scaled[[variable1, variable2, variable3]]
    X_test = test_scaled[[variable1, variable2, variable3]]
    
    # initiate and fit kmeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42) 
    kmeans.fit(X_train)
    
    train_scaled['cluster'] = kmeans.predict(X_train)
    val_scaled['cluster'] = kmeans.predict(X_val)
    test_scaled['cluster'] = kmeans.predict(X_test)
    
    # Create centroids dataframe for potential use or display
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=[variable1, variable2, variable3])
    
    # Rename cluster labels for clarity
    train_scaled['cluster'] = 'cluster_' + train_scaled.cluster.astype(str)
    val_scaled['cluster'] = 'cluster_' + val_scaled.cluster.astype(str)
    test_scaled['cluster'] = 'cluster_' + test_scaled.cluster.astype(str)
    
    # Create dummy columns for the 'cluster' column
    cluster_dummies_train = pd.get_dummies(train_scaled['cluster'], prefix='cluster', drop_first=False)
    cluster_dummies_val = pd.get_dummies(val_scaled['cluster'], prefix='cluster', drop_first=False)
    cluster_dummies_test = pd.get_dummies(test_scaled['cluster'], prefix='cluster', drop_first=False)
    
    # Append these dummy columns to the dataframes
    train_scaled = pd.concat([train_scaled, cluster_dummies_train], axis=1)
    val_scaled = pd.concat([val_scaled, cluster_dummies_val], axis=1)
    test_scaled = pd.concat([test_scaled, cluster_dummies_test], axis=1)

    # Drop the original 'cluster' column
    train_scaled = train_scaled.drop(columns='cluster')
    val_scaled = val_scaled.drop(columns='cluster')
    test_scaled = test_scaled.drop(columns='cluster')
    
    # Rename the cluster columns
    for i in range(n_clusters):
        train_scaled = train_scaled.rename(columns={'cluster_cluster_' + str(i): 'composite_cluster_' + str(i)})
        val_scaled = val_scaled.rename(columns={'cluster_cluster_' + str(i): 'composite_cluster_' + str(i)})
        test_scaled = test_scaled.rename(columns={'cluster_cluster_' + str(i): 'composite_cluster_' + str(i)})

    return train_scaled, val_scaled, test_scaled


def one_hot_encode_and_drop(df, columns_to_encode):
    """
    Apply one-hot encoding to specified columns and drop the original columns from the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to apply one-hot encoding to.
        columns_to_encode (list): List of column names to one-hot encode.

    Returns:
        pd.DataFrame: The DataFrame with one-hot encoding applied and original columns dropped.
    """
    # Iterate through the columns and apply one-hot encoding
    for column in columns_to_encode:
        df = pd.concat([df, pd.get_dummies(df[column], prefix=column, drop_first=True)], axis=1)

    # Drop the original columns from the dataframe
    df.drop(columns=columns_to_encode, inplace=True)
    
    return df


# ------------------------ XY SPLIT FUNCTION ----------------------
# xy_split function to create usable subsets; reusable.
def xy_split(df, col):
    """
    Split the DataFrame into feature and target data.

    Args:
        df (pd.DataFrame): The DataFrame containing both features and the target.
        col (str): The name of the target column.

    Returns:
        tuple: A tuple containing feature data (X) and target data (y).
    """
    X = df.drop(columns=[col])
    y = df[col]
    return X, y


def baseline_model(train, target, train_scaled, val_scaled, test_scaled, variable1, variable2, variable3):
    """
    Calculate and display baseline model metrics.

    Args:
        train (pd.DataFrame): Training data.
        target (str): Name of the target column.
        train_scaled (pd.DataFrame): Scaled training data.
        val_scaled (pd.DataFrame): Scaled validation data.
        test_scaled (pd.DataFrame): Scaled test data.
        variable1 (str): Name of the first feature for clustering.
        variable2 (str): Name of the second feature for clustering.
        variable3 (str): Name of the third feature for clustering.
    """
    # Call find_clusters function to add cluster columns to train, val, and test
    train, val, test = model_features(train_scaled, val_scaled, test_scaled, variable1, variable2, variable3, n_clusters=5)

    # drop categorical features
    train = train.drop(columns=(['quality_bins', 'alcohol_bins', 'ph_bins']))
    val = val.drop(columns=(['quality_bins', 'alcohol_bins', 'ph_bins']))

    # Split data into X and y for train and val
    X_train, y_train = xy_split(train, target)  # Pass target as a list
    X_val, y_val = xy_split(val, target)  # Pass target as a list

    # Calculate baseline
    bl = y_train.median()

    # Create a DataFrame to work with
    preds = pd.DataFrame({'y_actual' : y_train,
                          'y_baseline': bl})

    # Calculate baseline residuals
    preds['y_baseline_residuals'] = bl - preds['y_actual']

    # Initialize and fit a linear regression model
    lm = LinearRegression()
    lm.fit(X_train, y_train)

    # Make predictions with the model
    preds['y_hat'] = lm.predict(X_train)

    # Calculate model residuals
    preds['y_hat_residuals'] = preds['y_hat'] - preds['y_actual']

    # Plot residuals
    plot_residuals(preds.y_actual, preds.y_hat)

    print(f"\n-------------------------------------")
    # Calculate regression errors
    SSE, ESS, TSS, MSE, RMSE = regression_errors(preds.y_actual, preds.y_hat)
    print(f"\nModel RMSE: {RMSE:.2f}\n")
    print(f"\n-------------------------------------")

    # Calculate baseline errors
    SSE_baseline, MSE_baseline, RMSE_baseline = baseline_mean_errors(preds.y_actual)
    print(f"\nBaseline RMSE: {RMSE_baseline:.2f}\n")
    print(f"\n-------------------------------------")


# _______________________Select kBest function _____________________

def select_k_features(data, target, k=5):
    """
    Select the top k features using SelectKBest and return their names.

    Args:
        data (pd.DataFrame): The DataFrame containing features and target.
        target (str): Name of the target column.
        k (int): Number of top features to select.

    Returns:
        list: List of selected feature names.
    """
    # # drop categorical features
    # data = data.drop(columns=(['quality_bins_Medium',
    #                             'quality_bins_High',
    #                             'alcohol_bins_Medium',
    #                             'alcohol_bins_High',
    #                             'ph_bins_Medium',
    #                             'ph_bins_High']))

    # Split scale data
    X_train, y_train = xy_split(data, target)

    # Create a SelectKBest instance with k specified
    k_best = SelectKBest(score_func=f_regression, k=k)

    # Fit and transform your training data
    X_train_selected = k_best.fit_transform(X_train, y_train)

    # Get the indices of the selected features
    selected_indices = k_best.get_support(indices=True)

    # Get the names of the selected features
    selected_feature_names = X_train.columns[selected_indices]

    return selected_feature_names


#=========================feature prep===============================

def modeling_features(train, val, test):
    """
    Prepare features for modeling by adding cluster features and applying one-hot encoding.

    Args:
        train (pd.DataFrame): Training data.
        val (pd.DataFrame): Validation data.
        test (pd.DataFrame): Test data.

    Returns:
        tuple: A tuple containing modified train, validation, and test DataFrames.
    """
    # # drop categorical features
    # train = train.drop(columns=(['quality_bins', 'alcohol_bins', 'ph_bins']))
    # val = val.drop(columns=(['quality_bins', 'alcohol_bins', 'ph_bins']))
    # test = test.drop(columns=(['quality_bins', 'alcohol_bins', 'ph_bins']))

    # Define the columns you want to one-hot encode
    columns_to_encode = ['quality_bins', 'alcohol_bins', 'ph_bins']
    
    # Add cluster features using the 'model_features' function or your preferred method
    train, val, test = model_features(train, val, test, 'fixed_acidity', 'residual_sugar', 'density')

    # Apply one-hot encoding to the specified columns
    for column in columns_to_encode:
        train = pd.concat([train, pd.get_dummies(train[column], prefix=column, drop_first=False)], axis=1)
        val = pd.concat([val, pd.get_dummies(val[column], prefix=column, drop_first=False)], axis=1)
        test = pd.concat([test, pd.get_dummies(test[column], prefix=column, drop_first=False)], axis=1)

    # Drop the original columns from the dataframes
    train.drop(columns=columns_to_encode, inplace=True)
    val.drop(columns=columns_to_encode, inplace=True)
    test.drop(columns=columns_to_encode, inplace=True)

    # Scale data for modeling using your 'scale_data' function or your preferred method
    train, val, test = scale_data(train, val, test)

    return train, val, test


# ========================model functions=========================

def model_1(X_train, y_train, X_val, y_val):
    """
    Train and evaluate a Random Forest model with hyperparameter tuning.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
    """
    # Define the hyperparameter grid to search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    # Create the grid search object
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

    # Train the model with hyperparameter tuning on the training data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Initialize the RandomForestRegressor with the best hyperparameters
    rfr = RandomForestRegressor(**best_params, random_state=42)

    # Train the model on the training data
    rfr.fit(X_train, y_train)

    # Make predictions on training and validation sets
    train_preds = rfr.predict(X_train)
    val_preds = rfr.predict(X_val)

    # Calculate RMSE for training and validation sets
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

    # Calculate R-squared (R2) for training and validation sets
    train_r2 = r2_score(y_train, train_preds)
    val_r2 = r2_score(y_val, val_preds)

    # Print the metrics and best hyperparameters
    print(f"\n-------------------------------------")
    print(f"\nTraining RMSE: {train_rmse:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nValidation RMSE: {val_rmse:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nTraining R-squared (R2): {train_r2:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nValidation R-squared (R2): {val_r2:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nBest Hyperparameters: {best_params}")






def model_2(X_train, y_train, X_val, y_val, early_stopping_rounds=10, params=None):
    """
    Train and evaluate an XGBoost model with hyperparameter tuning and early stopping.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
        early_stopping_rounds (int): Number of rounds with no improvement to trigger early stopping.
        params (dict): XGBoost hyperparameters.
    """
    # Define the hyperparameters for your XGBoost model (or pass them as an argument)
    if params is None:
        params = {
            'learning_rate': 0.1,      # Typical values are between 0.01 and 0.3
            'n_estimators': 100,       # Start with a moderate number of trees
            'max_depth': 4,            # Start shallow to prevent overfitting
            'min_child_weight': 5,     # Typically between 1 and 10
            'gamma': 0,                # Start with no regularization
            'subsample': 0.9,          # Fraction of samples used for training
            'colsample_bytree': 0.9,   # Fraction of features used for training each tree
            'objective': 'reg:squarederror',  # For regression tasks
            'random_state': 42,        # Set a random seed for reproducibility
            'eval_metric': 'rmse',     # Use RMSE as the evaluation metric
            'early_stopping_rounds': 10,  # To prevent overfitting, stop if the validation score doesn't improve for 10 rounds
            # Add other hyperparameters as needed
        }

    # Define weight data (you can replace this with your actual weights)
    sample_weights = np.ones(X_train.shape[0])  # Example: All weights are set to 1
    
    # Create the XGBoost regressor with your specified hyperparameters
    xgb = XGBRegressor(**params)
    
    # Fit the model to your training data with eval_set and verbose
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, sample_weight=sample_weights)

    # Access the best iteration and best score
    best_iteration = xgb.best_iteration
    best_score = xgb.best_score
    
    # Make predictions on validation set
    val_preds = xgb.predict(X_val)
    
    # Calculate RMSE and R2 for the validation set
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    val_r2 = r2_score(y_val, val_preds)

    # Make predictions on training set
    train_preds = xgb.predict(X_train)

    # Calculate RMSE and R2 for the training set
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    train_r2 = r2_score(y_train, train_preds)
    
    # Create a dictionary to store the results
    results = {
        'model': xgb,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'val_rmse': val_rmse,
        'val_r2': val_r2,
        'best_score': best_score,
        'best_iteration': best_iteration
    }

    # Print the metrics within the function
    print(f"\n-------------------------------------")
    print(f"\nTrain RMSE: {train_rmse:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nTrain R-squared (R2): {train_r2:.2f}")
    print(f"\n------------------------------------")
    print(f"\nValidation RMSE: {val_rmse:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nValidation R-squared (R2): {val_r2:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nBest Score: {best_score:.2f}")
    


def train_and_evaluate_model(model, X_train, y_train, X_val, y_val):
    """
    Train and evaluate an XGBoost model with hyperparameter tuning and early stopping.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
        early_stopping_rounds (int): Number of rounds with no improvement to trigger early stopping.
        params (dict): XGBoost hyperparameters.
    """
    # Train the model on the training data
    model.fit(X_train, y_train)
    
    # Predict on training and validation data
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    # Evaluate the model's performance
    train_rmse = eval_model(y_train, train_preds)
    val_rmse = eval_model(y_val, val_preds)

    # Calculate R-squared (R2) for training and validation sets
    train_r2 = r2_score(y_train, train_preds)
    val_r2 = r2_score(y_val, val_preds)
    
    # Print the results
    print(f"\n-------------------------------------")
    print(f'The train RMSE is {train_rmse:.2f}.\n')
    print(f"\n-------------------------------------")
    print(f'The validation RMSE is {val_rmse:.2f}.\n\n')
    print(f"\n-------------------------------------")
    print(f"\nTraining R-squared (R2): {train_r2:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nValidation R-squared (R2): {val_r2:.2f}")
    
    return model, train_rmse, val_rmse




def model_3(X_train, y_train, X_val, y_val):
    """
    Train and evaluate a polynomial regression model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
    """ 
        
        # Calculate mean and median of y_train
        y_train_mean = y_train.mean()
        y_train_median = y_train.median()
        
        # Create a DataFrame with y_train statistics
        bl = pd.DataFrame({"y_actual" : y_train,
                           "y_mean" : y_train_mean,
                           "y_median" : y_train_median})
        
        # Apply polynomial feature transformation
        poly = PolynomialFeatures()
        X_train = poly.fit_transform(X_train)
        X_val = poly.transform(X_val)

        
    
        # Train a Linear Regression model and evaluate it
        lm = LinearRegression()
        trained_model, train_rmse, val_rmse = train_and_evaluate_model(lm, X_train, y_train, X_val, y_val)







def best_model(X_train, y_train, X_test, y_test, early_stopping_rounds=10, params=None):
    """
    Train and evaluate the best model (XGBoost) with hyperparameter tuning, early stopping, and test data.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        early_stopping_rounds (int): Number of rounds with no improvement to trigger early stopping.
        params (dict): XGBoost hyperparameters.
    """
   # Define the hyperparameters for your XGBoost model (or pass them as an argument)
    if params is None:
        params = {
            'learning_rate': 0.1,
            'n_estimators': 100,
            'max_depth': 4,
            'min_child_weight': 5,
            'gamma': 0,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'objective': 'reg:squarederror',
            'random_state': 42,
            'eval_metric': 'rmse',
            'early_stopping_rounds': 10,
            # Add other hyperparameters as needed
        }

    # Define weight data (you can replace this with your actual weights)
    sample_weights = np.ones(X_train.shape[0])  # Example: All weights are set to 1
    
    # Create the XGBoost regressor with your specified hyperparameters
    xgb = XGBRegressor(**params)
    
    # Fit the model to your training data with eval_set and verbose
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False, sample_weight=sample_weights)

    # Access the best iteration and best score
    best_iteration = xgb.best_iteration
    best_score = xgb.best_score
    
    # Make predictions on the test set
    test_preds = xgb.predict(X_test)
    
    # Calculate RMSE and R2 for the test set
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_r2 = r2_score(y_test, test_preds)

    # Make predictions on the training set
    train_preds = xgb.predict(X_train)

    # Calculate RMSE and R2 for the training set
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    train_r2 = r2_score(y_train, train_preds)
    
    # Create a dictionary to store the results
    results = {
        'model': xgb,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'best_score': best_score,
        'best_iteration': best_iteration
    }

    # Print the metrics within the function
    print(f"\n-------------------------------------")
    print(f"\nTrain RMSE: {train_rmse:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nTrain R-squared (R2): {train_r2:.2f}")
    print(f"\n------------------------------------")
    print(f"\nTest RMSE: {test_rmse:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nTest R-squared (R2): {test_r2:.2f}")
    print(f"\n-------------------------------------")
    print(f"\nBest Score: {best_score:.2f}")

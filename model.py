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
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from math import sqrt

#custom imports
import wrangle as w
import evaluate as ev




def eval_model(y_actual, y_hat):
    
    return sqrt(mean_squared_error(y_actual, y_hat))




def scale_data(train, val, test, scaler):
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
    
    # Fit the scaler on the training data for all of the columns
    scaler.fit(train[columns_to_scale])
    
    # Transform the data for each split
    train_scaled[columns_to_scale] = scaler.transform(train[columns_to_scale])
    val_scaled[columns_to_scale] = scaler.transform(val[columns_to_scale])
    test_scaled[columns_to_scale] = scaler.transform(test[columns_to_scale])

    scaled_col = [train_scaled, val_scaled, test_scaled]
    
    return train_scaled, val_scaled, test_scaled


def find_clusters(train_scaled, val_scaled, test_scaled, variable1, variable2, variable3, n_clusters=5):
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
    cluster_dummies_train = pd.get_dummies(train_scaled['cluster'], prefix='cluster', drop_first=True)
    cluster_dummies_val = pd.get_dummies(val_scaled['cluster'], prefix='cluster', drop_first=True)
    cluster_dummies_test = pd.get_dummies(test_scaled['cluster'], prefix='cluster', drop_first=True)
    
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


# ------------------------ XY SPLIT FUNCTION ----------------------
# xy_split function to create usable subsets; reusable.
def xy_split(df, col):
    X = df.drop(columns=[col])
    y = df[col]
    return X, y


def baseline_model(train, target, train_scaled, val_scaled, test_scaled, variable1, variable2, variable3):
    
    # Call find_clusters function to add cluster columns to train, val, and test
    train, val, test = find_clusters(train_scaled, val_scaled, test_scaled, variable1, variable2, variable3, n_clusters=5)

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

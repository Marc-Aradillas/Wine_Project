import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Constant scaler for data scaling
mms = MinMaxScaler()

def acquire_wine():
    """
    Acquires the wine dataset from a CSV file.

    Returns:
        pd.DataFrame or None: A DataFrame containing the wine data if the file exists,
                             or None if the file is not found.
    """
    filename = 'winequality_red_white.csv'
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # Handle the case where the file doesn't exist
        return None

def clean_wine():
    """
    Cleans the wine dataset by converting column names to lowercase and replacing spaces with underscores.

    Returns:
        pd.DataFrame: A cleaned DataFrame with standardized column names and additional bins columns.
    """

    # Call acquire_wine to get the DataFrame
    df = acquire_wine()

    if df is not None:
        # Clean the column names
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]

        # Create 'alcohol_bins' and 'quality_bins' columns
        df['alcohol_bins'] = pd.cut(df['alcohol'], [0, 8, 10, 12, 15], labels=['no_alcohol', 'low_alcohol', 'medium_alcohol', 'high_alcohol'])
        df['quality_bins'] = pd.cut(df['quality'], [0, 3, 7, 10], labels=['low', 'medium', 'high'])

    return df


def train_val_test(df, target=None, seed = 42):
    """
    Splits the dataset into training, validation, and test sets.

    Args:
        df (pd.DataFrame): The DataFrame to be split.
        target (pd.Series or None): The target variable (if applicable).
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing train, validation, and test DataFrames.
    """
    train, val_test = train_test_split(df, train_size=0.7, random_state=seed, stratify=target)
    val, test = train_test_split(val_test, train_size=0.5, random_state=seed, stratify=target)
    return train, val, test

# def scale_data(train, val, test, scaler):
#     """
#     Scales the numerical columns of the data using the specified scaler.

#     Args:
#         train (pd.DataFrame): Training data.
#         val (pd.DataFrame): Validation data.
#         test (pd.DataFrame): Test data.
#         scaler (sklearn.preprocessing.Scaler): The scaler to use for data scaling.

#     Returns:
#         tuple: A tuple containing scaled versions of train, validation, and test DataFrames.
#     """
#     # Make copies for scaling
#     train_scaled = train.copy()
#     val_scaled = val.copy()
#     test_scaled = test.copy()

#     columns_to_scale = ['fixed_acidity',
#                         'volatile_acidity',
#                         'citric_acid',
#                         'residual_sugar',
#                         'chlorides',
#                         'free_sulfur_dioxide',
#                         'total_sulfur_dioxide',
#                         'density',
#                         'ph',
#                         'sulphates',
#                         'alcohol',
#                         'quality',
#                        ]
    
#     # Fit the scaler on the training data for all of the columns
#     scaler.fit(train[columns_to_scale])
    
#     # Transform the data for each split
#     train_scaled[columns_to_scale] = scaler.transform(train[columns_to_scale])
#     val_scaled[columns_to_scale] = scaler.transform(val[columns_to_scale])
#     test_scaled[columns_to_scale] = scaler.transform(test[columns_to_scale])

#     scaled_col = [train_scaled, val_scaled, test_scaled]
    
#     return train_scaled, val_scaled, test_scaled

def wrangle_wine():
    """
    Orchestrates the data acquisition, cleaning, splitting, and scaling process.

    Returns:
        tuple or None: A tuple containing train, validation, and test DataFrames if data is acquired successfully,
                      or None if data acquisition fails.
    """
    # Acquire the data
    df = acquire_wine()

    if df is not None:
        # Clean the data
        df = clean_wine()

        # Split the data
        train, val, test = train_val_test(df)

        # Scale the data
        # train_scaled, val_scaled, test_scaled = scale_data(train, val, test, mms)

        return df
    else:
        # Handle the case where data acquisition failed
        return None

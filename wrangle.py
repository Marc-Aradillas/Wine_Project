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

        # Create features
        
        # Quality Range
        df['quality_bins'] = pd.cut(df['quality'], [0, 3, 6, 10], labels=['Low', 'Medium', 'High'])

        # Alcohol Content Range
        df['alcohol_bins'] = pd.cut(df['alcohol'], [0, 10, 12, float('inf')], labels=['Low', 'Medium', 'High'])
        
        # Total Sulfur Dioxide per Free Sulfur Dioxide
        df['total_sulfur_ratio'] = df['total_sulfur_dioxide'] / df['free_sulfur_dioxide']
        
        # Acidity Level
        df['acidity_level'] = df['citric_acid'] * df['ph']
        
        # Sugar to Alcohol Ratio
        df['sugar_alcohol_ratio'] = df['residual_sugar'] / df['alcohol']
        
        # Chlorides to pH Ratio
        df['chlorides_ph_ratio'] = df['chlorides'] / df['ph']
        
        # Density to pH Ratio
        df['density_ph_ratio'] = df['density'] / df['ph']
        
        # Sulfur Dioxide Level
        df['sulfur_dioxide_level'] = df['free_sulfur_dioxide'] + df['total_sulfur_dioxide']
        
        # Sulfates to Chlorides Ratio
        df['sulfates_chlorides_ratio'] = df['sulphates'] / df['chlorides']
        
        # pH Level Range
        df['ph_bins'] = pd.cut(df['ph'], [0, 3.0, 3.5, float('inf')], labels=['Low', 'Medium', 'High'])
        
        
        # Total Acid Level
        df['total_acid'] = df['citric_acid'] + df['sulphates']
        
        # Sulfur Dioxide to Chlorides Ratio
        df['sulfur_dioxide_chlorides_ratio'] = df['total_sulfur_dioxide'] / df['chlorides']
        
        # Residual Sugar to pH Ratio
        df['residual_sugar_ph_ratio'] = df['residual_sugar'] / df['ph']
        
        # Acid Ratio
        df['acid_ratio'] = df['citric_acid'] / df['sulphates']
        
        # Alcohol to pH Ratio
        df['alcohol_ph_ratio'] = df['alcohol'] / df['ph']
        
        # Chlorides to Density Ratio
        df['chlorides_density_ratio'] = df['chlorides'] / df['density']
        
        # Total Sulfur Dioxide to Residual Sugar Ratio
        df['total_sulfur_residual_sugar_ratio'] = df['total_sulfur_dioxide'] / df['residual_sugar']
        
        # Sulfur Dioxide as a Percentage of Total Sulfur Dioxide
        df['sulfur_dioxide_percentage'] = (df['free_sulfur_dioxide'] / df['total_sulfur_dioxide']) * 100
        
        # pH to Chlorides Ratio
        df['ph_chlorides_ratio'] = df['ph'] / df['chlorides']
        
        # Alcohol to Sugar Ratio
        df['alcohol_sugar_ratio'] = df['alcohol'] / df['residual_sugar']
        
        # Density to Sulfates Ratio
        df['density_sulfates_ratio'] = df['density'] / df['sulphates']
        
        # Chlorides to Sulfates Ratio
        df['chlorides_sulfates_ratio'] = df['chlorides'] / df['sulphates']
        
        # Residual Sugar as a Percentage of Total Sulfur Dioxide
        df['residual_sugar_percentage'] = (df['residual_sugar'] / df['total_sulfur_dioxide']) * 100
        
        # Alcohol to Chlorides Ratio
        df['alcohol_chlorides_ratio'] = df['alcohol'] / df['chlorides']
        
        # Density to Sulfur Dioxide Ratio
        df['density_sulfur_dioxide_ratio'] = df['density'] / df['total_sulfur_dioxide']
        
        # pH to Sulfur Dioxide Ratio
        df['ph_sulfur_dioxide_ratio'] = df['ph'] / df['total_sulfur_dioxide']
        
        # Sulfur Dioxide to Sugar Ratio
        df['sulfur_dioxide_sugar_ratio'] = df['total_sulfur_dioxide'] / df['residual_sugar']



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

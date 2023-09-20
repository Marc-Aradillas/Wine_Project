import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PolynomialFeatur

def wrangle_wine():

    filename = 'winequality_red_white.csv'
    
    if os.path.isfile(filename):
        
        return pd.read_csv(filename)

        return df
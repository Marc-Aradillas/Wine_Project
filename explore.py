import warnings
warnings.filterwarnings("ignore")

# imported libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# constants
mms = MinMaxScaler()

def eval_p(p, a=0.05, decimal_places=2):
    """
    Evaluate the p-value and print the result of hypothesis testing.

    Args:
        p (float): The p-value to evaluate.
        a (float, optional): The significance level (default is 0.05).
        decimal_places (int, optional): The number of decimal places for formatting p-value (default is 2).

    Returns:
        None
    """
    formatted_p = "{:.{precision}e}".format(p, precision=decimal_places)
    if p < a:
        print(f'\nWe reject the null hypothesis with a p-value of {formatted_p}.')
    else:
        print(f'\nWe failed to reject the null hypothesis with a p-value of {formatted_p}.')

# ===========================================================================

def chi2_and_visualize(df, cat_var, target, a=0.05, decimal_places=2):
    """
    Perform chi-squared test and visualize the results.

    Args:
        df (DataFrame): The DataFrame containing the data.
        cat_var (str): The categorical variable to be tested.
        target (str): The target variable for the chi-squared test.
        a (float, optional): The significance level (default is 0.05).
        decimal_places (int, optional): The number of decimal places for formatting p-value (default is 2).

    Returns:
        None
    """
    observed = pd.crosstab(df[cat_var], df[target])
    chi2, p, degf, e = stats.chi2_contingency(observed)

    print('\n\n----------------------------------------------------------------------------------------------------\n')

    print(f'Chi2 Statistic: {chi2:.2f}\n')
    formatted_p = "{:.{precision}e}".format(p, precision=decimal_places)
    print(f'P-Value: {formatted_p}\n')

    sns.countplot(data=df, x=cat_var, hue=target)
    plt.title(f'Wine Quality vs. Alcohol Content Range')
    plt.xlabel(f'Alcohol Content')
    plt.ylabel(f'Wine Quality')
    plt.legend(title='Wine Quality', labels=['Poor', 'Average', 'Excellent'])
    plt.show()

    eval_p(p)

    print('\n\n\n')

def analysis_1(df, cat_var, target, a=0.05):
    """
    Perform chi-squared test and visualize the results for quality vs. alcohol.

    Args:
        df (DataFrame): The DataFrame containing the data.
        cat_var (str): The categorical variable (alcohol bins) to be tested.
        target (str): The target variable (wine quality) for the chi-squared test.
        a (float, optional): The significance level (default is 0.05).

    Returns:
        None
    """
    chi2_and_visualize(df, cat_var, target, a=0.05)

# ===========================================================================


<<<<<<< HEAD
    # Plot a histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(data)
    
    # Add a vertical line for the population mean
    plt.axvline(x=pop_mean, color='red', linestyle='--', label=f'Population Mean ({pop_mean:.1f})')
    
    plt.title(f'1-Sample t-test Analysis\nData: Alcohol percentage\n')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.legend()  # legend to label the vertical line
    plt.show()
    
    eval_p(p) 

def analysis_2(df, num_col_name, a=0.05):
    #FIXME: Reevaluate on exploratory analysis on the features
    num_col = df[num_col_name]
    pop_mean = df[num_col_name].mean()
    one_sample_t_test(num_col, pop_mean, a=0.05)
=======
def ttest_viz(df):
    '''
    run and plots ttest for categorical features against a continuous target
    '''
    cat, cont = cat_or_cont(df)
    for i in cont:
        print(f'Does the feature "{i}" have an effect on {cat[0]}?')
        print()
        print(f'H_0 {i} has no effect on {cat[0]}')
        print(f'H_a {i} has an effect on {cat[0]}')
        r_value, p_value = spearmanr(df[i], df[cat[0]])
        # box plot
        sns.barplot(data=df, y=i, x=cat[0])
        plt.title(f'{i.capitalize()} and {cat[0].capitalize()}')
        plt.grid(linestyle='-.', axis='y')  # Add grid line for better visualization
        plt.ylabel(i.capitalize())
        plt.xlabel(f'{cat[0].capitalize()}')
        plt.show()
        if p_value < 0.05:
            print(f'Reject the null hypothesis: {i} has an effect on {cat[0]}. (p-value: {p_value:.4e})')
            print(r_value)
            print('======================')
            print('======================')
            print()
            print()
        else:
            print(f"Fail to reject the null hypothesis: No significant evidence to suggest that {i} affects {cat[0]}. (p-value: {p_value:.4e})")
            print(r_value)
            print()
            print()
>>>>>>> f230d4699c53d9e5da5f0bdc4b77aaba3c212161


# ===========================================================================



def one_sample_t_test(data, pop_mean, a=0.05):
    """
    Perform a one-sample t-test and visualize the results.

    Args:
        data (array-like): The data to perform the t-test on.
        pop_mean (float): The population mean to compare against.
        a (float, optional): The significance level (default is 0.05).

    Returns:
        None
    """
    t, p = stats.ttest_1samp(data, pop_mean)

    print('\n\n----------------------------------------------------------------------------------------------------\n')

    print(f'P_Value: {p}\n')
    print(f'\nT-Statistic: {t}\n')

    plt.figure(figsize=(8, 6))
    sns.histplot(data)

    plt.axvline(x=pop_mean, color='red', linestyle='--', label=f'Population Mean ({pop_mean:.1f})')

    plt.title(f'1-Sample t-test Analysis\nData: Alcohol percentage\n')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    eval_p(p)

def analysis_2(df, num_col_name, a=0.05):
    """
    Perform a one-sample t-test and visualize the results for a numeric variable.

    Args:
        df (DataFrame): The DataFrame containing the data.
        num_col_name (str): The name of the numeric variable to analyze.
        a (float, optional): The significance level (default is 0.05).

    Returns:
        None
    """
    num_col = df[num_col_name]
    pop_mean = df[num_col_name].mean()
    one_sample_t_test(num_col, pop_mean, a=0.05)

# ===========================================================================

def pearson_test(data, x_col, y_col, a=0.05, decimal_places=2):
    """
    Perform a Pearson correlation test and visualize the results.

    Args:
        data (DataFrame): The DataFrame containing the data.
        x_col (str): The name of the first column for correlation.
        y_col (str): The name of the second column for correlation.
        a (float, optional): The significance level (default is 0.05).
        decimal_places (int, optional): The number of decimal places for formatting p-value (default is 2).

    Returns:
        None
    """
    if x_col not in data.columns or y_col not in data.columns:
        return "The required columns are not found in the DataFrame."

    print('\n----------------------------------------------------------------------------------------------------\n')
    
    r, p = stats.pearsonr(data[x_col], data[y_col])

    print(f'r_value: {r:.4f}\n')
    formatted_p = "{:.{precision}e}".format(p, precision=decimal_places)
    print(f'p_value: {formatted_p}\n')

    sns.scatterplot(x=x_col, y=y_col, data=data, hue='quality_bins')
    plt.title('Scatter Plot of Citric Acid vs Fixed Acidity')
    plt.show()

    eval_p(p)

def analysis_3(data, x_col, y_col):
    """
    Perform a Pearson correlation test and visualize the results for two numeric variables.

    Args:
        data (DataFrame): The DataFrame containing the data.
        x_col (str): The name of the first column for correlation.
        y_col (str): The name of the second column for correlation.

    Returns:
        None
    """
    pearson_test(data, x_col, y_col, a=0.05, decimal_places=2)

# ===========================================================================

def find_and_plot_clusters(train_scaled, val_scaled, test_scaled, variable1, variable2, variable3, n_clusters=5):
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
    
    X_train = train_scaled[[variable1, variable2, variable3]]
    X_val = val_scaled[[variable1, variable2, variable3]]
    X_test = test_scaled[[variable1, variable2, variable3]]
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42) 
    kmeans.fit(X_train)
    
    train_scaled['cluster'] = kmeans.predict(X_train)
    val_scaled['cluster'] = kmeans.predict(X_val)
    test_scaled['cluster'] = kmeans.predict(X_test)
    
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=[variable1, variable2, variable3])
    
    train_scaled['cluster'] = 'cluster_' + train_scaled.cluster.astype(str)
    val_scaled['cluster'] = 'cluster_' + val_scaled.cluster.astype(str)
    test_scaled['cluster'] = 'cluster_' + test_scaled.cluster.astype(str)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    sns.scatterplot(ax=axes[0], x=variable1, y=variable2, hue='cluster', palette='colorblind', data=train_scaled)
    axes[0].set_title("Actual Distribution of Quality Bins - Train")
    
    sns.scatterplot(ax=axes[1], x=variable1, y=variable2, hue='cluster', palette='colorblind', data=val_scaled)
    axes[1].set_title("Clusters Generated by KMeans - Validation")
    
    plt.show()

    cluster_dummies_train = pd.get_dummies(train_scaled['cluster'], prefix='cluster', drop_first=True)
    cluster_dummies_val = pd.get_dummies(val_scaled['cluster'], prefix='cluster', drop_first=True)
    cluster_dummies_test = pd.get_dummies(test_scaled['cluster'], prefix='cluster', drop_first=True)
    
    train_scaled = pd.concat([train_scaled, cluster_dummies_train], axis=1)
    val_scaled = pd.concat([val_scaled, cluster_dummies_val], axis=1)
    test_scaled = pd.concat([test_scaled, cluster_dummies_test], axis=1)

    train_scaled = train_scaled.drop(columns='cluster')
    val_scaled = val_scaled.drop(columns='cluster')
    test_scaled = test_scaled.drop(columns='cluster')
    
    for i in range(n_clusters):
        train_scaled = train_scaled.rename(columns={'cluster_cluster_' + str(i): 'composite_cluster_' + str(i)})
        val_scaled = val_scaled.rename(columns={'cluster_cluster_' + str(i): 'composite_cluster_' + str(i)})
        test_scaled = test_scaled.rename(columns={'cluster_cluster_' + str(i): 'composite_cluster_' + str(i)})

    return train_scaled, val_scaled, test_scaled

def analysis_4(train_scaled, val_scaled, test_scaled, variable1, variable2, variable3):
    
    train_scaled, val_scaled, test_scaled = find_and_plot_clusters(train_scaled, val_scaled, test_scaled, variable1, variable2, variable3, n_clusters=5)

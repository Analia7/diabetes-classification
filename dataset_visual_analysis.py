import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# Function to plot the distribution of each metric
def plot_distributions(df):
    # Plot distribution of all metrics
    plt.figure(figsize=(12, 10))
    
    # Loop through each column to create subplots
    for i, column in enumerate(df.columns, 1):
        plt.subplot(3, 2, i)
        sns.histplot(df[column], kde=True, bins=10, color='blue')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

# Function to clean the DataFrame and export
def clean_and_export_csv(file_path):
    # read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # remove rows where 'avg_max_depth' is 0, as this implies that the hyperparameters used were too constraining
    df_cleaned = df[df['avg_max_depth'] != 0]
    
    # create a new filename with '_edit' added before the file extension
    file_path_edit = file_path.replace('.csv', '_edit.csv')
    
    # export the cleaned DataFrame to a new CSV file
    df_cleaned.to_csv(file_path_edit, index=False)
    
    print(f"Cleaned CSV saved as {file_path_edit}")

file_path = 'grid_search_results_smote_full.csv'
clean_and_export_csv(file_path)
df = pd.read_csv('grid_search_results_smote_full_edit.csv')
#print(df.columns)
plot_distributions(df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'avg_max_depth']])
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 

# Load your dataset
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)
X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets
df = pd.DataFrame(X)
df['target'] = y

# Define the target column (the one you're predicting)
target_column = 'target'  # Replace with the name of your target column

# Check the distribution of classes
class_distribution = df[target_column].value_counts()

# Calculate the percentage distribution of each class
class_percentage = df[target_column].value_counts(normalize=True) * 100

# Print class distribution
print("Class Distribution:\n", class_distribution)
print("\nClass Percentage:\n", class_percentage)

# Visualize the class distribution
plt.figure(figsize=(8, 6))
class_distribution.plot(kind='bar', color='skyblue')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.show()

# Basic check for imbalance (threshold set to 80/20, can be adjusted)
imbalance_threshold = 0.8  # 80% for one class could indicate imbalance
most_frequent_class_percentage = class_percentage.max()

if most_frequent_class_percentage > imbalance_threshold * 100:
    print(f"\nWarning: The dataset may be imbalanced. The most frequent class represents {most_frequent_class_percentage:.2f}% of the data.")
else:
    print("\nThe dataset seems relatively balanced.")

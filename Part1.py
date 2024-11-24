import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data_path = 'C:/Users/hp/Desktop/DataMining/mushroom/agaricus-lepiota.data'
columns = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
]
data = pd.read_csv(data_path, header=None, names=columns)

# Replace '?' with NaN
data.replace('?', pd.NA, inplace=True)

# Drop duplicates
data.drop_duplicates(inplace=True)

# Handle missing values (e.g., drop rows with NaN)
data.dropna(inplace=True)

# Write the cleaned data back to the original file
data.to_csv(data_path, index=False)

# Display the first few rows
print(data.head())

# Check column names
print(data.columns)

# Check data types
print(data.dtypes)

# Summary statistics
print(data.describe(include='all'))

# Check for missing values
print(data.isnull().sum())

# Correlation heatmap for categorical variables
# Factorize the categorical variables to get numerical values for correlation
factorized_data = data.apply(lambda x: pd.factorize(x)[0])
correlation_matrix = factorized_data.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix for Categorical Variables')
plt.show()

# Stacked bar plot for the distribution of categorical variables
categorical_columns = data.columns[1:]  # Exclude the 'class' column

# Prepare the data for stacked bar plot
stacked_data = pd.DataFrame()
for col in categorical_columns:
    value_counts = data[col].value_counts().to_frame().T
    value_counts.index = [col]
    stacked_data = pd.concat([stacked_data, value_counts])

# Plot the stacked bar plot with unique colors for each value
stacked_data = stacked_data.fillna(0)
colors = sns.color_palette("tab20", n_colors=len(stacked_data.columns))

stacked_data.plot(kind='bar', stacked=True, figsize=(14, 8), color=colors)
plt.title('Distribution of Categorical Variables')
plt.xlabel('Categories')
plt.ylabel('Count')
plt.legend(title='Values', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Bar plot for the frequency of types (edible/poisonous)
plt.figure(figsize=(10, 6))
sns.countplot(x='class', data=data)
plt.xlabel('Class (edible=e, poisonous=p)')
plt.ylabel('Count')
plt.title('Frequency of Edible vs. Poisonous Mushrooms')
plt.show()

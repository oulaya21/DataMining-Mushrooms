import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load the dataset
data_path = 'C:/Users/hp/Desktop/DataMining/mushroom/agaricus-lepiota.data'
columns = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
]
data = pd.read_csv(data_path, header=None, names=columns)

# Select features and target variable
features = pd.get_dummies(data.drop(['class'], axis=1))
target = data['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Build a decision tree classifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(decision_tree, feature_names=features.columns, class_names=decision_tree.classes_, fontsize=6, filled=True)
plt.show()

# Make predictions on the test set using the decision tree classifier
y_pred_dt = decision_tree.predict(X_test)

# Calculate performance metrics for the decision tree classifier
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt, average='weighted')
recall_dt = recall_score(y_test, y_pred_dt, average='weighted')
f1_dt = f1_score(y_test, y_pred_dt, average='weighted')

# Build a random forest classifier
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)

# Make predictions on the test set using the random forest classifier
y_pred_rf = random_forest.predict(X_test)

# Calculate performance metrics for the random forest classifier
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

# Print the performance metrics
print('Decision Tree Classifier:')
print('Accuracy:', accuracy_dt)
print('Precision:', precision_dt)
print('Recall:', recall_dt)
print('F1 Score:', f1_dt)
print()
print('Random Forest Classifier:')
print('Accuracy:', accuracy_rf)
print('Precision:', precision_rf)
print('Recall:', recall_rf)
print('F1 Score:', f1_rf)

# Define a sample mushroom (replace with actual data)
sample_data = {
    'cap-shape': ['x'], 'cap-surface': ['s'], 'cap-color': ['n'], 'bruises': ['t'], 'odor': ['p'],
    'gill-attachment': ['f'], 'gill-spacing': ['c'], 'gill-size': ['n'], 'gill-color': ['k'],
    'stalk-shape': ['e'], 'stalk-root': ['e'], 'stalk-surface-above-ring': ['s'], 'stalk-surface-below-ring': ['s'],
    'stalk-color-above-ring': ['w'], 'stalk-color-below-ring': ['w'], 'veil-type': ['p'], 'veil-color': ['w'],
    'ring-number': ['o'], 'ring-type': ['p'], 'spore-print-color': ['k'], 'population': ['n'], 'habitat': ['g']
}

# Convert sample data to DataFrame
sample_df = pd.DataFrame(sample_data)

# Get dummies for sample data
sample_features = pd.get_dummies(sample_df).reindex(columns=features.columns, fill_value=0)

# Predict the class of the sample mushroom
sample_type_dt = decision_tree.predict(sample_features)[0]
sample_type_rf = random_forest.predict(sample_features)[0]

print('\nBased on the Decision Tree Classifier, the type of the sample mushroom is:', sample_type_dt)
print('Based on the Random Forest Classifier, the type of the sample mushroom is:', sample_type_rf)

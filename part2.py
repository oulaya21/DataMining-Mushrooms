import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from tabulate import tabulate

# Load the dataset
data_path = 'C:/Users/hp/Desktop/DataMining/mushroom/agaricus-lepiota.data'
columns = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
]
data = pd.read_csv(data_path, header=None, names=columns)

# Select relevant columns
df = data[['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
           'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
           'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']]

# Transform data into a transactional format
transactions = df.apply(lambda row: row.index[row.astype(bool)].tolist(), axis=1).tolist()

# Encode transactions
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Generate frequent itemsets using Apriori algorithm
frequent_itemsets = apriori(df_encoded, min_support=0.3, use_colnames=True, max_len=2)

# Generate association rules
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)

# Filter rules with 'class' in the RHS to get subrules
subrules = rules[rules['consequents'].apply(lambda x: 'class' in x)]

# Function to check if a rule is redundant
def is_redundant(rule, rules_df):
    for _, other_rule in rules_df.iterrows():
        if rule.name != other_rule.name:
            if (rule['antecedents'].issubset(other_rule['antecedents']) and
                rule['consequents'] == other_rule['consequents'] and
                rule['confidence'] <= other_rule['confidence']):
                return True
    return False

# Filter out redundant rules
if not subrules.empty:
    non_redundant_rules = subrules[~subrules.apply(lambda row: is_redundant(row, subrules), axis=1)]
else:
    non_redundant_rules = subrules

# Convert rules to a table format
def format_rules_table(rules_df):
    formatted_df = rules_df[['antecedents', 'consequents', 'confidence']].copy()
    formatted_df['antecedents'] = formatted_df['antecedents'].apply(lambda x: ', '.join(x))
    formatted_df['consequents'] = formatted_df['consequents'].apply(lambda x: ', '.join(x))
    return tabulate(formatted_df, headers='keys', tablefmt='psql')

rules_table_formatted = format_rules_table(rules)
subrules_table_formatted = format_rules_table(subrules)
non_redundant_rules_table_formatted = format_rules_table(non_redundant_rules)

# Print the formatted tables
print("Association Rules:\n")
print(rules_table_formatted)
print("\n")
print("Subrules:\n")
print(subrules_table_formatted)
print("\n")
print("Non-Redundant Rules:\n")
print(non_redundant_rules_table_formatted)

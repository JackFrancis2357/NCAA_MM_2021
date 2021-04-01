import pandas as pd

df = pd.read_csv('MM_2021_Data/Training_All.csv', index_col=0)
test_df = pd.read_csv('MM_2021_Data/test_All_2.csv', index_col=0)

unique_ordinals = ['MAS', 'PGH', 'KPK', 'SAG', 'MOR', 'POM', 'WIL', 'DOK', 'COL', 'DOL', 'RTH', 'WLK', 'WOL']

order_to_delete = ['WOL', 'RTH', 'COL', 'DOL', 'WLK']

order_selection = order_to_delete


def clean_data(df, un_ord, ord_sel):
    for i in un_ord:
        if i in ord_sel:
            df = df.drop(f'{i}_Team_1', axis=1)
            df = df.drop(f'{i}_Team_2', axis=1)
        else:
            df = df[df[f'{i}_Team_1'] != 'No Ranking Found']
            df = df[df[f'{i}_Team_2'] != 'No Ranking Found']
            df[f'{i}_Team_1'] = pd.to_numeric(df[f'{i}_Team_1'])
            df[f'{i}_Team_2'] = pd.to_numeric(df[f'{i}_Team_2'])
    my_ordinals_to_use = list((set(un_ord) | set(ord_sel)) - (set(un_ord) & set(ord_sel)))

    for i in my_ordinals_to_use:
        df[f'{i}_Differential'] = df[f'{i}_Team_1'] - df[f'{i}_Team_2']

    df['Year_Played'] = df['Year']
    df['Results_Final'] = df['Results']
    df = df.drop('Year', axis=1)
    df = df.drop('Results', axis=1)
    df = df.reset_index(drop=True)
    return df


training_df = clean_data(df, un_ord=unique_ordinals, ord_sel=order_selection)
testing_df = clean_data(test_df, un_ord=unique_ordinals, ord_sel=order_selection)

# Now let's do some machine learning lol
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.neural_network import MLPClassifier

# First try only differentials as input features
# Make selection
my_ordinals_to_use = list((set(unique_ordinals) | set(order_selection)) - (set(unique_ordinals) & set(order_selection)))

feature_method_list = ['all', 'only_diff', 'no_diff']
for feature_method in feature_method_list:
    begin, end = 0, 0
    if feature_method == 'all':
        begin = 2
        end = -1
    elif feature_method == 'only_diff':
        begin = len(my_ordinals_to_use) * 2 + 2
        end = -1
    elif feature_method == 'no_diff':
        begin = 2
        end = len(my_ordinals_to_use) * 2 + 2

    X_train = training_df.iloc[:, begin:end]
    y_train = training_df.iloc[:, -1]

    X_test = testing_df.iloc[:, begin:end]
    y_test = testing_df.iloc[:, -1]

    print(X_train.shape)
    print(X_train.columns)

    print(X_test.shape)
    print(X_test.columns)

    lr = MLPClassifier(hidden_layer_sizes=(8, 4), alpha=0.1, max_iter=1000, tol=1e-9, solver='adam')
    lr.fit(X_train, y_train)
    y_pred = lr.predict_proba(X_test)
    print(feature_method, log_loss(y_test, y_pred))

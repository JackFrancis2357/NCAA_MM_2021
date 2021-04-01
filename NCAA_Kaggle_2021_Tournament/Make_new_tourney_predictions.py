import pandas as pd
from bracketeer import build_bracket

df = pd.read_csv('MM_2021_Data/Training_All.csv', index_col=0)
df2 = pd.read_csv('MM_2021_Data/test_All_2.csv', index_col=0)
test_df = pd.read_csv('MM_2021_Data/Test_data_2021_Tourney.csv', index_col=0)

# train_df = pd.concat([df, df2], ignore_index=True)
train_df = df
unique_ordinals = ['MAS', 'PGH', 'KPK', 'SAG', 'MOR', 'POM', 'WIL', 'DOK', 'COL', 'DOL', 'RTH', 'WLK', 'WOL']
order_to_delete = ['WOL', 'RTH', 'COL', 'DOL', 'WLK']
order_selection = order_to_delete


def clean_data(df, un_ord, ord_sel, test=False):
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
    if not test:
        df['Results_Final'] = df['Results']
        df = df.drop('Results', axis=1, errors='ignore')
    df = df.drop('Year', axis=1)
    df = df.reset_index(drop=True)
    return df


training_df = clean_data(df, un_ord=unique_ordinals, ord_sel=order_selection)
validation_df = clean_data(df2, un_ord=unique_ordinals, ord_sel=order_selection)
testing_df = clean_data(test_df, un_ord=unique_ordinals, ord_sel=order_selection, test=True)

# Now let's do some machine learning lol
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier

# First try only differentials as input features
# Make selection
my_ordinals_to_use = list((set(unique_ordinals) | set(order_selection)) - (set(unique_ordinals) & set(order_selection)))

feature_method_list = ['all']
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
    X_valid = validation_df.iloc[:, begin:end]
    y_train = training_df.iloc[:, -1]
    y_valid = validation_df.iloc[:, -1]

    if feature_method == 'no_diff':
        X_test = testing_df.iloc[:, begin:end]
    else:
        X_test = testing_df.iloc[:, begin:]

    # lr = MLPClassifier(hidden_layer_sizes=(32, 16, 8, 4), alpha=0.1, max_iter=300, tol=1e-4, solver='adam', verbose=1,
    #                    n_iter_no_change=20)
    # lr = LogisticRegression(max_iter=1000)
    lr = SVC(probability=True, verbose=1, cache_size=1000)
    # lr = GradientBoostingClassifier()
    lr.fit(X_train, y_train)
    y_valid_pred = lr.predict_proba(X_valid)
    print('\n')
    print(log_loss(y_true=y_valid, y_pred=y_valid_pred))
    y_pred = lr.predict_proba(X_test)


my_preds = y_pred[:, 1]

sample_submission = pd.read_csv("MDataFiles_Stage2/MSampleSubmissionStage2.csv")
sample_submission['Pred'] = my_preds
sample_submission.to_csv('my_predictions.csv')

b = build_bracket(
    outputPath='output.png',
    teamsPath='MDataFiles_Stage2/MTeams.csv',
    seedsPath='MDataFiles_Stage2/MNCAATourneySeeds.csv',
    submissionPath='my_predictions.csv',
    slotsPath='MDataFiles_Stage2/MNCAATourneySlots.csv',
    year=2021
)

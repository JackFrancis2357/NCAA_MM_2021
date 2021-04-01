import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, log_loss

all_games_df = pd.read_csv("Training_Data_2003_2019.csv")
test_df = all_games_df[all_games_df["Season"] > 2014]
train_df = all_games_df[all_games_df["Season"] < 2015]

x_train = train_df.iloc[:, 4:-1]
y_train = train_df.iloc[:, -1]
x_test = test_df.iloc[:, 4:-1]
y_test = test_df.iloc[:, -1]

lr_model = LogisticRegressionCV(cv=10, max_iter=1000)
# lr_model.fit(x_train, y_train)

svc_model = SVC(probability=True)
# svc_model.fit(x_train, y_train)

gb_model = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.3, max_depth=3, random_state=0)
# gb_model.fit(x_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100)
# rf_model.fit(x_train, y_train)

vc_model = VotingClassifier(estimators=[('lr', lr_model), ('rf', rf_model)],
                            voting='soft')
vc_model.fit(x_train, y_train)
y_pred = vc_model.predict_proba(x_test)
# y_pred[234, 1] = 0.5
print(log_loss(y_test, y_pred))

test_games_df = pd.read_csv("Test_Games.csv")
x_new_test = test_games_df.iloc[:, 4:-1]
y_new_pred = vc_model.predict_proba(x_new_test)

submission_df = pd.DataFrame(data=0, index=range(test_games_df.shape[0]), columns=["ID", "Pred"])
for i in range(submission_df.shape[0]):
    submission_df.iloc[i, 0] = test_games_df.iloc[i, 1].astype(str) + "_" + test_games_df.iloc[i, 2].astype(str) + "_" + \
                               test_games_df.iloc[i, 3].astype(str)
    submission_df.iloc[i, 1] = y_new_pred[i, 1]

submission_df.to_csv("Test_Submission.csv", index=False)

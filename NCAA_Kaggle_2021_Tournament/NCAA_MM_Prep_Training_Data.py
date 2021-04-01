import numpy as np
import pandas as pd
from itertools import combinations


def get_training_data(tourney_results, beg_season, end_season):
    training_colnames = ['Season', 'Team_1', 'Team_2',
                         'T1_COL', 'T1_DOL', 'T1_MOR', 'T1_POM', 'T1_RTH', 'T1_SAG', 'T1_WLK', 'T1_WOL', 'T1_Seed',
                         'T2_COL', 'T2_DOL', 'T2_MOR', 'T2_POM', 'T2_RTH', 'T2_SAG', 'T2_WLK', 'T2_WOL', 'T2_Seed',
                         'Result']
    df = pd.DataFrame(data=0, index=range(tourney_results.shape[0]), columns=training_colnames)
    counter = 0
    for season in range(beg_season, end_season):
        seasonal_tourney_results = tourney_results[tourney_results["Season"] == season]
        seasonal_tourney_seeds = tourney_seeds[tourney_seeds["Season"] == season]
        seasonal_ordinals = ordinals[ordinals["Season"] == season]
        for i in range(seasonal_tourney_results.shape[0]):
            team_1 = min(seasonal_tourney_results.iloc[i, 1], seasonal_tourney_results.iloc[i, 2])
            team_2 = max(seasonal_tourney_results.iloc[i, 1], seasonal_tourney_results.iloc[i, 2])
            team_1_ordinals = seasonal_ordinals[seasonal_ordinals["TeamID"] == team_1]["OrdinalRank"].values
            team_1_seed = seasonal_tourney_seeds[seasonal_tourney_seeds["TeamID"] == team_1]["Seed"].values
            team_2_ordinals = seasonal_ordinals[seasonal_ordinals["TeamID"] == team_2]["OrdinalRank"].values
            team_2_seed = seasonal_tourney_seeds[seasonal_tourney_seeds["TeamID"] == team_2]["Seed"].values
            if seasonal_tourney_results.iloc[i, 1] == team_1:
                outcome = np.asarray([1])
            else:
                outcome = np.asarray([0])
            df.iloc[counter, :] = np.concatenate(
                (np.asarray([season]), np.asarray([team_1]), np.asarray([team_2]), team_1_ordinals, team_1_seed,
                 team_2_ordinals, team_2_seed, outcome), axis=0)
            counter += 1
            if counter % 100 == 0:
                print("Done with iteration %s" % counter)
    return df


ordinals = pd.read_csv("MMasseyOrdinals.csv")
unique_ordinals = ['COL', 'DOL', 'MOR', 'POM', 'RTH', 'SAG', 'WLK', 'WOL']
ordinals = ordinals[ordinals["SystemName"].isin(unique_ordinals)]
ordinals = ordinals[ordinals["RankingDayNum"] == 133]
ordinals = ordinals.sort_values(by=["Season", "SystemName"])

tourney_seeds = pd.read_csv("MNCAATourneySeeds.csv")
tourney_seeds = tourney_seeds[tourney_seeds["Season"] > 2002]
tourney_seeds["Seed"] = tourney_seeds["Seed"].str.extract('(\d+)').astype('int64')

team_list_actual_games = pd.read_csv("MNCAATourneyDetailedResults.csv", usecols=["Season", "WTeamID", "LTeamID"])

# training_df = get_training_data(team_list_actual_games, beg_season=2003, end_season=2020)
# training_df.to_csv("Training_Data_2003_2019.csv")

# Get the ids for all games in 2015-2019 tourney
for season in range(2015, 2020):
    seasonal_teams = team_list_actual_games[team_list_actual_games["Season"] == season]
    test = np.unique(seasonal_teams[['WTeamID', 'LTeamID']].values)
    # print(list(combinations(test, 2)))
    if season == 2015:
        df = pd.DataFrame(data=list(combinations(test, 2)), columns=['Team_1', 'Team_2'])
        df.insert(loc=0, column="Season", value=season)
        print(df.shape)
    else:
        new_df = pd.DataFrame(data=list(combinations(test, 2)), columns=['Team_1', 'Team_2'])
        new_df.insert(loc=0, column="Season", value=season)
        df = pd.concat((df, new_df))

all_test_games = get_training_data(df, 2015, 2020)
all_test_games.to_csv("Test_Games.csv")

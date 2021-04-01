import pandas as pd
import numpy as np
import time

# TODO for 2022, add in year as part of the training row. Need that for model building.
ordinals = pd.read_csv('MDataFiles_Stage2/MMasseyOrdinals.csv')
unique_ordinals = ['MAS', 'PGH', 'KPK', 'SAG', 'MOR', 'POM', 'WIL', 'DOK', 'COL', 'DOL', 'RTH', 'WLK', 'WOL']
ordinals = ordinals[ordinals["SystemName"].isin(unique_ordinals)]
final_ordinals = ordinals[ordinals["Season"] >= 2010]


def get_ranks(team_id, seasonal_ordinals, uniq_ord):
    ranking_list = []
    for ord in uniq_ord:
        gameday_ords = seasonal_ordinals[season_ords_df['SystemName'] == ord]
        gameday_ords = gameday_ords[gameday_ords['RankingDayNum'] < day]
        try:
            team_rank = gameday_ords[gameday_ords['TeamID'] == team_id]['OrdinalRank'].values[-1]
            ranking_list.append(team_rank)
        except IndexError:
            ranking_list.append('No Ranking Found')
    return ranking_list


training = True
testing = False

regular_season_games_df = pd.read_csv("MDataFiles_Stage2/MRegularSeasonCompactResults.csv")
tourney_season_games_df = pd.read_csv("MM_2021_Data/MNCAATourneyCompactResults.csv")

if training:
    identifier = 'training'
    my_games_df = regular_season_games_df
elif testing:
    identifier = 'test'
    my_games_df = tourney_season_games_df

team_1_cols = [f'{x}_Team_1' for x in unique_ordinals]
team_2_cols = [f'{x}_Team_2' for x in unique_ordinals]
training_cols = ['Year'] + ['Team_1_ID'] + ['Team_2_ID'] + team_1_cols + team_2_cols + ['Results']

num_years = range(2020, 2021)
training_df = pd.DataFrame(0, range(my_games_df.shape[0]), columns=training_cols)
ctr = 0

start = time.time()
for year in num_years:
    games_df = my_games_df[my_games_df['Season'] == year]
    season_ords_df = final_ordinals[final_ordinals['Season'] == year]
    for day in range(110, 200):
        day_games_df = games_df[games_df['DayNum'] == day]
        if day_games_df.shape[0] == 0:
            continue
        else:
            for day_game_num in range(day_games_df.shape[0]):
                team_1_id = day_games_df.iloc[day_game_num, 2]
                team_2_id = day_games_df.iloc[day_game_num, 4]
                team_1_rankings = get_ranks(team_id=team_1_id,
                                            seasonal_ordinals=season_ords_df,
                                            uniq_ord=unique_ordinals)
                team_2_rankings = get_ranks(team_id=team_2_id,
                                            seasonal_ordinals=season_ords_df,
                                            uniq_ord=unique_ordinals)
                if np.random.rand() < 0.5:
                    training_row = [year] + [team_1_id] + [team_2_id] + team_1_rankings + team_2_rankings + [1]
                else:
                    training_row = [year] + [team_2_id] + [team_1_id] + team_2_rankings + team_1_rankings + [0]
                training_df.iloc[ctr, :] = training_row
                ctr += 1
        print(f'Finished with day {day}')
    print(f'Finished with season {year}')
print(time.time() - start)
training_df.to_csv(f'MM_2021_Data/{identifier}_2020_addon.csv')

import pandas as pd

sample_sub_file = pd.read_csv("MDataFiles_Stage2/MSampleSubmissionStage2.csv")
sample_sub_file = sample_sub_file.drop('Pred', axis=1)
sample_sub_file[['Year', 'Team_1', 'Team_2']] = sample_sub_file['ID'].str.split('_', expand=True)
print(sample_sub_file.head())


def get_ranks(team_id, seasonal_ordinals, uniq_ord, day):
    ranking_list = []
    for ord in uniq_ord:
        gameday_ords = seasonal_ordinals[seasonal_ordinals['SystemName'] == ord]
        gameday_ords = gameday_ords[gameday_ords['RankingDayNum'] < day]
        try:
            team_rank = gameday_ords[gameday_ords['TeamID'] == team_id]['OrdinalRank'].values[-1]
            ranking_list.append(team_rank)
        except IndexError:
            ranking_list.append('No Ranking Found')
    return ranking_list


year = 2020

ordinals = pd.read_csv('MDataFiles_Stage2/MMasseyOrdinals.csv')
unique_ordinals = ['MAS', 'PGH', 'KPK', 'SAG', 'MOR', 'POM', 'WIL', 'DOK', 'COL', 'DOL', 'RTH', 'WLK', 'WOL']
ordinals = ordinals[ordinals["SystemName"].isin(unique_ordinals)]
season_ords_df = ordinals[ordinals["Season"] >= year]
print(season_ords_df.shape[0])

team_1_cols = [f'{x}_Team_1' for x in unique_ordinals]
team_2_cols = [f'{x}_Team_2' for x in unique_ordinals]
training_cols = ['Year'] + ['Team_1_ID'] + ['Team_2_ID'] + team_1_cols + team_2_cols
tourney_test_df = pd.DataFrame(0, range(sample_sub_file.shape[0]), columns=training_cols)

for i in range(sample_sub_file.shape[0]):
    team_1_id = int(sample_sub_file.iloc[i, 2])
    team_2_id = int(sample_sub_file.iloc[i, 3])
    team_1_rankings = get_ranks(team_id=team_1_id,
                                seasonal_ordinals=season_ords_df,
                                uniq_ord=unique_ordinals,
                                day=200)
    team_2_rankings = get_ranks(team_id=team_2_id,
                                seasonal_ordinals=season_ords_df,
                                uniq_ord=unique_ordinals,
                                day=200)
    training_row = [year] + [team_1_id] + [team_2_id] + team_1_rankings + team_2_rankings
    tourney_test_df.iloc[i, :] = training_row
    if i % 100 == 0:
        print(i)
        print(training_row)

tourney_test_df.to_csv('MM_2021_Data/Test_data_2021_Tourney.csv')

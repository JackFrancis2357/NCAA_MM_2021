import pandas as pd

ordinals = pd.read_csv("MM_2021_Data/MMasseyOrdinals.csv")
# unique_ordinals = ['COL', 'DOL', 'MOR', 'POM', 'RTH', 'SAG', 'WLK', 'WOL']
# ordinals = ordinals[ordinals["SystemName"].isin(unique_ordinals)]
# ordinals = ordinals[ordinals["RankingDayNum"] == 133]
# ordinals = ordinals.sort_values(by=["Season", "SystemName"])
# ordinals.to_csv('MM_2021_Data/MMassey_Ord_Cleaned.csv')

all_ranking_metrics = set(ordinals['SystemName'])
first_year = 2010
last_year = 2020
num_years = last_year - first_year
ords_df = pd.DataFrame(0, index=range(num_years), columns=all_ranking_metrics)
for year in range(num_years):
    for ranking_metric in all_ranking_metrics:
        first_step = ordinals[ordinals['Season'] == year+first_year]
        second_step = first_step[first_step['SystemName'] == ranking_metric]
        ords_df.loc[year, ranking_metric] = second_step.shape[0]
ords_df.to_csv("MM_2021_Data/MMassey_2010_2020.csv")

ordinals_to_use = ['MAS', 'PGH', 'KPK', 'SAG', 'MOR', 'POM', 'WIL', 'DOK', 'COL', 'DOL', 'RTH', 'WLK', 'WOL']
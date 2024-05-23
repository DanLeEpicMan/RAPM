import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Ridge
import nba_on_court as noc
from nba_api.stats.endpoints import playbyplayv2
from preprocess_tools import preprocess_game, get_design_matrix


# load data
nba_data = pd.read_csv('data/combined_data_2022.csv')
games = pd.unique(nba_data['GAME_ID'])
list_of_game_data = []

for game in games:
    current = nba_data[nba_data['GAME_ID'] == game].reset_index(drop=True)
    current = noc.players_on_court(current)
    list_of_game_data.append(preprocess_game(current))

data = pd.concat(list_of_game_data, ignore_index=True)

design_matrix, players = get_design_matrix(data, return_players=True)

model = Ridge(alpha=1).fit(X=design_matrix, y=data['PM'])

results = pd.DataFrame({
        'Player': noc.players_name(players),
        'RAPM': model.coef_
})

print(results)
results.to_csv('results.csv')

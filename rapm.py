import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Ridge
import nba_on_court as noc
from nba_api.stats.endpoints import playbyplayv2

# load data of a specific game
nbastats = pd.read_csv('data/nbastats_po_2022.csv')
pbpstats = pd.read_csv('data/pbpstats_po_2022.csv')

nbastats = nbastats.loc[nbastats['GAME_ID'] == 42200405].reset_index(drop=True) 
pbpstats = pbpstats.loc[pbpstats['GAMEID'] == 42200405].reset_index(drop=True) 

full_data = noc.players_on_court(noc.left_join_nbastats(nbastats, pbpstats))

# extract features we care about
subset = full_data[
    [
        'SCOREMARGIN',
        'TURNOVERS', 
        'AWAY_PLAYER1', 'AWAY_PLAYER2', 'AWAY_PLAYER3', 'AWAY_PLAYER4', 'AWAY_PLAYER5', 
        'HOME_PLAYER1', 'HOME_PLAYER2', 'HOME_PLAYER3', 'HOME_PLAYER4', 'HOME_PLAYER5'
     ]
]

# have to preprocess stints that end in scores differently from stints that end in turnovers
# because otherwise the PM can't be calculated
scores = subset[~subset['SCOREMARGIN'].isna()].replace('TIE', 0).reset_index(drop=True)
scores['PM'] = scores['SCOREMARGIN'].astype(np.int64).diff().replace(np.nan, 0)
scores.at[0, 'PM'] = np.int64(scores.at[0, 'SCOREMARGIN']) # manually add in the first PM

turnovers = subset[subset['TURNOVERS'] == 1].reset_index(drop=True)
turnovers['SCOREMARGIN'].fillna(0, inplace=True)
turnovers['PM'] = np.zeros(len(turnovers), dtype=np.int64)

subset = pd.concat([scores, turnovers], ignore_index=True)

# get indicators of players
# -1 := away team
# 0  := not on court
# 1  := home team
all_players = np.unique(subset.filter(like='PLAYER').to_numpy())
design_matrix = np.empty((len(subset), len(all_players)))        # stints x players. to be filled

def determine_coefficient(row: pd.Series, player_id: int) -> int:
    '''
    Determines whether a player's coefficient is -1, 0, 1 in a given stint.
    '''
    away = row[['AWAY_PLAYER1', 'AWAY_PLAYER2', 'AWAY_PLAYER3', 'AWAY_PLAYER4', 'AWAY_PLAYER5']].to_list()
    home = row[['HOME_PLAYER1', 'HOME_PLAYER2', 'HOME_PLAYER3', 'HOME_PLAYER4', 'HOME_PLAYER5']].to_list()
    if player_id in away:
        return -1
    elif player_id in home:
        return 1
    else:
        return 0

# fill the design matrix with coefficients
for i in range(len(all_players)):
    player = all_players[i]
    design_matrix[:, i] = subset.apply(determine_coefficient, axis=1, player_id=player).to_numpy()

# fit the model and print results
model = Ridge(alpha=1).fit(X=design_matrix, y=subset['PM'])

results = pd.DataFrame({
        'Player': noc.players_name(all_players),
        'RAPM': model.coef_
})

print(results)

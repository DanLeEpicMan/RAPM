'''
Contains all functions used for preprocessing basketball games.
'''
import numpy as np
import pandas as pd
import nba_on_court as noc
from typing import Tuple


_BASE_FEATURES  = [
    'GAME_ID',
    'SCOREMARGIN',
    'TURNOVERS', 
    'PLAYER1_ID', 
    'AWAY_PLAYER1', 'AWAY_PLAYER2', 'AWAY_PLAYER3', 'AWAY_PLAYER4', 'AWAY_PLAYER5', 
    'HOME_PLAYER1', 'HOME_PLAYER2', 'HOME_PLAYER3', 'HOME_PLAYER4', 'HOME_PLAYER5'
]

def preprocess_season(season: pd.DataFrame) -> pd.DataFrame:
    '''
    Given a full season of raw data, preprocess the data
    and return the resulting DataFrame. Note that this
    is essentially `preprocess_game` but for an entire season.

    This will return a DataFrame with the following features:
      `GAME_ID`: The game that the stint belongs to.
      `SCOREMARGIN`: The score margin of the stint. Note that positive means in favor for home, negative in favor for away.
      `TURNOVERS`: Indicator of whether the stint resulted in a turnover.
      `BALL_ID`: (Presumed to be) the player who possessed the ball during the stint.
      `PM`: The Plus-Minus of the stint. See SCOREMARGIN for meaning of the sign.
      `[AWAY|HOME]_PLAYER[NUM]`: Player ID of an Away/Home player.
    '''
    games = pd.unique(season['GAME_ID'])
    list_of_game_data = []

    for game in games:
        current = season[season['GAME_ID'] == game].reset_index(drop=True)
        current = noc.players_on_court(current)
        list_of_game_data.append(preprocess_game(current))

    return pd.concat(list_of_game_data, ignore_index=True)

def preprocess_game(game_data: pd.DataFrame) -> pd.DataFrame:
    '''
    Given the full data of a game (including players on court), 
    preprocess the game data and return the result.

    This will return a DataFrame with the following features:
      `GAME_ID`: The game that the stint belongs to.
      `SCOREMARGIN`: The score margin of the stint. Note that positive means in favor for home, negative in favor for away.
      `TURNOVERS`: Indicator of whether the stint resulted in a turnover.
      `BALL_ID`: (Presumed to be) the player who possessed the ball during the stint.
      `PM`: The Plus-Minus of the stint. See SCOREMARGIN for meaning of the sign.
      `[AWAY|HOME]_PLAYER[NUM]`: Player ID of an Away/Home player.
    '''
    # extract features we care about
    subset = game_data[_BASE_FEATURES]
    subset.rename(columns={'PLAYER1_ID': 'BALL_ID'}, inplace=True)

    # have to preprocess stints that end in scores differently from stints that end in turnovers
    # because otherwise the PM can't be calculated
    scores = subset[~subset['SCOREMARGIN'].isna()].replace('TIE', 0).reset_index(drop=True)
    scores['PM'] = scores['SCOREMARGIN'].astype(np.int64).diff().replace(np.nan, 0)
    scores.at[0, 'PM'] = np.int64(scores.at[0, 'SCOREMARGIN']) # manually add in the first PM
    score_stints = scores[(scores['TURNOVERS'] == 0) & ((scores['BALL_ID'].isna() == False) & (scores['BALL_ID'] != 0))]

    turnovers = subset[subset['TURNOVERS'] == 1].reset_index(drop=True)
    turnovers['SCOREMARGIN'].fillna(0, inplace=True)
    turnovers['PM'] = np.zeros(len(turnovers), dtype=np.int64)
    turnover_stints = turnovers[((turnovers['BALL_ID'].isna() == False) & (turnovers['BALL_ID'] != 0))]

    return pd.concat([score_stints, turnover_stints], ignore_index=True)

_AWAY_LIST = ['AWAY_PLAYER1', 'AWAY_PLAYER2', 'AWAY_PLAYER3', 'AWAY_PLAYER4', 'AWAY_PLAYER5']
_HOME_LIST = ['HOME_PLAYER1', 'HOME_PLAYER2', 'HOME_PLAYER3', 'HOME_PLAYER4', 'HOME_PLAYER5']

def _determine_coefficient(row: pd.Series, player_id: int) -> int:
    '''
    Determines whether a player's coefficient is -1, 0, 1 in a given stint.

    This is meant to be used in the `get_design_matrix` function.
    '''
    away = row[_AWAY_LIST].values
    home = row[_HOME_LIST].values
    if player_id in away:
        return -1
    elif player_id in home:
        return 1
    else:
        return 0

def get_design_matrix(game_data: pd.DataFrame, *, return_players: bool = False) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    '''
    Given a preprocessed game/season of data, return a design matrix for a regression model.
    See `preprocess_game`

    Each row of the matrix will correspond to a stint, and each column corresponds to a specific player.
    The matrix will be filled as follows:
      `-1`: Present on the Away team.
      `0`: Not present during the stint.
      `1`: Present on the Home team. 

    If `return_players` is True, then this will also return
    a list of all player IDs encountered in processing.
    The order of this list matches the order of columns in the design matrix.
    '''
    all_players = np.unique(game_data.filter(like='PLAYER').to_numpy())
    design_matrix = np.empty((len(game_data), len(all_players)))        # stints x players. to be filled

    # fill the design matrix with coefficients
    for i in range(len(all_players)):
        player = all_players[i]
        design_matrix[:, i] = game_data.apply(_determine_coefficient, axis=1, player_id=player).to_numpy()

    if return_players:
        return design_matrix, all_players
    else:
        return design_matrix
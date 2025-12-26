import pandas as pd
import numpy as np
from src.data_loader import load_all_datasets

class FeatureEngineering:
    # initialize variable names
    def __init__(self):
        self.datasets = load_all_datasets()
        self.kb = self.datasets["kenpom_barttorvik"]
        self.resumes = self.datasets["resumes"]
        self.team_rankings = self.datasets["team_rankings"]
        self.tournament_matchups = self.datasets["tournament_matchups"]
    
    # combine everything into one big dataset
    def combine_into_one_dataset(self):
        '''
        Example Output:
        Year | Team NO | Team | Seed | ... | Q1 W | Q2 W | Q1 PLUS Q2 W | ... | TR RATING | SOS RANK | LUCK RANK | ... |
        '''

        # get a subset of features from kb dataset
        kenpom_bart = self.kb[[
            'YEAR', 'TEAM NO', 'TEAM', 'SEED',  # year, team number, team name, seed number
            'BADJ EM', 'BADJ O', 'BADJ D',      # adjusted efficiency metrics
            'EFG%', 'EFG%D',                    # effective field goal percentages
            '3PT%', '3PT%D',                    # 3-point shooting percentages
            'FTR', 'FTRD',                      # free throw rates
            'TOV%', 'TOV%D',                    # turnover percentages
            'OREB%', 'DREB%',                   # rebound percentages
            'AST%'                              # assist percentages
        ]].copy()

        # get a subset of features from resumes dataset
        resumes = self.resumes[[
            'YEAR', 'TEAM NO',
            'Q1 W', 'Q2 W',                     # quality wins (Q1 = a win against a top 25 team, Q2 = a win against a team that's ranked 26-50)
            'Q1 PLUS Q2 W',                     # total quality wins
            'ELO',                              # ELO rating
            'NET RPI'                           # net RPI ranking
        ]].copy()

        # merge kb and resumes together
        kenpom_bart = kenpom_bart.merge(resumes, on=['YEAR', 'TEAM NO'], how='left')

        # get a subset of features from team rankings dataset
        tr = self.team_rankings[[
            'YEAR', 'TEAM NO',
            'TR RATING',                        # team rankings power rating
            'SOS RANK',                         # strength of schedule rank
            'LUCK RANK',                        # luck factor rank
            'CONSISTENCY RANK',                 # consistency rank
            'V 1-25 WINS', 'V 1-25 LOSS'        # record vs ranked teams
        ]].copy()

        # merge the new kb and tr to form one big row of features
        kenpom_bart = kenpom_bart.merge(tr, on=['YEAR', 'TEAM NO'], how='left')

        return kenpom_bart
    
    # creates pairs of teams that simulate two teams playing each other and determines a winner between the two teams
    def create_matchup_pairs(self):
        matchups = self.tournament_matchups.copy()
        matchup_pairs = []

        # go through every year in order
        for year in sorted(matchups['YEAR'].unique()):
            # get the year's data
            year_data = matchups[matchups['YEAR'] == year]

            # go through every round number
            for round_num in sorted(year_data['ROUND'].unique()):
                # get the round number's data
                round_data = year_data[year_data['ROUND'] == round_num]

                # sort from decreasing
                round_data = round_data.reset_index(drop=True)

                # go through and pair up adjacent teams
                for i in range(0, len(round_data) - 1, 2):
                    team_1_row = round_data.iloc[i]
                    team_2_row = round_data.iloc[i + 1]

                    # get the team names
                    team_1_name = team_1_row['TEAM']
                    team_2_name = team_2_row['TEAM']

                    # get the team scores
                    team_1_score = team_1_row['SCORE']
                    team_2_score = team_2_row['SCORE']

                    # handle missing scores (future predictions)
                    if pd.isna(team_1_score) or pd.isna(team_2_score):
                        winner = None
                    else:
                        # determine the winner (1 = team 1 wins, 0 = team 2 wins)
                        winner = 1 if team_1_score > team_2_score else 0

                    # add the pair to matchup_pairs
                    matchup_pairs.append({
                        'YEAR': year,
                        'ROUND': round_num,
                        'TEAM_1_NAME': team_1_name,
                        'TEAM_2_NAME': team_2_name,
                        'TEAM_1_SCORE': team_1_score,
                        'TEAM_2_SCORE': team_2_score,
                        'WINNER': winner
                    })
        matchup_df = pd.DataFrame(matchup_pairs)
        return matchup_df
    
    # combines the matchup pairs with both teams' stats
    def create_matchup_features(self):
        team_features = self.combine_into_one_dataset()
        matchup_df = self.create_matchup_pairs()

        # align column names for merging
        matchup_df = matchup_df.rename(columns={
            'TEAM_1_NAME': 'TEAM1',
            'TEAM_2_NAME': 'TEAM2'
        })

        # rename team_features columns for team 1
        team_1_features = team_features.copy()
        team_1_cols = {col: f'{col}_T1' for col in team_1_features.columns if col not in ['YEAR', 'TEAM']}
        team_1_features = team_1_features.rename(columns=team_1_cols)
        team_1_features = team_1_features.rename(columns={'TEAM': 'TEAM1'})
        
        # merge team 1's features
        matchup_df = matchup_df.merge(
            team_1_features[['YEAR', 'TEAM1'] + [col for col in team_1_features.columns if '_T1' in col]],
            on=['YEAR', 'TEAM1'],
            how='left'
        )
        
        # rename team_features columns for team 2
        team_2_features = team_features.copy()
        team_2_cols = {col: f'{col}_T2' for col in team_2_features.columns if col not in ['YEAR', 'TEAM']}
        team_2_features = team_2_features.rename(columns=team_2_cols)
        team_2_features = team_2_features.rename(columns={'TEAM': 'TEAM2'})
        
        # merge team 2's features
        matchup_df = matchup_df.merge(
            team_2_features[['YEAR', 'TEAM2'] + [col for col in team_2_features.columns if '_T2' in col]],
            on=['YEAR', 'TEAM2'],
            how='left'
        )
        
        return matchup_df
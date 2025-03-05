import numpy as np
import data_loader
import joblib

#Open the raw data using built-in function and store it using
file_path = "../data/S18/raw_data/battlesStaging_12272020_WL_tagged.csv"
data_loader(file_path)
df = joblib.load(f'data/joblib/{file_path}')

#Pre-processing by shuffling winner/loser order and adding winner_value column to be predicted
df_mod = df[[
 'winner.card1.id',
 'winner.card2.id',
 'winner.card3.id',
 'winner.card4.id',
 'winner.card5.id',
 'winner.card6.id',
 'winner.card7.id',
 'winner.card8.id',
 'winner.totalcard.level',
 'winner.elixir.average',
 'loser.card1.id',
 'loser.card2.id',
 'loser.card3.id',
 'loser.card4.id',
 'loser.card5.id',
 'loser.card6.id',
 'loser.card7.id',
 'loser.card8.id',
 'loser.totalcard.level',
 'loser.elixir.average']].copy()

#Rename winner and loser by player 1 and 2
df_mod.rename(columns={col: col.replace('winner', 'player1') for col in df_mod.columns}, inplace=True)
df_mod.rename(columns={col: col.replace('loser', 'player2') for col in df_mod.columns}, inplace=True)

#Add predicted value
df_mod['winner.value'] = np.random.choice([0, 1], size=len(df_mod))

column_pairs = [
        ('player1.card1.id', 'player2.card1.id'),
        ('player1.card2.id', 'player2.card2.id'),
        ('player1.card3.id', 'player2.card3.id'),
        ('player1.card4.id', 'player2.card4.id'),
        ('player1.card5.id', 'player2.card5.id'),
        ('player1.card6.id', 'player2.card6.id'),
        ('player1.card7.id', 'player2.card7.id'),
        ('player1.card8.id', 'player2.card8.id'),
        ('player1.totalcard.level', 'player2.totalcard.level'),
        ('player1.elixir.average', 'player2.elixir.average')
        ]

# Create a mask for when 'winner.value' is 1 (indicating player 2 is the winner)
mask = df_mod['winner.value'] == 1

# Efficiently swap the columns for the rows where 'winner.value' is 1
for col1, col2 in column_pairs:
    df_mod.loc[mask, col1], df_mod.loc[mask, col2] = df_mod.loc[mask, col2], df_mod.loc[mask, col1]

print(df_mod.sample(n=10))
df_mod.to_csv('../data/S18/pre_prep/pre_prep_battlesStaging_12272020_WL_tagged.csv', index=False)
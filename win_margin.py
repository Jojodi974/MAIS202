import numpy as np
import data_loader
import joblib

#Open the raw data using built-in function
file_path = "../data/S18/raw_data/battlesStaging_12272020_WL_tagged.csv"
data_loader(file_path)
df = joblib.load(f'data/joblib/{file_path}')

###Obtain win margin

#New columns to create:
#winner.hp
#loser.hp
#hp.difference
#crown.difference
#win.margin using sigmoid function and normalized value

# Define the function to handle NaN and sum the values in the list
def sum_princess_hp(princess_hp):
    # If it's NaN, return 0; otherwise, sum the values in the list
    if isinstance(princess_hp, list):
        return sum([float(i) for i in princess_hp])  # sum the values in the list
    return 0  # Return 0 for NaN or other invalid cases

#Replace NaN by 0 HP
df['winner.kingTowerHitPoints'] = df['winner.kingTowerHitPoints'].fillna(0)
df['loser.kingTowerHitPoints'] = df['loser.kingTowerHitPoints'].fillna(0)
df['winner.princessTowersHitPoints'] = df['winner.princessTowersHitPoints'].fillna(0)
df['loser.princessTowersHitPoints'] = df['loser.princessTowersHitPoints'].fillna(0)

df['winner.hp'] = df.apply(lambda row: float(row['winner.kingTowerHitPoints']) + sum_princess_hp(row['winner.princessTowersHitPoints']), axis=1)
df['loser.hp'] = df.apply(lambda row: float(row['loser.kingTowerHitPoints']) + sum_princess_hp(row['loser.princessTowersHitPoints']), axis=1)
df['hp.difference'] = df['winner.hp'] - df['loser.hp'] #Float, can be negative if loser finish with more HP
df['crown.difference'] = df['winner.crowns'] - df['loser.crowns'] #Float

hp_mean = df['hp.difference'].mean()
hp_sd = df['hp.difference'].std()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def win_margin(df, crown_dif, hp_dif, w=0.7):
    # Normalize crown difference using min-max
    norm_crown_dif = (crown_dif - 1) / 2  # Can be 0, 0.5, or 1
    # Normalize hp difference using z-score
    norm_hp_dif = sigmoid((hp_dif - hp_mean) / hp_sd)  # Normal distribution around 0
    #Define sigmoid function
    return w*norm_crown_dif + (1-w)*norm_hp_dif  # This is an example; adjust logic as needed.

df['win.margin'] = df.apply(lambda row: win_margin(df, row['crown.difference'], row['hp.difference']), axis=1) #Float

# Example output check
print(df[['winner.hp', 'loser.hp', 'hp.difference', 'crown.difference', 'win.margin']].sample(n=30))

df.to_csv('../data/S18/win_margin_battlesStaging_12272020_WL_tagged.csv', index=False)
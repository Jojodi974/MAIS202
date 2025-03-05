import numpy as np
import pandas as pd
import csv

###Import data from different files and compile it into single df
tsv_filename = "./data/S18/BattlesStaging_12272020_WL_tagged/battlesStaging_12272020_WL_tagged.csv"
chunk_size = 500000

text_file_reader = pd.read_csv(tsv_filename, engine='python',encoding='utf-8-sig', quoting=csv.QUOTE_MINIMAL, chunksize = chunk_size)

dfList = []
counter = 0

for df in text_file_reader:
    dfList.append(df)
    counter= counter +1
    print("Max rows read: " + str(chunk_size * counter) )

df = pd.concat(dfList,sort=False)

print(df.shape)
print(df.columns)
print(df['winner.crowns'].sample(n=10))
print(len(np.unique(df[["winner.tag", "loser.tag"]].values))) #How many players

df[["winner.clan.tag", "loser.clan.tag"]] = df[["winner.clan.tag", "loser.clan.tag"]].astype(str)

print(len(np.unique(df[["winner.clan.tag", "loser.clan.tag"]].values)))#How many clans
import pandas as pd
import csv
import joblib

#Open the df of single file since it is very large (2,000,000 games)
def data_loader(file_path, chunk_size = 500000):

    text_file_reader = pd.read_csv(file_path, engine='python',encoding='utf-8-sig', quoting=csv.QUOTE_MINIMAL, chunksize = chunk_size)

    dfList = []
    counter = 0

    for df in text_file_reader:
        dfList.append(df)
        counter= counter +1
        print("Max rows read: " + str(chunk_size * counter) )

    df = pd.concat(dfList,sort=False)

    # Save the DataFrame to a joblib file
    joblib.dump(df, f'data/joblib/file_path')

file_path = "/Users/jojod/Desktop/MAIS202/data/S18/raw_data/battlesStaging_12272020_WL_tagged.csv"
data_loader(file_path)
# Later, load the DataFrame from the joblib file using:
df_loaded = joblib.load(f'../data/joblib/{file_path}')

print(df_loaded.head())  # Verifying the loaded data
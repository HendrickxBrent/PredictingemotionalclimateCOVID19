# -*- coding: utf-8 -*-
"""
Created on Sun Jul 9 15:49:58 2022

@author: Brent
"""

import pandas as pd
import glob
import os



# setting the path for joining multiple files
files = os.path.join("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\Psychologie\\thesis\\Twitter\\All2020\\2023\\april", "results*.csv")

# list of merged files returned
files = glob.glob(files)
print("Resultant CSV after joining all CSV files at a particular location...");
# joining files with concat and read_csv
df = pd.concat(map(pd.read_csv, files), ignore_index=True)
print(df)


os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\Psychologie\\thesis\\Twitter\\All2020\\2023\\april")


## clean data
#df2  = pd.DataFrame(df[['author.id', 'text','created_at']])

df2 = df
df.to_csv('resultsAprilset2.csv')




os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\Psychologie\\thesis\\Twitter\\All2020\\2023\\april")


aprilset1 = pd.read_csv(r'resultsAprilset1.csv')
aprilset2 = pd.read_csv(r'resultsAprilset2.csv')



os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\Psychologie\\thesis\\Twitter\\All2020\\2023\\april\\older data")

part1april = pd.read_csv(r'part1April.csv')

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 19:29:25 2022

@author: Brent
"""

#RoBERTa
import concurrent.futures
import time
import pandas as pd
import torch
from transformers import pipeline
import tensorflow as tf
import timeit
import dask.dataframe as dd
import os
import numpy as np



os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\Psychologie\\thesis\\Twitter\\All2020\\2023\\april")

df1 = pd.read_csv(r'Aprilxrest.csv')
N = 6
# Drop first N columns of dataframe
df1 = df1.iloc[: , :N]
#df1.to_csv('part2April.csv')


#df2 = df1.head(50000)

## split dataset to train in chunks
one_df1 = 100000  #this is the estimated amount of data my computer can proces over night (9 hours)
two_df1 = 100001

A = 0

df1=df
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel1Aprilx.csv')  ## tot 
twoApril.to_csv('Aprilxrest.csv')

df1 = pd.read_csv(r'Aprilxrest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel2Aprilx.csv')  ## tot 
twoApril.to_csv('Aprilxrest.csv')

df1 = pd.read_csv(r'Aprilxrest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel3Aprilx.csv')  ## tot 
twoApril.to_csv('Aprilxrest.csv')

df1 = pd.read_csv(r'Aprilxrest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel4Aprilx.csv')  ## tot 
twoApril.to_csv('Aprilxrest.csv')


df1 = pd.read_csv(r'Aprilxrest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel5Aprilx.csv')  ## tot 
twoApril.to_csv('Aprilxrest.csv')

df1 = pd.read_csv(r'Aprilxrest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel6Aprilx.csv')  ## tot 
twoApril.to_csv('Aprilxrest.csv')

df1 = pd.read_csv(r'Aprilxrest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel7Aprilx.csv')  ## tot 
twoApril.to_csv('Aprilxrest.csv')

df1 = pd.read_csv(r'Aprilxrest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel8Aprilx.csv')  ## tot 
twoApril.to_csv('Aprilxrest.csv')


df1 = pd.read_csv(r'Aprilxrest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel9Aprilx.csv')  ## tot 
twoApril.to_csv('Aprilxrest.csv')

df1 = pd.read_csv(r'Aprilxrest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel10Aprilx.csv')  ## tot 
twoApril.to_csv('Aprilxrest.csv')


df1 = pd.read_csv(r'Aprilxrest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel11Aprilx.csv')  ## tot 
twoApril.to_csv('Aprilxrest.csv')

df1 = pd.read_csv(r'Aprilxrest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel12Aprilx.csv')  ## tot 
twoApril.to_csv('Aprilxrest.csv')


df1 = pd.read_csv(r'Aprilxrest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel13Aprilx.csv')  ## tot 
twoApril.to_csv('Aprilxrest.csv')

df1 = pd.read_csv(r'Aprilxrest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel14Aprilx.csv')  ## tot 
twoApril.to_csv('Aprilxrest.csv')


df1 = pd.read_csv(r'Aprilxrest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel15Aprilx.csv')  ## tot 
twoApril.to_csv('Aprilxrest.csv')

df1 = pd.read_csv(r'Aprilxrest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel16Aprilx.csv')  ## tot 
twoApril.to_csv('Aprilxrest.csv')

df1 = pd.read_csv(r'Aprilxrest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel17Aprilx.csv')  ## tot 
twoApril.to_csv('Aprilxrest.csv')

df1 = pd.read_csv(r'Aprilxrest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel18Aprilx.csv')  ## tot 
twoApril.to_csv('Aprilxrest.csv')

df1 = pd.read_csv(r'Aprilxrest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel19Aprilx.csv')  ## tot 
twoApril.to_csv('Aprilxrest.csv')

df1 = pd.read_csv(r'Aprilxrest.csv')
A = 1
oneApril = df1.iloc[:one_df1,A:]
twoApril = df1.iloc[two_df1:,A:]
oneApril.to_csv('deel20Aprilx.csv')  ## tot 
twoApril.to_csv('Aprilxrest.csv')


df17b = df17b.replace(np.nan,'a')




df7 = pd.read_csv(r'deel6Aprilx.csv')


df2 = part1April

df2 = df1

emotion_df.to_csv('resultpart1April.csv')



df1  = (pd.read_csv(r'deel1Aprilx.csv')).head(50000)
df1b = (pd.read_csv(r'deel1Aprilx.csv')).tail(50000)
df2  = (pd.read_csv(r'deel2Aprilx.csv')).head(50000)
df2b = (pd.read_csv(r'deel2Aprilx.csv')).tail(50000)

df3  = (pd.read_csv(r'deel3Aprilx.csv')).head(50000)
df3b = (pd.read_csv(r'deel3Aprilx.csv')).tail(50000)
df4 = (pd.read_csv(r'deel4Aprilx.csv')).head(50000)
df4b = (pd.read_csv(r'deel4Aprilx.csv')).tail(50000)

df5 = (pd.read_csv(r'deel5Aprilx.csv')).head(50000)
df5b = (pd.read_csv(r'deel5Aprilx.csv')).tail(50000)
df6 = (pd.read_csv(r'deel6Aprilx.csv')).head(50000)
df6b = (pd.read_csv(r'deel6Aprilx.csv')).tail(50000)

df7 = (pd.read_csv(r'deel7Aprilx.csv')).head(50000)
df7b = (pd.read_csv(r'deel7Aprilx.csv')).tail(50000)
df8 = (pd.read_csv(r'deel8Aprilx.csv')).head(50000)
df8b = (pd.read_csv(r'deel8Aprilx.csv')).tail(50000)

df9   = (pd.read_csv(r'deel9Aprilx.csv')).head(50000)
df9b  = (pd.read_csv(r'deel9Aprilx.csv')).tail(50000)
df10   = (pd.read_csv(r'deel10Aprilx.csv')).head(50000)
df10b = (pd.read_csv(r'deel10Aprilx.csv')).tail(50000)

df11   = (pd.read_csv(r'deel11Aprilx.csv')).head(50000)
df11b = (pd.read_csv(r'deel11Aprilx.csv')).tail(50000)
df12   = (pd.read_csv(r'deel12Aprilx.csv')).head(50000)
df12b = (pd.read_csv(r'deel12Aprilx.csv')).tail(50000)

df13   = (pd.read_csv(r'deel13Aprilx.csv')).head(50000)
df13b = (pd.read_csv(r'deel13Aprilx.csv')).tail(50000)
df14   = (pd.read_csv(r'deel14Aprilx.csv')).head(50000)
df14b = (pd.read_csv(r'deel14Aprilx.csv')).tail(50000)

df15   = (pd.read_csv(r'deel15Aprilx.csv')).head(50000)
df15b = (pd.read_csv(r'deel15Aprilx.csv')).tail(50000)
df16   = (pd.read_csv(r'deel16Aprilx.csv')).head(50000)
df16b = (pd.read_csv(r'deel16Aprilx.csv')).tail(50000)

df17   = (pd.read_csv(r'deel17Aprilx.csv')).head(50000)
df17b = (pd.read_csv(r'deel17Aprilx.csv')).tail(50000)
df18   = (pd.read_csv(r'deel18Aprilx.csv')).head(50000)
df18b = (pd.read_csv(r'deel18Aprilx.csv')).tail(50000)

df19   = (pd.read_csv(r'deel19Aprilx.csv')).head(50000)
df19b = (pd.read_csv(r'deel19Aprilx.csv')).tail(50000)
df20   = (pd.read_csv(r'deel20Aprilx.csv')).head(50000)
df20b = (pd.read_csv(r'deel20Aprilx.csv')).tail(50000)


df1b = df1b.reset_index()
df2b = df2b.reset_index()
df3b = df3b.reset_index()
df4b = df4b.reset_index()
df5b = df5b.reset_index()


df6b = df6b.reset_index()
df7b = df7b.reset_index()
df8b = df8b.reset_index()
df9b = df9b.reset_index()
df10b = df10b.reset_index()
df11b = df11b.reset_index()
df12b = df12b.reset_index()
df13b = df13b.reset_index()
df14b = df14b.reset_index()
df15b = df15b.reset_index()
df16b = df16b.reset_index()
df17b = df17b.reset_index()
df18b = df18b.reset_index()
df19b = df19b.reset_index()
df20b = df20b.reset_index()


df1 = df1.replace(np.nan,'a')
df1b = df1b.replace(np.nan,'a')
df2 = df2.replace(np.nan,'a')
df2b = df2b.replace(np.nan,'a')
df3 = df3.replace(np.nan,'a')
df3b = df3b.replace(np.nan,'a')

df4 = df4.replace(np.nan,'a')
df4b = df4b.replace(np.nan,'a')

df5 = df5.replace(np.nan,'a')
df5b = df5b.replace(np.nan,'a')
df6 = df6.replace(np.nan,'a')
df6b = df6b.replace(np.nan,'a')

df7 = df7.replace(np.nan,'a')
df7b = df7b.replace(np.nan,'a')
df8 = df8.replace(np.nan,'a')
df8b = df8b.replace(np.nan,'a')

df9 = df9.replace(np.nan,'a')
df9b = df9b.replace(np.nan,'a')
df10 = df10.replace(np.nan,'a')
df10b = df10b.replace(np.nan,'a')

df11 = df11.replace(np.nan,'a')
df11b = df11b.replace(np.nan,'a')
df12 = df12.replace(np.nan,'a')
df12b = df12b.replace(np.nan,'a')

df13 = df13.replace(np.nan,'a')
df13b = df13b.replace(np.nan,'a')
df14 = df14.replace(np.nan,'a')
df14b = df14b.replace(np.nan,'a')

df15 = df15.replace(np.nan,'a')
df15b = df15b.replace(np.nan,'a')
df16 = df16.replace(np.nan,'a')
df16b = df16b.replace(np.nan,'a')

df17 = df17.replace(np.nan,'a')
df17b = df17b.replace(np.nan,'a')
df18 = df18.replace(np.nan,'a')
df18b = df18b.replace(np.nan,'a')

df19 = df19.replace(np.nan,'a')
df19b = df19b.replace(np.nan,'a')
df20 = df20.replace(np.nan,'a')
df20b = df20b.replace(np.nan,'a')



N = 22
df20b = df20b.iloc[: , N:]
N = 21
df19b = df19b.iloc[: , N:]
N = 20
df18b = df18b.iloc[: , N:]
N = 19
df17b = df17b.iloc[: , N:]
N = 18
df16b = df16b.iloc[: , N:]
N = 17
df15b = df15b.iloc[: , N:]
N = 16
df14b = df14b.iloc[: , N:]
N = 15
df13b = df13b.iloc[: , N:]
N = 14
df12b = df12b.iloc[: , N:]
N = 13
df11b = df11b.iloc[: , N:]
N = 12
df10b = df10b.iloc[: , N:]
N = 11
df9b = df9b.iloc[: , N:]
N = 10
df8b = df8b.iloc[: , N:]
N = 9
df7b = df7b.iloc[: , N:]
N = 8
df6b = df6b.iloc[: , N:]


N = 22
df2 = df2.iloc[: , N:]
df1 = df1.iloc[: , N:]
df3 = df3.iloc[: , N:]
df4 = df4.iloc[: , N:]
df5 = df5.iloc[: , N:]
df6 = df6.iloc[: , N:]



####################################################################################################
###                                          threading
### (I7, 10th edition, 11 cores of the 12 cores on my laptop)
### speed benchmark: dataset 2000 rows -> 37.27 seconds  --> 53.66 rows/s
###                          4000 rows -> 73.08    --> 54.73 rows/s
###                          10000 rows-> 134.29   --> 74.46 rows/s
###                          50000 rows-> 816.85 s --> 61.21 rows/s  (10 cores instead of 11)
###                          100000 rows-> 22 min  --> 75.06 rows/s  (however, breaks down, only completes 80%)
################################################################################################################

#############################
### deel1
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df1.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df1.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df1.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1 = pd.concat([df1, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results1.to_csv('results1.csv')


#############################
### deel1b
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df1b.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df1b.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df1b[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df1b.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results1b = pd.concat([df1b, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results1b.to_csv('results1b.csv')


#############################
### deel2
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df2.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df2.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df2[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results2 = pd.concat([df2, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results2.to_csv('results2.csv')




#############################
### deel2b
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df2b.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df2b.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df2b[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2b.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results2b = pd.concat([df2b, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results2b.to_csv('results2b.csv')



#############################
### deel 3
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df3.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df3.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df3[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df3.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results3 = pd.concat([df3, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results3.to_csv('results3.csv')



#############################
### deel 3b
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df3b.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df3b.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df3b[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df3b.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results3b = pd.concat([df3b, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results3b.to_csv('results3b.csv')



#############################
### deel 4
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df4.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df4.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df4[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df4.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results4 = pd.concat([df4, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results4.to_csv('results4.csv')


#############################
### deel 4b
#############################    ERROR: execution aborted
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df4b.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df4b.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df4b[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df4b.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results4b = pd.concat([df4b, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results4b.to_csv('results4b.csv')



#############################
### deel 5
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df5.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df5.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df5[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df5.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results5 = pd.concat([df5, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results5.to_csv('results5.csv')


#############################
### deel 5b
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df5b.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df5b.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df5b[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df5b.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results5b = pd.concat([df5b, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results5b.to_csv('results5b.csv')


#############################
### deel 6
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df6.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df6.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df6[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df6.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results6 = pd.concat([df6, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results6.to_csv('results6.csv')


#############################
### deel 6b
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df6b.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df6b.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df6b[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df6b.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results6b = pd.concat([df6b, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results6b.to_csv('results6b.csv')


#############################
### deel 7
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df7.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df7.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df7[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df7.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results7 = pd.concat([df7, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results7.to_csv('results7.csv')


#############################
### deel 7b
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df7b.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df7b.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df7b[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df7b.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results7b = pd.concat([df7b, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results7b.to_csv('results7b.csv')


#############################
### deel 8
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df8.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df8.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df8[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df8.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results8 = pd.concat([df8, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results8.to_csv('results8.csv')


#############################
### deel 8b
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df8b.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df8b.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df8b[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df8b.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results8b = pd.concat([df8b, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results8b.to_csv('results8b.csv')


#############################
### deel 9
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df9.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df9.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df9[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df9.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results9 = pd.concat([df9, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results9.to_csv('results9.csv')


#############################
### deel 9b
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df9b.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df9b.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df9b[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df9b.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results9b = pd.concat([df9b, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results9b.to_csv('results9b.csv')


#############################
### deel 10
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df10.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df10.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df10[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df10.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results10 = pd.concat([df10, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results10.to_csv('results10.csv')


#############################       --> same error as in 7
### deel 10b
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df10b.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df10b.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df10b[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df10b.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results10b = pd.concat([df10b, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results10b.to_csv('results10b.csv')


#############################
### deel 11
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df11.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df11.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df11[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df11.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results11 = pd.concat([df11, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results11.to_csv('results11.csv')


#############################
### deel 11b
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df11b.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df11b.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df11b[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df11b.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results11b = pd.concat([df11b, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results11b.to_csv('results11b.csv')


#############################
### deel 12
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df12.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df12.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df12[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df12.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results12 = pd.concat([df12, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results12.to_csv('results12.csv')


#############################
### deel 12b
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df12b.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df12b.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df12b[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df12b.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results12b = pd.concat([df12b, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results12b.to_csv('results12b.csv')


#############################
### deel 13
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df13.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df13.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df13[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df13.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results13 = pd.concat([df13, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results13.to_csv('results13.csv')


#############################
### deel 13b
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df13b.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df13b.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df13b[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df13b.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results13b = pd.concat([df13b, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results13b.to_csv('results13b.csv')


#############################
### deel 14
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df14.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df14.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df14[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df14.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results14 = pd.concat([df14, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results14.to_csv('results14.csv')


#############################
### deel 14b
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df14b.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df14b.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df14b[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df14b.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results14b = pd.concat([df14b, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results14b.to_csv('results14b.csv')


#############################
### deel 15
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df15.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df15.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df15[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df15.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results15 = pd.concat([df15, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results15.to_csv('results15.csv')


#############################
### deel 15b
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df15b.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df15b.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df15b[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df15b.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results15b = pd.concat([df15b, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results15b.to_csv('results15b.csv')


#############################
### deel 16
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df16.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df16.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df16[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df16.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results16 = pd.concat([df16, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results16.to_csv('results16.csv')


#############################
### deel 16b
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df16b.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df16b.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df16b[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df16b.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results16b = pd.concat([df16b, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results16b.to_csv('results16b.csv')

#############################
### deel 17
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df17.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df17.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df17[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df17.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results17 = pd.concat([df17, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results17.to_csv('results17.csv')


#############################
### deel 17b
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df17b.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df17b.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df17b[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df17b.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results17b = pd.concat([df17b, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results17b.to_csv('results17b.csv')


#############################
### deel 18
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df18.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df18.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df18[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df18.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results18 = pd.concat([df18, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results18.to_csv('results18.csv')


#############################
### deel 18b
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df18b.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df18b.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df18b[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df18b.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results18b = pd.concat([df18b, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results18b.to_csv('results18b.csv')


#############################
### deel 19
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df19.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df19.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df19[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df19.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results19 = pd.concat([df19, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results19.to_csv('results19.csv')


#############################
### deel 19b
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df19b.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df19b.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df19b[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df19b.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results19b = pd.concat([df19b, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results19b.to_csv('results19b.csv')


#############################
### deel 20
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df20.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df20.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df20[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df20.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results20 = pd.concat([df20, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results20.to_csv('results20.csv')


#############################
### deel 20b
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df20b.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df20b.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df20b[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df20b.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    results20b = pd.concat([df20b, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results20b.to_csv('results20b.csv')














#############################
### deel 
#############################
###                                          threading
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df2.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]
emotion_df = pd.DataFrame(index=df2.index, columns=labels)
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]
start = time.time()
# Initialize a ThreadPoolExecutor with 4 worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df2[['Tweet']].itertuples(index=True)
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)
    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
    result2 = pd.concat([df2, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")

results.to_csv('results.csv')




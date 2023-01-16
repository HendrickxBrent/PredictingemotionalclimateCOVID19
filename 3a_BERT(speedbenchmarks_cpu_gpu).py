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



os.chdir("C:\\Users\\Brent\\Documents\\Psychology\\KUL\\Psychologie\\thesis\\Twitter\\All2020\\2023\\april")

df1 = pd.read_csv(r'April2020CleanNew.csv')
N = 3
# Drop first N columns of dataframe
df1 = df1.iloc[: , N:]
df1.to_csv('April2020CleanNew.csv')


df2 = df1.head(10000)

## split dataset to train in chunks
part1_df1 = 1800000  #this is the estimated amount of data my computer can proces over night (9 hours)
part2_df1 = 1800001
part1April = df1.iloc[:part1_df1,]
part2April = df1.iloc[part2_df1:,]

part1April.to_csv('part1April.csv')
part2April.to_csv('part2April.csv')

df2 = part1April


####################################################################################################
###                         withhout gpu and without multiprocessing
###
### speed benchmark: dataset 2000 rows -> 73 seconds
###                          4000 rows -> 127.21
####################################################################################################
classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df2.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]

emotion_df = pd.DataFrame(index=df2.index, columns=labels)

start = timeit.default_timer()
for i, row in df2.iterrows():
    # Apply the classifier function to the Tweet in the 'Tweet' column
    result = classifier(df2.at[i, 'Tweet'])
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    # Loop through each emotion in the dictionary
    for emotion in result[0]:
        # If the emotion label exists in the new dataframe,
        # set the value for that emotion in the current row
        # to the corresponding score from the dictionary
        if emotion['label'] in emotion_df.columns:
            emotion_df.at[i, emotion['label']] = emotion['score']
result2 = pd.concat([df2, emotion_df], axis=1)
end = timeit.default_timer()
print("Time taken:", end - start)






####################################################################################################
###                                          threading
###
### speed benchmark: dataset 2000 rows -> 37.27 seconds (I7, 10th edition, 11 cores)
###                          4000 rows -> 73.08
###                          10000 rows-> 220.31
####################################################################################################



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
with concurrent.futures.ThreadPoolExecutor(max_workers=11) as executor:
    # Get an iterator of (index, tweet) tuples from the dataframe
    tweet_iter = df2[['Tweet']].itertuples(index=True)

    # Map the classify_emotion function to the iterator
    # using the executor's map method
    
    results = executor.map(classify_emotion, (x[1] for x in tweet_iter))
    

    # Initialize the emotion dataframe with the same shape as the original dataframe
    emotion_df = pd.DataFrame(index=df2.index, columns=labels)

    # Loop through the results and populate the emotion dataframe
    for i, (labels, emotions) in enumerate(results):
        for emotion in emotions:
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']

    # Concatenate the original dataframe with the emotion dataframe
    result2 = pd.concat([df2, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")




####################################################################################################
###                                  Parallel processing (multiple CPU) DASK
###
### speed benchmark: dataset 2000 rows -> 39.63 (I7, 10th edition, 11 cores)
###                          4000 rows -> 74.22
###                          10000
####################################################################################################
####################################################################################################
## DASK   this is 28.9 seconds (2000)   for 4000 = 57.63


start = time.time()
def classify_emotion(tweet):
    # Apply the classifier function to the given tweet text
    result = classifier(tweet)
    # Get the labels (emotions) from the result dictionary
    labels = [entry['label'] for entry in result[0]]
    return labels, result[0]

# Convert the Pandas dataframe to a Dask dataframe
df = dd.from_pandas(df2, npartitions=11)

# Use the map method of the Dask dataframe to apply the classify_emotion function to each tweet
results = df['Tweet'].map(classify_emotion)

# Compute the result as a Pandas dataframe
results = results.compute()

# Initialize the emotion dataframe with the same shape as the original dataframe
emotion_df = pd.DataFrame(index=df2.index, columns=labels)

# Loop through the results and populate the emotion dataframe
for i, (labels, emotions) in enumerate(results):
    for emotion in emotions:
        if emotion['label'] in emotion_df.columns:
            emotion_df.at[i, emotion['label']] = emotion['score']

# Concatenate the original dataframe with the emotion dataframe
result2 = pd.concat([df2, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")




####################################################################################################
###                                          DASK + threading
###
### speed benchmark: dataset 2000 rows ->  37.60   (I7, 10th edition, 2*11 cores) 
###                          2000 rows ->  38.78   (I7, 10th edition, 5+6 cores) 
###                          4000 rows ->  79.45   (I7, 10th edition, 11 cores)
####################################################################################################
####################################################################################################
## DASK + threathing   this is 28.9 seconds+ multiple threathing   =  28.5 (2000)    for 4000 = 55.57   (using cpu 11 cores and intel 10th gen i7)


import concurrent.futures

start = time.time()
def classify_emotion(tweet):
    # Initialize a ThreadPoolExecutor with 4 worker threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Apply the classifier function to the given tweet text
        result = classifier(tweet)
        # Get the labels (emotions) from the result dictionary
        labels = [entry['label'] for entry in result[0]]
        return labels, result[0]

# Convert the Pandas dataframe to a Dask dataframe
df = dd.from_pandas(df2, npartitions=6)

# Use the map method of the Dask dataframe to apply the classify_emotion function to each tweet
results = df['Tweet'].map(classify_emotion)

# Compute the result as a Pandas dataframe
results = results.compute()

# Initialize the emotion dataframe with the same shape as the original dataframe
emotion_df = pd.DataFrame(index=df2.index, columns=labels)

# Loop through the results and populate the emotion dataframe
for i, (labels, emotions) in enumerate(results):
    for emotion in emotions:
        if emotion['label'] in emotion_df.columns:
            emotion_df.at[i, emotion['label']] = emotion['score']

# Concatenate the original dataframe with the emotion dataframe
result2 = pd.concat([df2, emotion_df], axis=1)
end = time.time()
print(f"Time taken: {end - start} seconds")




















############################### code############################### code    GPU acceleration test   (didn't work)
############################### code############################### code


print(torch.backends.cudnn.enabled)
print(torch.cuda.is_available())
# Check if the GPU is available
if tf.test.is_gpu_available():
    # Set the device to GPU
    device = '/device:GPU:0'
    print("Running on the GPU")
else:
    # Set the device to CPU
    device = '/device:CPU:0'
    print("Running on the CPU")

tf.config.list_physical_devices('GPU')

classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=None)
# Get the labels (emotions) from the classifier's output dictionary
# for the first row of the dataframe
result = classifier(df2.at[0, 'Tweet'])
labels = [entry['label'] for entry in result[0]]

emotion_df = pd.DataFrame(index=df2.index, columns=labels)

##### with gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start = timeit.default_timer()
with tf.device('/GPU:0' if tf.test.is_gpu_available() else '/CPU:0'):
    for i, row in df2.iterrows():
        # Apply the classifier function to the text in the 'Tweet' column
        result = classifier(df2.at[i, 'Tweet'])
        # Get the labels (emotions) from the result dictionary
        labels = [entry['label'] for entry in result[0]]
        # Loop through each emotion in the dictionary
        for emotion in result[0]:
            # If the emotion label exists in the new dataframe,
            # set the value for that emotion in the current row
            # to the corresponding score from the dictionary
            if emotion['label'] in emotion_df.columns:
                emotion_df.at[i, emotion['label']] = emotion['score']
result2 = pd.concat([df2, emotion_df], axis=1)
end = timeit.default_timer()
print("Time taken:", end - start)


import pandas as pd
import json
import os
import logging
import datetime
import time
import csv
from Local_LLM import llm_request


def llm_stance(text):
    prompt = """Analyze the following tweet, generate the target for this tweet, and determine its stance towards the generated target.
    The target follows the following rules:
    1. A target should be the topic on which the tweet is talking or debating.
    2. The target can be entities such as people, event and organization, as well as subjective ideas such as claim, but it must be less than 4 words in length. 
    3. The target must be meaningful and avoid emotionless words such as "you", "me", "he", "friendship", "attack", etc.
    4. The same meaning is expressed in the same way, for example, 'ask an atheist day' and 'askanatheistday' are both expressed in the first way.
    5. As concise as possible, for example, 'belief in god' is represented by 'god'.
    If the stance is in favor of the target, write FAVOR, if it is against the target write AGAINST and if it is ambiguous, write NONE. The answer only has to be one of these three words: FAVOR, AGAINST, or NONE. Do not provide any explanation.
    The target stance pair can be more than one.
    The output format should be: “<target1>: <stance1>, <target1>: <stance1>“."""

    messages = [{'role': 'system', 'content': prompt},
        {'role': 'user', 'content': tweet}]

    response = llm_request(model="llama-3.1-70b-awq", messages=messages)
    return response

if __name__ == "__main__":
    data=pd.read_excel("data/sem16_dataset_ts.xlsx")
    data_path= "data/sem16_dataset_ts.xlsx"

    data['llm_stance'] = None
    for index, row in data.iterrows():
        tweet = row['Tweet']
        print(tweet)
        if row['llm_stance']==None or pd.isna(row['llm_stance']):
            if "SemST" in tweet:
                tweet = tweet[:-6]#del #SemST
            stance = llm_stance(tweet)[0]
            print(index,stance)
            data.at[index, 'llm_stance'] = stance
            if index%50==0:
                data.to_excel(data_path)
    data.to_excel(data_path)
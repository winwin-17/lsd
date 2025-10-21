import pandas as pd
import json
import os
import logging
import datetime
import time
import csv
from Local_LLM import llm_request
# from mistral_LLM import llm_request
import time,json
from typing import List, Dict

import requests
import openai

def llm_request(model: str,messages: List[Dict[str, str]],temperature: float = 0.5,url: str = None,verify: bool = False,timeout: int = None,stream: bool = False):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer EMPTY'
    }
    if url is not None:
        url = url + "/v1/chat/completions"
    else:
        url = "http://localhost:50003/v1/chat/completions"  #
    payload_dict = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "temperature": temperature,
    }
    payload = json.dumps(payload_dict, ensure_ascii=False)
    result_payload = requests.post(url=url,
                                   data=payload,
                                   headers=headers,
                                   verify=verify,
                                   timeout=timeout)
    response_dict = json.loads(result_payload.text)
    #print(response_dict)
    if 'choices' not in response_dict.keys():
        return f"Error Response: {response_dict}"
    return response_dict['choices'][0]['message']['content'], response_dict['usage']['total_tokens']


def llm_response(model: str, system_prompt: str, query: str):
    messages = [{'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': query}]
    response, token_cnt = llm_request(model=model, messages=messages)
    return response, token_cnt
from tqdm import tqdm

def llm_stance(text,target):
    prompt = f"What is the attitude of the sentence: '{text}' to the target: '{target}'. select from favor, against or neutral. keep your answer 1 word long. please double-check your answer before responding and be sure about it."
    ll = "Note: Stance detection is different from sentiment analysis. Stance detection focuses on the author's support, opposition, or neutral stance towards a specific target, while sentiment analysis focuses more on identifying the positive, negative, or neutral emotional tendencies of the text as a whole or specific parts. Sentiments are positive, but the  stance to the target may be aggainst. Their labels are different."
    prompt = prompt + ll
    messages = [{'role': 'system', 'content': prompt},
        {'role': 'user', 'content': tweet}]

    response,_ = llm_request(model="llama-3.1-70b-awq", messages=messages)#"mistral-large-latest"llama-3.1-70b-awq
    return response
import numpy as np
def metric(stance,predict):
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
    stance_id,predict_id=[],[]
    for i in range(len(stance)):
        if "favor" in stance[i].lower():
            stance_id.append(0)
        elif "against" in stance[i].lower():
            stance_id.append(1)
        else:
            stance_id.append(2)
        if "favor" in predict[i].lower():
            predict_id.append(0)
        elif "against" in predict[i].lower():
            predict_id.append(1)
        else:
            predict_id.append(2)

    predicted_labels = np.array(predict_id)
    all_labels = np.array(stance_id)
    metrics = dict()
    metrics["f1macro"] = f1_score(all_labels, predicted_labels, average='macro')
    metrics["f1micro"] = f1_score(all_labels, predicted_labels, average='micro')
    metrics["f1_1"] = f1_score(all_labels, predicted_labels, average=None)
    metrics["f1average"] = (metrics["f1_1"][0] + metrics["f1_1"][1]) / 2
    metrics["precision"] = precision_score(all_labels, predicted_labels, average='macro')
    metrics["recall"] = recall_score(all_labels, predicted_labels, average='macro')
    return metrics
if __name__ == "__main__":
    data=pd.read_csv("data/sem16_dataset.csv")

    column_name="llama370b1213"
    data[column_name] = None
    for index, row in tqdm(data.iterrows()):
        tweet = row['Tweet']
        target =row['Target']

        if (row[column_name]==None or pd.isna(row[column_name])) and (row["set"]=="test"):
            stance = llm_stance(tweet,target)
            print(stance)
            data.at[index, column_name] = stance
            if index%100==0:
                data.to_excel(data_path)
    data.to_excel(data_path)

    data = data[data["set"] == "test"]
    metric(data["Stance"].values.tolist(), data[column_name].values.tolist())
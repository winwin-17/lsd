import pandas as pd
import json
import os
from tqdm import tqdm
import argparse
import re
import sys
# from Local_LLM import llm_response
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
import numpy as np
from collections import deque
import wordninja

class Reasoning_Graph_Baseline:
    def __init__(self, args):
        self.args = args
        self.prompt_path = args.prompt_path
        self.data_path = args.data_path
        self.target_name = args.target_name
        self.split = args.split
        self.model_name = args.model_name
        self.k=args.k
        self.save_path = args.save_path#os.path.join(args.save_path,f"{self.target_name.lower().replace(' ','')}_{self.split}_{self.model_name}.csv")#results_k1/Atheism_test_llama-3.1-70b-awq.json

        self.premise_data = pd.read_csv(
            f"reasoning_premise_{args.dataset}/{self.target_name.lower().replace(' ', '')}_premise.csv")  # ,index_col='index')
        self.sys_prompt = "You are a helpful assistant. Make sure you carefully and fully understand the details of user's requirements before you start solving the problem."

        # load raw dataset
        self.raw_dataset = self.load_raw_dataset()
        print(f"Loaded {len(self.raw_dataset)} examples from {self.target_name}.")

        self.adj_graph=pd.read_csv(f"target_graph_{args.dataset}/{self.target_name.lower().replace(' ', '')}_relation_matrix.csv")
        # 删除所有包含 'Unnamed' 的列
        self.adj_graph = self.adj_graph.loc[:, ~self.adj_graph.columns.str.contains('^Unnamed')]
        self.targets = self.adj_graph.columns.tolist()

    def bfs_multiple_paths(self,start, end, max_hops=2):

        n = len(self.adj_graph)
        start=self.targets.index(start)
        end=self.targets.index(end)
        queue = deque([(start, [start], 0, [])])  # (current node, path, path length)
        paths = []
        path_values = []
        while queue:
            node, path, path_length, path_value = queue.popleft()
            if node == end and path_length <= max_hops:
                paths.append(path)
                path_values.append(path_value)
                if len(paths) >= max_hops:
                    break
            if path_length < max_hops:
                for neighbor in range(n):
                    if self.adj_graph.iloc[node,neighbor] != 0 and neighbor not in path:
                        new_path = path + [neighbor]
                        new_path_value = path_value + [self.adj_graph.iloc[node,neighbor]]
                        queue.append((neighbor, new_path, path_length + 1, new_path_value))

        return paths, path_values

    def load_raw_dataset(self):
        raw_dataset=pd.read_excel(self.data_path)#,index_col='index')
        print(len(raw_dataset))

        if len(self.target_name)>0:
            raw_dataset=raw_dataset[raw_dataset["Target"]==self.target_name]
        return raw_dataset

    def post_process_c(self, response_c):
        pattern_bracket = r"Final answer: \{([^}]*)\}"#Final answer: {false}
        match = re.search(pattern_bracket, response_c)
        if match:
            answers = match.group(1).lower()
            return answers
        pattern_direct = r"Final answer: \[([^}]*)\]"
        match = re.search(pattern_direct, response_c, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        return "No final answer found in the text."

    def final_process(self, final_answer):
        final_answer = final_answer.lower()
        if final_answer == "true":
            final_answer = 'favor'
        elif final_answer == "false":
            final_answer = 'against'
        elif final_answer == "unknown":
            final_answer = 'none'
        else:
            final_answer = "No final answer found in the text."
        return final_answer

    def load_in_context_examples_trans(self):
        file_path = os.path.join(self.prompt_path, 'translation_example.txt')
        with open(file_path,encoding="utf-8") as f:
            in_context_examples = f.read()
        return in_context_examples
    def construct_prompt_a(self, context,question, in_context_examples_trans):
        full_prompt = in_context_examples_trans
        full_prompt = full_prompt.replace('[[CONTEXT]]', context)
        full_prompt = full_prompt.replace('[[QUESTION]]', question)
        return full_prompt

    def reasoning_graph_generation(self):
        if len(self.split)>0:
            raw_dataset=self.raw_dataset[self.raw_dataset["type"]==self.split]
            print(f"Loaded {len(raw_dataset)} examples from {self.split} split and {self.target_name}.")

        # load in-context examples
        in_context_examples_trans = self.load_in_context_examples_trans()

        try:
            outputs=pd.read_csv(os.path.join(self.save_path,f"{self.target_name.lower().replace(' ','')}_{self.split}.csv"))
            outputs = outputs.loc[:, ~outputs.columns.str.contains('^Unnamed')]

            tweets = outputs["Tweet"].values.tolist()
        except:
            outputs = pd.DataFrame()
            tweets=[]

        for i,example in tqdm(raw_dataset.iterrows()):
            question_format = "Based on the above information, is the following statement true, false, or unknown? The tweet supports {target}."
            context_format = "The tweet is \"{tweet}\". {sub_context}"
            sub_context = ""

            tweet = example["Tweet"]
            if tweet in tweets:
                continue
            target = example["Target"]
            llm_stance= example["llm_stance"]


            if pd.isna(llm_stance) or llm_stance=="None" or ":" not in llm_stance:
                sub_context=""
            else:
                i = llm_stance.replace('"', '')
                i = i.replace("'", '')
                s = i.split(",")
                stances=[]
                for ss in s:
                    relations = []
                    zi = ss.split(":")[0].lower().strip()
                    # zi=" ".join(wordninja.split(zi))
                    zi_stance=ss.split(":")[1].lower().strip()
                    if zi_stance=="favor":
                        sub_context1=f"The tweet supports {zi}."
                    elif zi_stance=="against":
                        sub_context1=f"The tweet doesn't support {zi}."
                    else:
                        continue
                    sub_context += sub_context1
                    sub_context += " "
                    paths,path_values=self.bfs_multiple_paths(zi, self.target_name.lower(), max_hops=self.k)
                    for path in paths:
                        for i in range(len(path)-1):
                            target1=self.targets[path[i]]
                            target2 = self.targets[path[i+1]]
                            row = self.premise_data[
                                (self.premise_data["target1"] == target1) & (self.premise_data["target2"] == target2) & (
                                        self.premise_data["target"] == self.target_name)]
                            if row.empty:
                                continue
                            realtion = row['relation'].values[0]
                            sub_context += realtion
                            sub_context += " "
            context = context_format.format(tweet=tweet, sub_context=sub_context)
            question = question_format.format(target=target)
            # print(context)

            # print("Translating...")
            prompts_a = self.construct_prompt_a(context, question, in_context_examples_trans)
            responses_a, _ = llm_response(model=self.model_name, system_prompt=self.sys_prompt, query=prompts_a)
            # print(responses_a)

            final_answer = self.post_process_c(responses_a)
            # print(final_answer)
            final_stance = self.final_process(final_answer)
            print(final_stance)
            #
            output1 = {'question':question,
                       "context":context,
                       "execution":responses_a,
                       'predicted_stance': final_stance}

            output = example.to_dict().copy()
            output.update(output1)
            new_df = pd.DataFrame([output])
            outputs = pd.concat([outputs, new_df], ignore_index=True)

            outputs.to_csv(os.path.join(self.save_path,f"{self.target_name.lower().replace(' ','')}_{self.split}.csv"),encoding="utf-8")
        stance=outputs["Stance"]
        predict=outputs["predicted_stance"]
        metrics=metric(stance,predict)
        print(metrics)
        # return outputs

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
    # print(stance_id)
    # print(predict_id)
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
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/sem16_dataset_ts.xlsx')
    parser.add_argument('--prompt_path', type=str, default='prompts')
    parser.add_argument('--dataset', type=str,default="sem16")#Atheism")#Hillary Clinton")Legalization of Abortion  Atheism  Climate Change is a Real Concern  Feminist Movement  Donald Trump
    parser.add_argument('--split', type=str,default="test")
    parser.add_argument('--save_path', type=str, default='results_graph')#results_k4
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--demonstration_path', type=str, default='icl_examples')

    parser.add_argument('--model_name', type=str, default="llama-3.1-70b-awq")
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--mode', type=str)
    parser.add_argument('--max_new_tokens', type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    targets = ["Atheism", "Climate Change is a Real Concern", "Feminist Movement", "Hillary Clinton",
               "Legalization of Abortion",
               "Donald Trump", "Joe Biden", "Bernie Sanders", "abortion", "cloning", "death penalty", "gun control",
               "marijuana legalization", "minimum wage", "nuclear energy", "school uniforms", "face masks", "fauci",
               "school closures", "stay at home orders"]
    targets=["Feminist Movement"]
    for target in targets:
        args.target_name=target
        print(args)
        problem_reduction = Reasoning_Graph_Baseline(args)
        problem_reduction.reasoning_graph_generation()
    # gpt3_problem_reduction.batch_reasoning_graph_generation(batch_size=1)






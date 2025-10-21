import pandas as pd
import os,csv,requests,time,json
from tqdm import tqdm
# from Local_LLM import llm_response
from typing import List, Dict

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

def get_targets(raw_dataset,target_name):
    llm_stance = raw_dataset["llm_stance"].values.tolist()
    zi_targets = []
    stances = []
    for index, i in enumerate(llm_stance):
        print(i)
        if pd.isna(i) or i.lower() == "None" or ":" not in i:
            continue
        i = i.replace('"', '')
        i = i.replace("'", '')
        s = i.split(",")
        for ss in s:
            zi = ss.split(":")[0].lower().strip()
            stance = ss.split(":")[1].lower().strip()
            if (stance == "favor" or stance == "against"):
                zi_targets.append(zi)
                stances.append(stance)
    zi_targets.append(target_name.lower())
    # 词语边界恢复
    zi_targets = [" ".join(wordninja.split(target)) for target in zi_targets]
    zi_targets = sorted(list(set(zi_targets)))
    # print(zi_targets)
    # print(len(zi_targets))
    return zi_targets

def get_premise(targets,premise_path,threshold = 0.5,model_path="w2vec/all.model"):
    with open("prompts/premise_generation.txt", encoding="utf-8") as f:
        prompt = f.read()
    if os.path.exists(premise_path):
        df = pd.read_csv(premise_path)
    else:
        df = pd.DataFrame({"target1": [], "target2": [], "relation": [], "target": []})
    # 删除所有包含 'Unnamed' 的列
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    from gensim.models import Word2Vec
    model = Word2Vec.load(model_path)
    for i, z1 in tqdm(enumerate(targets)):
        for j, z2 in enumerate(targets):
            if z1 in model.wv and z2 in model.wv:
                similarity = model.wv.similarity(z1, z2)
            else:
                continue
            if similarity < threshold:
                continue
            else:
                row = df[(df["target1"] == z1) & (df["target2"] == z2) & (df["target"] == target)]
                if not row.empty:
                    continue
                p = prompt.format(target1=z1, target2=z2)
                response, token_cnt = llm_response(model="llama-3.1-70b-awq",
                                                   system_prompt="you are a professional linguist",
                                                   query=p)
                # print(response)
                new_row = {"target1": z1, "target2": z2, "relation": response, "target": target}
                print(new_row)
                # 将新行转换为DataFrame
                new_row_df = pd.DataFrame([new_row])
                # 使用pd.concat()合并DataFrame
                df = pd.concat([df, new_row_df], ignore_index=True)
                # df.append({"target1":zi,"target2":target,"relation":response},ignore_index=True)

                if j % 50 == 0:
                    df.to_csv(premise_path, quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8")
    # 去除所有完全相同的行
    df_unique = df.drop_duplicates()
    df_unique.to_csv(premise_path, quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8")
    return df_unique
def get_targetgraph(premise_data,targets,target_name,save_path):
    l = len(targets)
    df = pd.DataFrame(columns=targets)
    for index1, target1 in tqdm(enumerate(targets)):
        row_data = [0] * l
        for index2, target2 in enumerate(targets):
            row = premise_data[
                (premise_data["target1"] == target1) & (premise_data["target2"] == target2) & (
                        premise_data["target"] == target_name)]
            # print(target1,target2,self.target_name,row)
            if row.empty:
                continue
            realtion = row['relation'].values[0]

            if ("oppose" in realtion or "against" in realtion or "support" in realtion or "favor" in realtion) and (
                    "may or may not" not in realtion):
                row_data[index2] = 1
            else:
                row_data[index2] = 0
        df.loc[index1] = row_data
    print(df)
    df.to_csv(save_path)  # 保存为 CSV 文件
    return df

dataset="sem16"
data=pd.read_excel(f"data/tse_sem16_{dataset}_ts.xlsx")
targets = ["Atheism","Climate Change is a Real Concern", "Feminist Movement", "Hillary Clinton", "Legalization of Abortion",
               "Donald Trump","Joe Biden","Bernie Sanders","abortion", "cloning", "death penalty", "gun control", "marijuana legalization", "minimum wage","nuclear energy", "school uniforms","face masks","fauci", "school closures", "stay at home orders" ]

target=targets[0]
if len(target)>0:
    raw_dataset=data[data["Target"]==target]
premise_path=f"reasoning_premise_{dataset}/{target.lower().replace(' ','')}_premise2.csv"
targetgraph_path=f"target_graph_{dataset}/{target.lower().replace(' ', '')}_relation_matrix2.csv"#"target_graph_am"
targets=get_targets(raw_dataset,target)
premise_data=get_premise(targets,premise_path)
l=len(targets)
if os.path.exists(targetgraph_path):
    df = pd.read_csv(targetgraph_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    if df.shape[0] != l:
        target_graph=get_targetgraph(premise_data,targets,target,targetgraph_path)
else:
    target_graph=get_targetgraph(premise_data,targets,target,targetgraph_path)


import os
import json

from exceptiongroup import catch
from fontTools.ttLib.tables.otData import otData
# from triton.compiler import ptx_get_kernel_name

from model.ExperModel import Expert

# from transformers.models.pop2piano.convert_pop2piano_weights_to_hf import model
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
# # from utils import cal_anchor_embedding
# # from tqdm import tqdm
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# # tokenizer.pad_token = tokenizer.eos_token
# # model = AutoModelForCausalLM.from_pretrained(model_name, device_map='balanced')
# # for param in model.parameters():
# #     param.requires_grad = False
# #
# # sentence = "Question:Which city is the caption of China?\nAnswer:"
sentence = "Question"
# # 输入句子并调用模型
input_ids = tokenizer(sentence, return_tensors="pt")
#
# tok_label = tokenizer.tokenize(sentence)
print(input_ids)
# tok_label = tokenizer.convert_tokens_to_ids(tok_label)
# print(tok_label)
# tok_label = [1, 19320,  5485, 29901,    13,  8256, 23032,  3512,  1250,   297,
#            931,   304,  5040,   278,   830,  3901, 29899,  8754,  1161,   515,
#          23393,   405,  2207, 29889,  2860,   498,  1450,   484,   338,  9445,
#            491,   445, 16832, 29915, 29879, 12579,  1171,   411,   385, 16631,
#            713, 22378, 29892,   278, 21967,  9850, 29879,  1250,   297,   931,
#            304,  5040, 23032, 29915, 29879, 20023,  1583,   515, 10551,   292,
#           4955,   541,  2012, 29892,  1090,   349,   392,  2207, 29915, 29879,
#          11525,  8250, 29892,   263,  4654, 29892,   716,  5335,  5570,   338,
#           2825, 29892,   297,   607, 13681, 25083, 29915,  3133,   537,  4893,
#           2058,   515, 29871, 29906, 29900, 29896, 29896,   373,  2935, 29889,
#            512,   450,  1570, 29871, 29945, 29906, 22538,   310, 13681, 29915,
#          29879,  3133,   537, 29892,   382,   711,   538, 29915, 29879,  3978,
#            338,   337, 29899,   342,   370,  3726,   408,   447,  6504,   515,
#            278,  8068,  4412,   310,   263, 29871, 29906, 29945,   386, 29899,
#          27371, 24600,   304,   278, 21967, 29889,  1094,   263,  2278, 29892,
#            382,   711,   538, 16277,   287,   670,  4783, 13406,   670,  5637,
#            322, 17602,   679, 24383, 29889,  3118,    13, 16492, 29901,  1058,
#            413,  6090,  2594,   719,   525, 29879, 16823,   297,   278, 11013,
#          29973,    13, 22550, 29901]
tok_label = [894, 29871]
print(tokenizer.convert_ids_to_tokens(tok_label))
print(tokenizer.decode(tok_label))
#
# output = model(input_ids["input_ids"].cuda())
# #
# logits = output.logits
# probs = torch.softmax(logits, dim=-1)  # 在最后一个维度上应用softmax
# predicted_tokens = torch.argmax(probs, dim=-1)
# print(predicted_tokens)
# # #
# output_text = tokenizer.batch_decode(predicted_tokens, skip_special_tokens=True)
# print(output_text)
#
# hook.remove()


# a = cal_anchor_embedding(['Which is the of?'], model, tokenizer)
# print(a)
# inputs = tokenizer([''], return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
# print(inputs['input_ids'])

# root_dir = '/data3/liuxb/datasets/syn_knowledges/NQ/1'
# files = os.listdir(root_dir)
# for file in tqdm(files):
#     # try:
#     new_file = os.path.join(root_dir, file)
#     with open(new_file, 'r') as f:
#         datas = json.load(f)
#     sentences = [sen[0] for sen in datas]
#     aver = cal_anchor_embedding(sentences, model, tokenizer)
    # except Exception as e:
    #     # 捕获异常并跳过此循环
    #     print(f"Error encountered with input {file}")
    #     continue  # 继续处理下一个元素


# import pandas as pd
#
# data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
# df = pd.DataFrame(data)
# print(df)
# # 选择第2行、第3列的元素 (注意，索引从0开始)
# value = df.iloc[1, 2]
# print(value)  # 输出 8
# from tqdm import tqdm
# from vllm import LLM, SamplingParams
# import numpy as np
# from time import sleep
#
# model = LLM("selfrag/selfrag_llama2_7b", dtype="half")
# sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100, skip_special_tokens=False)
#
#
# def format_prompt(input, paragraph=None):
#   prompt = "### Instruction:\n{0}\n\n### Response:\n".format(input)
#   if paragraph is not None:
#     prompt += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)
#   return prompt
#
# with open("/data3/liuxb/datasets/NQ/NQ_test_results.json") as valid_data:
#     datas = json.load(valid_data)
# Accs = []
# for data in tqdm(datas):
#     sleep(1111)
#     question = data["question"] + "?" if data["question"][-1] != "?" else data["question"]
#     ctxs = data['ctxs'][0]['text']
#     answers = data['answers']
#     prompt = format_prompt(question, paragraph=ctxs)
#     preds = model.generate([prompt], sampling_params)
#     for answer in answers:
#         if answer in preds[0].outputs[0].text:
#             Accs.append(1)
#             break
    # EM, F1 = cal_EM_F1(final_answer['pred'], answers)
    # EMs.append(EM)
    # F1s.append(F1)

# print(len(EMs), np.mean(EMs) * 100)
# print(len(F1s), np.mean(F1s) * 100)
# print(len(Accs), (len(Accs)/len(datas)) * 100)

# generate with retrieved passage
# preds = model.generate([prompt], sampling_params)
# print([pred.outputs[0].text for pred in preds])
# ['[Relevant]Alpacas are considerably smaller than llamas, and unlike llamas, they were not bred to be working animals, but were bred specifically for their fiber.[Fully supported][Utility:5]</s>']

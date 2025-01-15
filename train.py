import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tqdm import tqdm, trange
from data.base import make_Training_loader, make_Validation_loader
from utils import cal_anchor_embedding, cross_entropy, cal_EM_F1, find_subsequence_index
from model.make_model import make_model, replace_layer, recover_layer
import hydra
from torch.optim import AdamW
import torch
from time import sleep
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import copy
import json

def average_embedding_create_pre_hook_fn(sentences, model, tok):
    def pre_hook_fn(module, inputs):
        average_embedding = cal_anchor_embedding(sentences, model, tok)
        # print(average_embedding)
        return (inputs[0], average_embedding)
    return pre_hook_fn


def create_pre_hook_fn(id, model, tok):
    def pre_hook_fn(module, inputs):
        return (inputs[0], id)
    return pre_hook_fn


def EM_F1_of_model_generation(input_id, input, model, tok, answers):
    logits = model.generate(input_id, max_new_tokens=10, temperature=0.0, top_p=1.0, top_k=-1,
                                 do_sample=False)
    predict = tok.decode(logits[0], skip_special_tokens=True)
    EM, F1 = cal_EM_F1(predict[len(input):].strip().split('\n')[0], answers)
    return EM, F1


def train(config, model, tok, train_loader, optimizer, scheduler, layer):
    model.train()
    for epoch in range(config.epochs):
        running_losses = 0.0
        for tuples in tqdm(train_loader, desc="Train"):
            input_id, sentences = tuples
            optimizer.zero_grad()
            hook = model.model.layers[layer].mlp.register_forward_pre_hook(create_pre_hook_fn(0, model, tok))
            logits = model(**input_id)["logits"]
            loss = cross_entropy(logits, input_id["labels"])
            loss.backward()
            optimizer.step()
            running_losses += loss.item()
            hook.remove()
        scheduler.step()
        print(f"Training Loss: {running_losses:.4f}")



def valid(model, tok, valid_loader, layer):
    model.eval()
    for tuples in tqdm(valid_loader, desc="Valid"):
        base_input_id, base_input, direct_input_id, direct_input, prompt_input_id, prompt_input, sentences, answers = tuples
        with torch.no_grad():
            hook = model.model.layers[layer].mlp.register_forward_pre_hook(create_pre_hook_fn(0, model, tok))
            base_EM, base_F1 = EM_F1_of_model_generation(base_input_id, base_input, model, tok, answers)
            hook.remove()

            hook = model.model.layers[layer].mlp.register_forward_pre_hook(create_pre_hook_fn(direct_input_id.size(1)-base_input_id.size(1), model, tok))
            # hook = model.model.layers[layer].mlp.register_forward_pre_hook(create_pre_hook_fn(0, model, tok))
            dir_EM, dir_F1 = EM_F1_of_model_generation(direct_input_id, direct_input, model, tok, answers)
            hook.remove()

            hook = model.model.layers[layer].mlp.register_forward_pre_hook(create_pre_hook_fn(prompt_input_id.size(1)-base_input_id.size(1), model, tok))
            # hook = model.model.layers[layer].mlp.register_forward_pre_hook(create_pre_hook_fn(0, model, tok))
            pro_EM, pro_F1 = EM_F1_of_model_generation(prompt_input_id, prompt_input, model, tok, answers)
            hook.remove()

    return base_EM, base_F1, dir_EM, dir_F1, pro_EM, pro_F1



@hydra.main(config_path="config", config_name="config")
def main(config):
    model, tok = make_model(config)
    metrics = {
        "base_EM": [],
        "base_F1": [],
        "dir_EM": [],
        "dir_F1": [],
        "pro_EM": [],
        "pro_F1": []
    }
    # with open('/data3/liuxb/code/MEMOE/Mechanistic_Interpretation/mlp.json') as file:
    #     layers = json.load(file)

    for i in trange(1000, 1500):
        original_layer = model.model.layers[config.single_layer].mlp
        replace_layer(model, config.single_layer, original_layer, config.num_experts)
        optimizer = AdamW(model.parameters(), lr=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)  # eta_min 是最小学习率
        train_loader = make_Training_loader(config, tok, i)
        valid_loader = make_Validation_loader(config, tok, i)
        train(config, model, tok, train_loader, optimizer, scheduler, layer=config.single_layer)
        results  = valid(model, tok, valid_loader, layer=config.single_layer)
        recover_layer(model, config.single_layer, original_layer)

        for key, value in zip(metrics.keys(), results):
            metrics[key].append(value)

        if (i+1) % 10 == 0:
            for key, value in metrics.items():
                print(key, len(metrics[key]), np.mean(metrics[key]) * 100)



if __name__ == '__main__':
    main()
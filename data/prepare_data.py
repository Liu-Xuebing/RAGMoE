from FlagEmbedding import FlagLLMReranker
import json
import hydra

def write_json(x, path):
    with open(path, "w") as f:
        f.write(json.dumps(x, indent=4))

@hydra.main(config_path='../config', config_name='prepare_data')
def rerank_passage(config):
    # Setting use_fp16 to True speeds up computation with a slight performance degradation
    reranker = FlagLLMReranker(config.model_path, use_fp16=True)
    with open(config.data_path, 'r') as file:
        datas = json.load(file)

    rerank_datas = []
    for data in datas:
        scores = []
        question = data['question'] + '?' if data['question'][-1] != '?' else data['question']
        passages = data['ctxs']
        for pas in passages:
            score = reranker.compute_score([question, pas['text']])
            scores.append(score[0])
        new_passages = [x for _, x in sorted(zip(scores, passages), reverse=True,  key=lambda pair: pair[0])]

        rerank_datas.append({'question': data['question'],
                             'answers': data['answers'],
                             'ctxs': new_passages})
    write_json(rerank_datas, config.output_path)




@hydra.main(config_path='../config', config_name='prepare_data')
def cal_acc_after_rerank(config):
    with open(config.data_path) as fp:
        before_datas = json.load(fp)
    with open(config.output_path) as fp:
        after_datas = json.load(fp)

    before, after = [], []
    for before_data in before_datas:
        before_passages = before_data['ctxs'][:config.topK]
        for bp in before_passages:
            if bp['has_answer']:
                before.append(1)
                break
    for after_data in after_datas:
        after_passages = after_data['ctxs'][:config.topK]
        for ap in after_passages:
            if ap['has_answer']:
                after.append(1)
                break

    print("before acc of Top-{}:".format(config.topK), len(before) / len(before_datas))
    print("after acc of Top-{}:".format(config.topK), len(after) / len(after_datas))



if __name__ == '__main__':
    # rerank_passage()
    cal_acc_after_rerank()
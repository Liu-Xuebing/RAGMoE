# Retrieval Augmented Editing Generation


### Retrieval
The retrieval model uses the **DPR** model. The weights and NQ/TQA data can be obtained using the `./dpr/data/download.py` script. 

For details, please refer to the original [**DPR** repository](https://github.com/facebookresearch/DPR/tree/main).


### Re-ranker
We use the [**bge-reranker-v2-gemma**](https://huggingface.co/BAAI/bge-reranker-v2-gemma) as re-ranker base model in this work and extend our gratitude to its developers for their contributions to advancing re-ranking technologies.
1. Use the `./reranker/prepare_data.py` script to prepare the data for the re-ranker.
2. Training
3. After training, use the following code to merge model.
```bash
python -c "from FlagEmbedding.llm_reranker.merge import merge_llm; merge_llm('BAAI/bge-reranker-v2-gemma', model_dir, target_dir)"
```
# ConvRelExpand

A temporary repository of our KDD 2023 accepted paper - Learning to Relate to Previous Turns in Conversational Search.

# Environment Dependency

Main packages:
- python 3.8
- torch 1.8.1
- transformer 4.2.0
- numpy 1.22
- faiss-gpu 1.7.2
- pyserini 0.16

# Running Steps - Example

We take TopiOCQA dataset as example. (The same for the remaining datasets)

## 1. Download data and Preprocessing

Four public datasets can be download from [QReCC](https://github.com/apple/ml-qrecc), [TopiOCQA](https://github.com/McGill-NLP/topiocqa), [CAsT-19 and CAsT-20](https://www.treccast.ai/). Data preprocessing can refer to the file end with "_raw" in "preprocess" folder.

## 2. Generate gold pseudo relevant labels

First, generate the qrel file by function "create_label_rel_turn" in "preprocess_topiocqa.py"

Second, generate the pseudo relevant label (PRL) by
```
python test_rel_topiocqa.py --config=Config/test_rel_topiocqa.toml
```

The output file "(train)dev_rel_label.json" contains the PRL for each turn.

## 3. Train selector

Use the pseudo relevant training data generated in step 2.
```
python train_selector.py --config=Config/train_selector.toml
```

Then the trained selector can be used for turn relevance judgment for off-the-shelf retriever.

## 4. Generating dense indexing

```
python gen_tokenized_doc.py --config=Config/gen_tokenized_doc.toml
python gen_doc_embeddings.py --config=Config/gen_doc_embeddings.toml
```

## 5. Evaluate with off-the-shelf retriever

Download [ANCE](https://huggingface.co/castorini/ance-msmarco-passage) model.

Change the config with using ANCE as backbone and "True" for "use_PRL"
```
python test_topiocqa.py --config=Config/test_topiocqa.toml
```

## 6. Jointly train selector and retriever

Using both the pseudo relevant training data generated in step 2 and conversational search data.

```
python train_filter_ranking.py --config=Config/train_filter_ranking.toml
```

## 7. Evaluate with fine-tuned retriever

Change the config with using S-R model trained in step 6 as backbone and "False" for "use_PRL"
```
python test_topiocqa.py --config=Config/test_topiocqa.toml
```

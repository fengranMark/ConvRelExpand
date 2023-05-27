from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')

import csv
import argparse
from models import load_model
# from new_models import load_new_model
from utils import check_dir_exist_or_build, pstore, pload, split_and_padding_neighbor, set_seed
from preprocess_topiocqa import load_collection
from data_structure import ConvDataset, ConvDataset_filter
import os
from os import path
from os.path import join as oj
import json
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import faiss
import time
import copy
import pickle
import toml
import torch
from torch import nn
import numpy as np
import pytrec_eval
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
 

def test(args):
    if args.model_type == "ANCE":
        query_tokenizer, query_encoder = load_model("ANCE_Filter", args.pretrained_query_encoder)
        query_encoder = query_encoder.to(args.device)
    elif args.model_type == "BERT":
        query_tokenizer, query_encoder = load_model("BERT_Filter", args.pretrained_query_encoder)
        query_encoder = query_encoder.to(args.device)
    else:
        raise ValueError

    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    test_dataset = ConvDataset_filter(args, query_tokenizer, args.test_file_path)

    test_loader = DataLoader(test_dataset, 
                                batch_size = args.batch_size, 
                                shuffle=False, 
                                collate_fn=test_dataset.get_collate_fn(args, add_doc_info=False))
    query_encoder.eval()
    pred_labels = []
    gold_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader,  desc="Step", disable=args.disable_tqdm):
            bt_query_pair = batch['bt_sent'].to(args.device)
            bt_query_pair_mask = batch['bt_sent_mask'].to(args.device)
            bt_labels = batch['bt_rel_label'].to(args.device)
            bt_sample_id = batch['bt_sample_id']
            logits, probabilities = query_encoder(bt_query_pair, bt_query_pair_mask)  # B * dim
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logits, bt_labels)

            _, bt_pred_labels = probabilities.max(dim=1)
            pred_labels.extend(bt_pred_labels.cpu().numpy().tolist())
            gold_labels.extend(bt_labels.cpu().numpy().tolist())
        
        assert len(pred_labels) == len(gold_labels)
        try:
            F1_score = f1_score(gold_labels, pred_labels, average='macro')
        except:
            embed()
            input()
        logger.info("F1 = {}".format(round(F1_score*100, 4)))

        true_one = 0
        true_zero = 0
        total_one = 0
        total_zero = 0
        for i in range(len(pred_labels)):
            if gold_labels[i] == 1:
                total_one += 1
                if pred_labels[i] == gold_labels[i]:
                    true_one += 1
            else:
                total_zero += 1
                if pred_labels[i] == gold_labels[i]:
                    true_zero += 1
        acc_one = true_one / total_one
        acc_zero = true_zero / total_zero
        acc = (true_one + true_zero) / len(pred_labels)
        logger.info("ACC = {}".format(round(acc*100, 4)))
        logger.info("ACC_one = {}".format(round(acc_one*100, 4)))
        logger.info("ACC_zero = {}".format(round(acc_zero*100, 4)))

    '''
    #with open("output/cast20/filter/cast20_dev_q_ancemultipred_1.json", "w") as f:
    #with open("output/cast19/filter/cast19_dev_q_ancemultipred_1.json", "w") as f:
    with open("output/topiocqa/filter/dev_q_anceweightfewpred_0.005.json", "w") as f:
    #with open("output/qrecc/filter/qrecc_dev_q_bertpred_1.json", "w") as f:
        with torch.no_grad():
            for batch in tqdm(test_loader,  desc="Step", disable=args.disable_tqdm):
                bt_query_pair = batch['bt_sent'].to(args.device)
                bt_query_pair_mask = batch['bt_sent_mask'].to(args.device)
                bt_labels = batch['bt_rel_label'].to(args.device)
                bt_sample_id = batch['bt_sample_id']
                
                logits, probabilities = query_encoder(bt_query_pair, bt_query_pair_mask)  # B * dim
                #loss_func = nn.CrossEntropyLoss()
                #loss = loss_func(logits, bt_labels)
                _, out_classes = probabilities.max(dim=1)

                for i in range(len(bt_sample_id)):
                    if args.model_type == "ANCE":
                        text = query_tokenizer.decode(bt_query_pair[i]).replace('<s>','').rstrip().split('</s>')
                    elif args.model_type == "BERT":
                        text = query_tokenizer.decode(bt_query_pair[i]).replace('[CLS] ','').replace('[PAD]', '').rstrip().split('[SEP]')
                    query = text[0].strip()
                    rel_query = text[1].strip()
                    pred_label = out_classes[i].cpu().numpy().tolist()
                    f.write(
                            json.dumps({
                                "id": bt_sample_id[i],
                                "query": query,
                                "rel_query": rel_query,
                                "pred_label": pred_label
                            }) + "\n")
            
    with open("output/topiocqa/filter/dev_q_anceweightfewpred_0.005.json", "r") as f:
    #with open("output/cast19/filter/cast19_dev_q_ancemultipred_1.json", "r") as f:
    #with open("output/cast20/filter/cast20_dev_q_ancemultipred_1.json", "r") as f:
    #with open("output/qrecc/filter/qrecc_dev_q_bertpred_1.json", "r") as f:
        data = f.readlines()
    print(len(data))
    rel_label = {}
    rel_list = []
    for i, line in enumerate(data):
        line = json.loads(line)
        id_list = line["id"].split('-')
        conv_id = int(id_list[0]) 
        turn_id = int(id_list[1])
        type_id = int(id_list[-1])
        if (i + 1) != len(data):
            next_turn_id = int(json.loads(data[i + 1])["id"].split('-')[1])
            next_conv_id = int(json.loads(data[i + 1])["id"].split('-')[0])

        if type_id != 0:
            rel_list.append(line["pred_label"])
        
        if (i + 1) == len(data) or turn_id != next_turn_id or conv_id != next_conv_id:
            rel_label[id_list[0] + '-1'] = []
            rel_label[id_list[0] + '-' + id_list[1]] = rel_list
            rel_list = []

    with open("output/topiocqa/dense_rel/dev_anceweightfewpred_label_rawq_0.005.json", "w") as f:
    #with open("output/cast20/dense_rel/dev_ancemultipred_label_rawq_1.json", "w") as f:
    #with open("output/cast19/dense_rel/dev_ancemultipred_label_rawq_1.json", "w") as f:
    #with open("output/qrecc/dense_rel/dev_bertpred_label_rawq_1.json", "w") as f:
        for key, value in rel_label.items():
            id_list = key.split('-')
            conv_id = id_list[0]
            turn_id = id_list[1]
            f.write(
                json.dumps({
                    "id": str(key),
                    "conv_id": str(conv_id),
                    "turn_id": str(turn_id),
                    "rel_label": value
                }) + "\n")        
    print("Write Finish")
    
    logger.info("Test finish!")
    '''

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config",
                        type = str,
                        required = True,
                        help = "Config file path.")

    args = parser.parse_args()
    config = toml.load(args.config)
    args = argparse.Namespace(**config)

    # device + dir check
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    #if os.path.exists(args.model_output_path) and os.listdir(
    #    args.model_output_path) and not args.overwrite_output_dir:
    #    raise ValueError(
    #        "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome."
    #        .format(args.model_output_path))

    #check_dir_exist_or_build([args.model_output_path, args.log_dir_path])
    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)

    return args



if __name__ == '__main__':
    args = get_args()
    set_seed(args)
  
    test(args)

from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')

import time
import copy
import pickle
import random
import numpy as np
import csv
import argparse
import toml
import os

from os import path
from os.path import join as oj
import json
from tqdm import tqdm, trange

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from transformers import get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter

from models import load_model
from new_models import load_new_model
from utils import check_dir_exist_or_build, pstore, pload, set_seed, get_optimizer, print_res
from data_structure import ConvDataset, ConvDataset_filter
#os.environ["CUDA_VISIBLE_DEVICES"]="1"



def save_model(args, model, query_tokenizer, save_model_order, epoch, step, loss):
    output_dir = oj(args.model_output_path, 'model-{}-epoch-{}-step-{}-loss-{}'.format(save_model_order, epoch, step, loss))
    check_dir_exist_or_build([output_dir])
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir)
    query_tokenizer.save_pretrained(output_dir)
    logger.info("Step {}, Save checkpoint at {}".format(step, output_dir))

def cal_ranking_loss(query_embs, pos_doc_embs, neg_doc_embs):
    batch_size = len(query_embs)

    pos_scores = query_embs.mm(pos_doc_embs.T)  # B * B
    neg_scores = torch.sum(query_embs * neg_doc_embs, dim = 1).unsqueeze(1) # B * 1 hard negatives
    score_mat = torch.cat([pos_scores, neg_scores], dim = 1)    # B * (B + 1)  in_batch negatives + 1 BM25 hard negative 
    label_mat = torch.arange(batch_size).to(args.device) # B
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(score_mat, label_mat)

    return loss


def train(args):
    # load the pretrained passage encoder model, but it will be frozen when training.
    #passage_tokenizer, passage_encoder = load_model("ANCE_Passage", args.pretrained_passage_encoder)
    #passage_encoder = passage_encoder.to(args.device)
    # load conversational query encoder model
    #if args.continue_train_query_encoder:
    #    query_tokenizer, query_encoder = load_model(args.model_type + "_Query", args.continue_train_query_encoder)
    #    query_encoder = query_encoder.to(args.device)
    if args.model_type == "ANCE":
        query_tokenizer, query_encoder = load_model("ANCE_Filter", args.pretrained_query_encoder)
        query_encoder = query_encoder.to(args.device)
    elif args.model_type == "BERT":
        query_tokenizer, query_encoder = load_model("BERT_Filter", args.pretrained_query_encoder)
        query_encoder = query_encoder.to(args.device)
    else:
        raise ValueError
    if args.n_gpu > 1:
        query_encoder = torch.nn.DataParallel(query_encoder, device_ids = list(range(args.n_gpu)))

    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    
    # data prepare
    #train_dataset = ConvDataset_filter(args, query_tokenizer, args.train_file_path, args.combine_train_file_path)
    train_dataset = ConvDataset_filter(args, query_tokenizer, args.train_file_path)

    train_loader = DataLoader(train_dataset, 
                                batch_size = args.batch_size, 
                                shuffle=True, 
                                collate_fn=train_dataset.get_collate_fn(args, add_doc_info=True))

    logger.info("train samples num = {}".format(len(train_dataset)))
    
    
    test_dataset = ConvDataset_filter(args, query_tokenizer, args.test_file_path)
    #test_dataset = ConvDataset_filter(args, query_tokenizer, args.test_file_path)

    test_loader = DataLoader(test_dataset, 
                                batch_size = args.batch_size, 
                                shuffle=False, 
                                collate_fn=test_dataset.get_collate_fn(args, add_doc_info=False))

    logger.info("test samples num = {}".format(len(test_dataset)))
    
    total_training_steps = args.num_train_epochs * (len(train_dataset) // args.batch_size + int(bool(len(train_dataset) % args.batch_size)))
    num_warmup_steps = args.num_warmup_portion * total_training_steps
    
    optimizer = get_optimizer(args, query_encoder, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_training_steps)

    global_step = 0
    save_model_order = 0

    # begin to train
    logger.info("Start training...")
    logger.info("Total training epochs = {}".format(args.num_train_epochs))
    logger.info("Total training steps = {}".format(total_training_steps))
    
    num_steps_per_epoch = total_training_steps // args.num_train_epochs
    logger.info("Num steps per epoch = {}".format(num_steps_per_epoch))

    if isinstance(args.save_steps, float):
        args.save_steps = int(args.save_steps * num_steps_per_epoch)
        args.save_steps = max(1, args.save_steps)
    if isinstance(args.print_steps, float):
        args.print_steps = int(args.print_steps * num_steps_per_epoch)
        args.print_steps = max(1, args.print_steps)

    epoch_iterator = trange(args.num_train_epochs, desc="Epoch", disable=args.disable_tqdm)

    #if args.continue_train_query_encoder:
    #    model_num = args.continue_train_query_encoder.strip().split('-')
    #    epoch_iterator = trange(int(model_num[3]) + 1, args.num_train_epochs, desc="Epoch", disable=args.disable_tqdm)
    #    global_step = int(model_num[5])
    #    save_model_order = int(model_num[1])
    best_score = 0
    for epoch in epoch_iterator:
        query_encoder.train()
        #passage_encoder.eval()
        for batch in tqdm(train_loader,  desc="Step", disable=args.disable_tqdm):
            query_encoder.zero_grad()

            bt_sent = batch['bt_sent'].to(args.device)
            bt_sent_mask = batch['bt_sent_mask'].to(args.device)
            bt_labels = batch['bt_rel_label'].to(args.device)
            bt_sample_id = batch['bt_sample_id']
            
            logits, probabilities = query_encoder(bt_sent, bt_sent_mask)  # B * dim
            # neg: 270000 260749 pos: 20000 30500
            # qrecc one 15294 zero 72048
            weights = [1.0, 270000 / 20000]
            class_weights = torch.FloatTensor(weights).to(args.device)
            loss_func = nn.CrossEntropyLoss(weight=class_weights)
            #loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logits, bt_labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(query_encoder.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            if args.print_steps > 0 and global_step % args.print_steps == 0:
                logger.info("Epoch = {}, Global Step = {}, Loss = {}".format(
                                epoch + 1,
                                global_step,
                                #ranking_loss.item()
                                loss.item())
                            )


            global_step += 1    # avoid saving the model of the first step.
            # save model finally
            #if args.save_steps > 0 and global_step % args.save_steps == 0:

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
                pred_labels.extend(bt_pred_labels)
                gold_labels.extend(bt_labels)
            
            assert len(pred_labels) == len(gold_labels)
            F1_score = f1_score(gold_labels, pred_labels, average='macro'))
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

        #if (epoch + 1) % 1 == 0 and acc_one > best_score:
        if acc_one > best_score:
            save_model(args, query_encoder, query_tokenizer, save_model_order, epoch, global_step, loss.item())
            save_model_order += 1
            best_score = acc_one

            query_encoder.eval()
            pred_labels = []
            gold_labels = []
            with open("output/topiocqa/filter/dev_q_anceweightfewpred_0.001.json", "w") as f:
            #with open("output/qrecc/dense_rel/dev_q_bertpred_1.json", "w") as f:
                with torch.no_grad():
                    for batch in tqdm(test_loader,  desc="Step", disable=args.disable_tqdm):
                        bt_query_pair = batch['bt_sent'].to(args.device)
                        bt_query_pair_mask = batch['bt_sent_mask'].to(args.device)
                        bt_labels = batch['bt_rel_label'].to(args.device)
                        bt_sample_id = batch['bt_sample_id']
                        
                        logits, probabilities = query_encoder(bt_query_pair, bt_query_pair_mask)  # B * dim
                        loss_func = nn.CrossEntropyLoss()
                        loss = loss_func(logits, bt_labels)
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
                
            with open("output/topiocqa/filter/dev_q_anceweightfewpred_0.001.json", "r") as f:
            #with open("output/qrecc/dense_rel/dev_q_bertpred_1.json", "r") as f:
                data = f.readlines()
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
                    next_conv_id = int(json.loads(data[i + 1])["id"].split('-')[1])

                if type_id != 0:
                    rel_list.append(line["pred_label"])
                
                if (i + 1) == len(data) or turn_id != next_turn_id or conv_id != next_conv_id:
                    rel_label[id_list[0] + '-1'] = []
                    rel_label[id_list[0] + '-' + id_list[1]] = rel_list
                    rel_list = []

            with open(oj("output/topiocqa/filter", "dev_anceweightfewpred_label_rawq_0.001.json"), "w") as f:
            #with open(oj("output/qrecc/dense_rel", "dev_bertpred_label_rawq_1.json"), "w") as f:
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
            logger.info("Write Finish")
    logger.info("Training finish!")          
         


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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    if os.path.exists(args.model_output_path) and os.listdir(
        args.model_output_path) and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            .format(args.model_output_path))

    check_dir_exist_or_build([args.model_output_path, args.log_dir_path])
    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)
    
    return args


if __name__ == '__main__':
    args = get_args()
    set_seed(args)
  
    train(args)

#  python train_filter.py --config Config/train_filter.toml
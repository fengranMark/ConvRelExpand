# data structure library file

from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')
import pandas as pd

import argparse
import torch
import toml
from utils import check_dir_exist_or_build, pstore, pload, split_and_padding_neighbor, set_seed, load_collection
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset
import json
from tqdm import tqdm, trange
import random
from itertools import combinations
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from models import load_model

class ConvExample_topiocqa:
    def __init__(self, sample_id,
                       conv_id,
                       turn_id, 
                       conv_query, 
                       conv_query_ans,
                       topic,
                       sub_topic,
                       pos_docs = None,
                       neg_docs = None,
                       answer = None,
                       raw_query = None,
                       oracle_query = None):
        self.sample_id = sample_id
        self.conv_id = conv_id
        self.turn_id = turn_id
        self.conv_query = conv_query
        self.conv_query_ans = conv_query_ans
        self.topic = topic
        self.sub_topic = sub_topic
        self.pos_docs = pos_docs
        self.neg_docs = neg_docs
        self.answer = answer
        self.raw_query = raw_query
        self.oracle_query = oracle_query


class ConvDataset_topiocqa(Dataset):
    def __init__(self, args, query_tokenizer, passage_tokenizer, filename, add_doc_info=True):
        self.examples = []
        
        with open(filename, 'r') as f:
            data = f.readlines()

        if args.use_PRF:
            with open(args.PRF_file, 'r') as f:
                PRF = f.readlines()

        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for testing
        #if n < len(data):
        #   random.seed(args.seed)
        #   data = random.sample(data, n)

        logger.info("Loading {} data file...".format(filename))
        for i in trange(n):
            # basic
            data[i] = json.loads(data[i])
            sample_id = data[i]['id']
            conv_id = data[i]['conv_id']
            turn_id = data[i]['turn_id']
            history_query = data[i]["history_query"]
            history_answer = data[i]["history_answer"]
            topic = data[i]["topic"]
            sub_topic = data[i]["sub_topic"]
            query = data[i]["query"]
            last_response = data[i]["last_response"]
            #PRF[i] = json.loads(PRF[i])
            if args.use_PRF:
                PRF[i] = json.loads(PRF[i])
                rel_label = PRF[i]["rel_label"]
            
            '''
            if args.topic_mode == "shift":
                #if args.use_PRF:
                    #PRF[i] = json.loads(PRF[i])
                rel_label = PRF[i]["rel_label"]
                turn_num = len(rel_label)
                if len(rel_label) == 0 or not ((rel_label[turn_num - 1] == 0) and (1 not in rel_label)):
                    continue
            elif args.topic_mode == "return":
                #if args.use_PRF:
                    #PRF[i] = json.loads(PRF[i])
                rel_label = PRF[i]["rel_label"]
                turn_num = len(rel_label)
                if len(rel_label) == 0 or not ((rel_label[turn_num - 1] == 0) and (1 in rel_label)):
                    continue
            elif args.topic_mode == "no":
                #if args.use_PRF:
                    #PRF[i] = json.loads(PRF[i])
                rel_label = PRF[i]["rel_label"]
                turn_num = len(rel_label)
                if len(rel_label) == 0 or not (rel_label[turn_num - 1] == 1):
                    continue
            '''

            # query
            conv_query = [] # q_i, q_i-1 ... q_2, q_1
            conv_query_ans = [] # q_i, q_i-1 ... a_2, q_1, a_1
            topic_info = topic + ' ' + sub_topic
                
            cur_query = query_tokenizer.encode(query, add_special_tokens=True)
            conv_query.extend(cur_query)
            conv_query_ans.extend(cur_query)

            if len(last_response) > 0 and args.use_last_response and args.mode != "convqa":
                lp = []
                #lp.extend = query_tokenizer.encode(last_response, add_special_tokens=True)
                lp.append(query_tokenizer.cls_token_id)
                lp.extend(query_tokenizer.convert_tokens_to_ids(["<response>"]))
                lp.extend(query_tokenizer.convert_tokens_to_ids(query_tokenizer.tokenize(last_response)))
                lp = lp[:args.max_doc_length]
                lp.append(query_tokenizer.sep_token_id)
                conv_query.extend(lp)
                history_context.extend(lp)
            # context
            # context = []
            assert len((history_query)) == len((history_answer))
            if not args.use_PRF:
                if len(history_query) > 0:
                    for j in range(len(history_query)-1, -1, -1):
                    #for j in range(len(history_query)):
                        conv_query.extend(query_tokenizer.encode(history_query[j], add_special_tokens=True, max_length=args.max_query_length))
                        conv_query_ans.extend(query_tokenizer.encode(history_query[j], add_special_tokens=True, max_length=args.max_query_length))
                        conv_query_ans.extend(query_tokenizer.encode(history_answer[j], add_special_tokens=True, max_length=args.max_query_length))
            else: # use PRF
                if len(rel_label) > 0:
                    if args.PRF_mode == "hard":
                        token_set = []
                        for key in history_query:
                            sent = key.strip().split()
                            token_set.extend(sent)
                        for j in range(len(rel_label)):
                            if rel_label[j] == 1:
                                conv_query.append(query_tokenizer.convert_tokens_to_ids(token_set[j]))
                        conv_query.append(query_tokenizer.sep_token_id)
                        #for j in range(len(rel_label)-1, -1, -1):
                        #    if rel_label[j] == 1:
                        #        conv_query.extend(query_tokenizer.encode(history_query[j], add_special_tokens=True, max_length=args.max_query_length))
                        #        conv_query_ans.extend(query_tokenizer.encode(history_query[j], add_special_tokens=True, max_length=args.max_query_length))
                        #        conv_query_ans.extend(query_tokenizer.encode(history_answer[j], add_special_tokens=True, max_length=args.max_query_length))
                    elif args.PRF_mode == "soft":
                        for j in range(len(rel_label)-1, -1, -1):
                            if rel_label[j] == 1:
                                rel_q = []
                                #rel_q.append(query_tokenizer.cls_token_id)
                                rel_q.extend(query_tokenizer.convert_tokens_to_ids(["<rel>"]))
                                rel_q.extend(query_tokenizer.convert_tokens_to_ids(query_tokenizer.tokenize(history_query[j])))
                                #rel_q.extend(query_tokenizer.convert_tokens_to_ids(["</rel>"]))
                                rel_q = rel_q[:args.max_query_length]
                                rel_q.append(query_tokenizer.sep_token_id)
                                conv_query.extend(rel_q)
                            else:
                                irrel_q = []
                                #irrel_q.append(query_tokenizer.cls_token_id)
                                irrel_q.extend(query_tokenizer.convert_tokens_to_ids(["<irrel>"]))
                                irrel_q.extend(query_tokenizer.convert_tokens_to_ids(query_tokenizer.tokenize(history_query[j])))
                                #irrel_q.extend(query_tokenizer.convert_tokens_to_ids(["</irrel>"]))
                                irrel_q = rel_q[:args.max_query_length]
                                irrel_q.append(query_tokenizer.sep_token_id)
                                conv_query.extend(irrel_q)
                            

            # doc 
            pos_docs = []
            neg_docs = []
            pos_docs_id = []
            neg_docs_id = []
            if add_doc_info:
                for doc in data[i]['pos_docs']:
                    pos_docs.append(passage_tokenizer.encode(doc, add_special_tokens=True))
                    pos_docs_id.append(data[i]['pos_docs_id'])
                #seen_neg_docs = set()
                for doc in data[i]['neg_docs']:
                #    if doc in data[i]['pos_docs'] or doc in seen_neg_docs:
                #        continue
                #    seen_neg_docs.add(doc)
                    neg_docs.append(passage_tokenizer.encode(doc, add_special_tokens=True))   
                    neg_docs_id = data[i]['neg_docs_id']
                # But if no neg_docs, at least add one
                #if len(neg_docs) == 0:
                #    neg_docs.append(passage_tokenizer.encode(doc , add_special_tokens=True, max_length=args.max_doc_length))

            # For baseline test
            raw_query = cur_query
            if "answer" in data[i]:
                answer = query_tokenizer.encode(data[i]['answer'], add_special_tokens=True)
            else:
                answer = None
            if "rewrite" in data[i]: 
                oracle_query = query_tokenizer.encode(data[i]['rewrite'], add_special_tokens=True)
            else:
                oracle_query = None

            self.examples.append(ConvExample_topiocqa(sample_id,
                                            conv_id,
                                            turn_id, 
                                            conv_query, 
                                            conv_query_ans, 
                                            topic,
                                            sub_topic,
                                            pos_docs, 
                                            neg_docs,
                                            answer = answer,
                                            raw_query = raw_query,
                                            oracle_query = oracle_query))          

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


    @staticmethod
    def get_collate_fn(args, add_doc_info:bool, mode:str):

        def collate_fn(batch: list):
            collated_dict = {
                "bt_sample_id": [],
                "bt_conv_id": [],
                "bt_turn_id": [],
                "bt_conv_query":[],
                "bt_conv_query_mask":[],
                "bt_conv_query_ans":[],
                "bt_conv_query_ans_mask":[],
                "bt_raw_query":[],
                "bt_raw_query_mask":[],
                "bt_oracle_query":[],
                "bt_oracle_query_mask":[],
                "bt_answer":[],
                "bt_answer_mask":[],
                "bt_pos_docs":[],
                "bt_pos_docs_mask":[],
                "bt_neg_docs":[],
                "bt_neg_docs_mask":[],
            }
            
            bt_sample_id = [] 
            bt_conv_id = [] 
            bt_turn_id = [] 
            bt_conv_query = []
            bt_conv_query_mask = []
            bt_conv_query_ans = []
            bt_conv_query_ans_mask = []
            bt_raw_query = []
            bt_raw_query_mask = []
            bt_oracle_query = []
            bt_oracle_query_mask = []
            bt_answer = []
            bt_answer_mask = []
            
            # for doc
            bt_pos_docs = []
            bt_pos_docs_mask = []
            bt_neg_docs = []
            bt_neg_docs_mask = []


            for example in batch:
                # padding
                conv_query, conv_query_mask = pad_seq_ids_with_mask(example.conv_query, max_length = args.max_concat_length)
                conv_query_ans, conv_query_ans_mask = pad_seq_ids_with_mask(example.conv_query_ans, max_length = args.max_concat_length)
                
                if example.raw_query:
                    raw_query, raw_query_mask = pad_seq_ids_with_mask(example.raw_query, max_length = args.max_query_length)
                if example.oracle_query:
                    oracle_query, oracle_query_mask = pad_seq_ids_with_mask(example.oracle_query, max_length = args.max_query_length)
                if example.answer:
                    answer, answer_mask = pad_seq_ids_with_mask(example.answer, max_length = args.max_query_length)

                bt_sample_id.append(example.sample_id)
                bt_conv_id.append(example.conv_id)
                bt_turn_id.append(example.turn_id)
                bt_conv_query.append(conv_query)
                bt_conv_query_mask.append(conv_query_mask)
                bt_conv_query_ans.append(conv_query_ans)
                bt_conv_query_ans_mask.append(conv_query_ans_mask)

                if example.raw_query:
                    bt_raw_query.append(raw_query)
                    bt_raw_query_mask.append(raw_query_mask)
                if example.oracle_query:
                    bt_oracle_query.append(oracle_query)
                    bt_oracle_query_mask.append(oracle_query_mask)
                if example.answer:
                    bt_answer.append(answer)
                    bt_answer_mask.append(answer_mask)
                
                if add_doc_info:
                    assert len(example.pos_docs) > 0
                    assert len(example.neg_docs) > 0
                    pos_doc = random.sample(example.pos_docs, 1)[0]
                    neg_doc = random.sample(example.neg_docs, 1)[0] # only one neg
                    pos_doc, pos_doc_mask = pad_seq_ids_with_mask(pos_doc, max_length = args.max_doc_length)
                    neg_doc, neg_doc_mask = pad_seq_ids_with_mask(neg_doc, max_length = args.max_doc_length)
    
                    bt_pos_docs.append(pos_doc)
                    bt_pos_docs_mask.append(pos_doc_mask)
                    bt_neg_docs.append(neg_doc)
                    bt_neg_docs_mask.append(neg_doc_mask)

            collated_dict["bt_sample_id"] = bt_sample_id
            collated_dict["bt_conv_id"] = bt_conv_id
            collated_dict["bt_turn_id"] = bt_turn_id
            collated_dict["bt_conv_query"] = bt_conv_query
            collated_dict["bt_conv_query_mask"] = bt_conv_query_mask
            collated_dict["bt_conv_query_ans"] = bt_conv_query_ans
            collated_dict["bt_conv_query_ans_mask"] = bt_conv_query_ans_mask
            collated_dict["bt_raw_query"] = bt_raw_query
            collated_dict["bt_raw_query_mask"] = bt_raw_query_mask
            collated_dict["bt_oracle_query"] = bt_oracle_query
            collated_dict["bt_oracle_query_mask"] = bt_oracle_query_mask
            collated_dict["bt_answer"] = bt_answer
            collated_dict["bt_answer_mask"] = bt_answer_mask

            collated_dict["bt_pos_docs"] = bt_pos_docs
            collated_dict["bt_pos_docs_mask"] = bt_pos_docs_mask
            collated_dict["bt_neg_docs"] = bt_neg_docs
            collated_dict["bt_neg_docs_mask"] = bt_neg_docs_mask
            
            # change to tensor
            for key in collated_dict:
                if key not in ["bt_sample_id", "bt_conv_id", "bt_turn_id"]:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        def collate_fn_test(batch: list):
            collated_dict = {
                "bt_sample_id": [],
                "bt_conv_id": [],
                "bt_turn_id": [],
                "bt_conv_query":[],
                "bt_conv_query_mask":[],
                "bt_conv_query_ans":[],
                "bt_conv_query_ans_mask":[],
                "bt_raw_query":[],
                "bt_raw_query_mask":[],
                "bt_oracle_query":[],
                "bt_oracle_query_mask":[],
            }

            # for query
            bt_sample_id = [] 
            bt_conv_id = [] 
            bt_turn_id = [] 
            bt_conv_query = []
            bt_conv_query_mask = []
            bt_conv_query_ans = []
            bt_conv_query_ans_mask = []
            bt_raw_query = []
            bt_raw_query_mask = []
            bt_oracle_query = []
            bt_oracle_query_mask = []

            for example in batch:
                # padding
                conv_query, conv_query_mask = pad_seq_ids_with_mask(example.conv_query, max_length = args.max_concat_length)
                conv_query_ans, conv_query_ans_mask = pad_seq_ids_with_mask(example.conv_query_ans, max_length = args.max_concat_length)

                if example.raw_query:
                    raw_query, raw_query_mask = pad_seq_ids_with_mask(example.raw_query, max_length = args.max_query_length)
                if example.oracle_query:
                    oracle_query, oracle_query_mask = pad_seq_ids_with_mask(example.oracle_query, max_length = args.max_query_length)

                bt_sample_id.append(example.sample_id)
                bt_conv_id.append(example.conv_id)
                bt_turn_id.append(example.turn_id)
                bt_conv_query.append(conv_query)
                bt_conv_query_mask.append(conv_query_mask)
                bt_conv_query_ans.append(conv_query_ans)
                bt_conv_query_ans_mask.append(conv_query_ans_mask)

                if example.raw_query:
                    bt_raw_query.append(raw_query)
                    bt_raw_query_mask.append(raw_query_mask)
                if example.oracle_query:
                    bt_oracle_query.append(oracle_query)
                    bt_oracle_query_mask.append(oracle_query_mask)
                            
            collated_dict["bt_sample_id"] = bt_sample_id
            collated_dict["bt_conv_id"] = bt_conv_id
            collated_dict["bt_turn_id"] = bt_turn_id
            collated_dict["bt_conv_query"] = bt_conv_query
            collated_dict["bt_conv_query_mask"] = bt_conv_query_mask
            collated_dict["bt_conv_query_ans"] = bt_conv_query_ans
            collated_dict["bt_conv_query_ans_mask"] = bt_conv_query_ans_mask
            collated_dict["bt_raw_query"] = bt_raw_query
            collated_dict["bt_raw_query_mask"] = bt_raw_query_mask
            collated_dict["bt_oracle_query"] = bt_oracle_query
            collated_dict["bt_oracle_query_mask"] = bt_oracle_query_mask

            
            # change to tensor
            for key in collated_dict:
                if key not in ["bt_sample_id", "bt_conv_id", "bt_turn_id"]:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
              
            return collated_dict

        if mode == "test":
            return collate_fn_test
        elif mode == "train":
            return collate_fn
        else:
            raise ValueError

class ConvExample_filter:
    def __init__(self, sample_id,
                       cur_query, 
                       pair_query,
                       sent,
                       rel_label=None):
        self.sample_id = sample_id
        self.cur_query = cur_query
        self.pair_query = pair_query
        self.sent = sent
        self.rel_label = rel_label

class ConvDataset_filter(Dataset):
    def __init__(self, args, query_tokenizer, filename, combine_file=False, add_doc_info=True):
        self.examples = []

        with open(filename, 'r') as f:
            data = f.readlines()

        if combine_file:
            logger.info("Loading {} data file...".format(combine_file))
            with open(combine_file, 'r') as f:
                combine_data = f.readlines()
            #m = len(combine_data)
            data.extend(combine_data)

        n = len(data)
        n = int(args.use_data_percent * n)


        if n < len(data):
            random.seed(args.seed)
            data = random.sample(data, n)

        logger.info("Loading {} data file...".format(filename))
        for i in trange(n):
            # basic
            data[i] = json.loads(data[i])
            if 'id' in data[i]:
                sample_id = data[i]['id']
            else:
                sample_id = i
            cur_query = data[i]['query']
            if 'rel_query' in data[i]:
                pair_query = data[i]['rel_query']
            elif 'query_pair' in data[i]:
                pair_query = data[i]['query_pair']
            if 'rel_label' in data[i]:
                rel_label = data[i]['rel_label']
            else:
                rel_label = None
            sent = []
            sent.append(query_tokenizer.cls_token_id)
            sent.extend(query_tokenizer.convert_tokens_to_ids(query_tokenizer.tokenize(cur_query)))
            sent.append(query_tokenizer.sep_token_id)
            sent.extend(query_tokenizer.convert_tokens_to_ids(query_tokenizer.tokenize(pair_query)))
            sent.append(query_tokenizer.sep_token_id)

            self.examples.append(ConvExample_filter(sample_id,
                                                    cur_query, 
                                                    pair_query,
                                                    sent,
                                                    rel_label=rel_label
                                                    )) 

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @staticmethod
    def get_collate_fn(args, add_doc_info:bool):

        def collate_fn(batch: list):
            collated_dict = {
                "bt_sample_id": [],
                "bt_cur_query":[],
                "bt_cur_query_mask":[],
                "bt_pair_query":[],
                "bt_pair_query_mask":[],
                "bt_sent":[],
                "bt_sent_mask":[],
                "bt_rel_label":[],
            }
            
            bt_sample_id = [] 
            #bt_query = []
            #bt_query_mask = []
            #bt_pair_query = []
            #bt_pair_query_mask = []
            bt_sent = []
            bt_sent_mask = []
            bt_rel_label = []

            for example in batch:
                # padding
                sent, sent_mask = pad_seq_ids_with_mask(example.sent, max_length = args.max_filter_length)
                bt_sample_id.append(example.sample_id)
                if example.rel_label is not None:
                    bt_rel_label.append(example.rel_label)
                bt_sent.append(sent)
                bt_sent_mask.append(sent_mask)          

            collated_dict["bt_sample_id"] = bt_sample_id
            if example.rel_label is not None:
                collated_dict["bt_rel_label"] = bt_rel_label
            collated_dict["bt_sent"] = bt_sent
            collated_dict["bt_sent_mask"] = bt_sent_mask
            
            # change to tensor
            for key in collated_dict:
                if key not in ["bt_sample_id"]:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        return collate_fn

class ConvExample_quretec:
    def __init__(self, sample_id,
                       cur_query, 
                       context,
                       expansion_term=None,
                       rel_label=None):
        self.sample_id = sample_id
        self.cur_query = cur_query
        self.context = context
        self.expansion_term = expansion_term
        self.rel_label = rel_label

class ConvDataset_quretec(Dataset):
    def __init__(self, args, query_tokenizer, filename):
        self.examples = []

        with open(filename, 'r') as f:
            data = f.readlines()

        n = len(data)
        n = int(args.use_data_percent * n)


        if n < len(data):
            random.seed(args.seed)
            data = random.sample(data, n)

        logger.info("Loading {} data file...".format(filename))
        for i in trange(n):
            
            # train
            '''
            data[i] = json.loads(data[i])
            #if 'id' in data[i]:
            sample_id = data[i]['id']
            #else:
            #    sample_id = i
            cur_query = data[i]['query']
            if "expansion" in data[i]:
                expansion_term = data[i]["expansion"] # a sequence
                expansion_tokenid = query_tokenizer.encode(expansion_term)[1:-1]
            #history_query = data[i]["history_query"] # a sequence
            if 'rel_label' in data[i]:
                rel_label = data[i]['rel_label']
            else:
                rel_label = []
            '''
            
            #test
            
            data[i] = json.loads(data[i])
            #history_query = data[i]["input"][:-1] # cast
            history_query = data[i]["context_queries"]
            history_answer = data[i]["context_answers"]
            history_query.extend(history_answer)
            cur_query = data[i]["query"]
            #cur_query = data[i]["input"][-1] # cast
            #sample_id = data[i]['id'] # cast
            sample_id = data[i]['sample_id']
            
            sent = []
            sent.append(query_tokenizer.cls_token_id)
            for key in history_query:
                sent.extend(query_tokenizer.convert_tokens_to_ids(query_tokenizer.tokenize(key)))
            sent.append(query_tokenizer.sep_token_id)
            sent.extend(query_tokenizer.convert_tokens_to_ids(query_tokenizer.tokenize(cur_query)))
            
            if "expansion" in data[i]:
                for idx in sent:
                    if idx in expansion_tokenid:
                        rel_label.append(1)
                    else:
                        rel_label.append(0)
                assert len(rel_label) == len(sent)
            else:
                expansion_term = None
                rel_label = None
                        
            self.examples.append(ConvExample_quretec(sample_id,
                                                    cur_query, 
                                                    sent,
                                                    expansion_term = expansion_term,
                                                    rel_label=rel_label
                                                    )) 

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @staticmethod
    def get_collate_fn(args):

        def collate_fn(batch: list):
            collated_dict = {
                "bt_sample_id": [],
                "bt_cur_query":[],
                "bt_cur_query_mask":[],
                "bt_context":[],
                "bt_context_mask":[],
                "bt_rel_label":[],
            }
            
            bt_sample_id = [] 
            bt_cur_query = []
            bt_cur_query_mask = []
            #bt_pair_query = []
            #bt_pair_query_mask = []
            bt_context = []
            bt_context_mask = []
            bt_rel_label = []

            for example in batch:
                # padding
                context, context_mask = pad_seq_ids_with_mask(example.context, max_length = args.max_concat_length)
                cur_query, cur_query_mask = pad_seq_ids_with_mask(example.context, max_length = args.max_query_length)
                bt_sample_id.append(example.sample_id)
                if example.rel_label is not None:
                    rel_label, _ = pad_seq_ids_with_mask(example.rel_label, max_length = args.max_concat_length)
                    bt_rel_label.append(rel_label)
                bt_context.append(context)
                bt_context_mask.append(context_mask)  
                bt_cur_query.append(cur_query)
                bt_cur_query_mask.append(cur_query_mask)        

            collated_dict["bt_sample_id"] = bt_sample_id
            if example.rel_label is not None:
                collated_dict["bt_rel_label"] = bt_rel_label
            collated_dict["bt_context"] = bt_context
            collated_dict["bt_context_mask"] = bt_context_mask
            collated_dict["bt_cur_query"] = bt_cur_query
            collated_dict["bt_cur_query_mask"] = bt_cur_query_mask


            #embed()
            #input()
            collated_dict["bt_cur_query"] = torch.tensor(collated_dict["bt_cur_query"], dtype=torch.long)
            collated_dict["bt_cur_query_mask"] = torch.tensor(collated_dict["bt_cur_query_mask"], dtype=torch.long)
            collated_dict["bt_context"] = torch.tensor(collated_dict["bt_context"], dtype=torch.long)
            collated_dict["bt_context_mask"] = torch.tensor(collated_dict["bt_context_mask"], dtype=torch.long)
            if example.rel_label is not None:
                collated_dict["bt_rel_label"] = torch.tensor(collated_dict["bt_rel_label"], dtype=torch.long)
            # change to tensor
            #for key in collated_dict:
            #    if key not in ["bt_sample_id"]:
            #        collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        return collate_fn
    
class ConvExample_rewrite:
    def __init__(self, sample_id, 
                       #query,
                       rewrite):
        self.sample_id = sample_id
        #self.query = query
        self.rewrite = rewrite


class ConvDataset_rewrite(Dataset):
    def __init__(self, args, query_tokenizer, filename):
        self.examples = []

        with open(filename, 'r') as f:
            data = f.readlines()

        with open(args.test_file_path_2, 'r') as f2:
            data_2 = f2.readlines()

        n = len(data)
        n = int(args.use_data_percent * n)

        logger.info("Loading {} data file...".format(filename))
        logger.info("Loading {} data file...".format(args.test_file_path_2))

        for i in trange(n):
            data[i] = json.loads(data[i])
            if 'id' in data[i]:
                sample_id = data[i]['id']
            else:
                sample_id = data[i]['sample_id']
            if 'output' in data[i]:
                rewrite = data[i]['output']
            elif 'rewrite' in data[i]:
                rewrite = data[i]['rewrite']
            else:
                rewrite = data[i]['oracle_utt_text']
            if 'query' in data[i]:
                cur_query = data[i]['query']
            else:
                cur_query = data[i]['cur_utt_text']
            #if "answer" in data[i]:
            #    answer = data[i]["answer"]
            #else:
            #    answer = data[i]["cur_response_text"]
            #rewrite = rewrite + ' ' + answer
            if args.eval_type == "answer":
                data_2[i] = json.loads(data_2[i])
                rewrite = data_2[i]['answer_utt_text']
            elif args.eval_type == "oracle+answer":
                data_2[i] = json.loads(data_2[i])
                rewrite = rewrite + ' ' + data_2[i]['answer_utt_text']
            elif args.eval_type == "oracle+nexq":
                data_2[i] = json.loads(data_2[i])
                rewrite = rewrite + ' ' + data_2[i]['next_q_utt_text']

            rewrite = query_tokenizer.encode(rewrite, add_special_tokens=True)
            #query = query_tokenizer.encode(cur_query, add_special_tokens=True)

            self.examples.append(ConvExample_rewrite(sample_id,
                                        #query,
                                        rewrite,
                                        )) 


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @staticmethod
    def get_collate_fn(args):

        def collate_fn(batch: list):
            collated_dict = {
                "bt_sample_id": [],
                "bt_query":[],
                "bt_query_mask":[],
                "bt_rewrite":[],
                "bt_rewrite_mask":[],
            }
            
            bt_sample_id = [] 
            #bt_query = []
            #bt_query_mask = []
            bt_rewrite = []
            bt_rewrite_mask = []

            for example in batch:
                # padding
                #query, query_mask = pad_seq_ids_with_mask(example.query, max_length = args.max_query_length)
                rewrite, rewrite_mask = pad_seq_ids_with_mask(example.rewrite, max_length = args.max_concat_length)
                bt_sample_id.append(example.sample_id)
                #bt_query.append(query)
                #bt_query_mask.append(query_mask)  
                bt_rewrite.append(rewrite)
                bt_rewrite_mask.append(rewrite_mask)     

            collated_dict["bt_sample_id"] = bt_sample_id
            #collated_dict["bt_query"] = bt_query
            #collated_dict["bt_query_mask"] = bt_query_mask
            collated_dict["bt_rewrite"] = bt_rewrite
            collated_dict["bt_rewrite_mask"] = bt_rewrite_mask
            
            # change to tensor
            for key in collated_dict:
                if key not in ["bt_sample_id"]:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        return collate_fn



class ConvExample_cast:
    def __init__(self, sample_id,
                       conv_query, 
                       history_context,
                       pos_docs = None,
                       neg_docs = None,
                       raw_query = None,
                       oracle_query = None):
        self.sample_id = sample_id
        self.conv_query = conv_query
        self.history_context = history_context
        self.pos_docs = pos_docs
        self.neg_docs = neg_docs
        self.raw_query = raw_query
        self.oracle_query = oracle_query

class ConvDataset_cast(Dataset):
    def __init__(self, args, query_tokenizer, passage_tokenizer, filename, add_doc_info=True):
        self.examples = []
        
        with open(filename, 'r') as f:
            data = f.readlines()

        if args.use_PRF:
            with open(args.PRF_file, 'r') as f:
                PRF = f.readlines()

        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for testing
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        logger.info("Loading {} data file...".format(filename))
        for i in trange(n):
            # basic
            data[i] = json.loads(data[i])
            sample_id = data[i]['id']
            conv_id = data[i]['topic_number']
            turn_id = data[i]['query_number']
            history_query = data[i]["input"][:-1]
            query = data[i]["input"][-1]
            if int(turn_id) > 1 and int(conv_id) > 80:
                last_response = data[i - 1]["automatic_response"][-1]
            else:
                last_response = ""
            if args.use_PRF:
                PRF[i] = json.loads(PRF[i])
                rel_label = PRF[i]["rel_label"]

            # query
            conv_query = [] # q_i, q_i-1 ... q_2, q_1
            history_context = []
                
            cur_query = query_tokenizer.encode(query, add_special_tokens=True)
            conv_query.extend(cur_query)

            if len(last_response) > 0 and args.use_last_response:
                lp = []
                lp.append(query_tokenizer.cls_token_id)
                lp.extend(query_tokenizer.convert_tokens_to_ids(["<response>"]))
                lp.extend(query_tokenizer.convert_tokens_to_ids(query_tokenizer.tokenize(last_response)))
                lp = lp[:args.max_doc_length]
                lp.append(query_tokenizer.sep_token_id)
                conv_query.extend(lp)
                history_context.extend(lp)

            # context
            if not args.use_PRF:
                if len(history_query) > 0:
                    for j in range(len(history_query)-1, -1, -1):
                        conv_query.extend(query_tokenizer.encode(history_query[j], add_special_tokens=True, max_length=args.max_query_length))
                        history_context.extend(query_tokenizer.encode(history_query[j], add_special_tokens=True, max_length=args.max_query_length))
            else: # use PRF
                if len(rel_label) > 0:
                    if args.PRF_mode == "hard":
                        for j in range(len(rel_label)-1, -1, -1):
                            if rel_label[j] == 1:
                                conv_query.extend(query_tokenizer.encode(history_query[j], add_special_tokens=True, max_length=args.max_query_length))
                    elif args.PRF_mode == "soft":
                        for j in range(len(rel_label)-1, -1, -1):
                            if rel_label[j] == 1:
                                rel_q = []
                                rel_q.append(query_tokenizer.cls_token_id)
                                rel_q.extend(query_tokenizer.convert_tokens_to_ids(["<relevant>"]))
                                rel_q.extend(query_tokenizer.convert_tokens_to_ids(query_tokenizer.tokenize(history_query[j])))
                                rel_q = rel_q[:args.max_query_length]
                                rel_q.append(query_tokenizer.sep_token_id)
                                conv_query.extend(rel_q)
                            else:
                                irrel_q = []
                                irrel_q.append(query_tokenizer.cls_token_id)
                                irrel_q.extend(query_tokenizer.convert_tokens_to_ids(["<irrelevant>"]))
                                irrel_q.extend(query_tokenizer.convert_tokens_to_ids(query_tokenizer.tokenize(history_query[j])))
                                irrel_q = rel_q[:args.max_query_length]
                                irrel_q.append(query_tokenizer.sep_token_id)
                                conv_query.extend(irrel_q)

            # doc 
            pos_docs = []
            neg_docs = []
            pos_docs_id = []
            neg_docs_id = []
            if add_doc_info:
                #for doc in data[i]['automatic_response']:
                pos_docs.append(passage_tokenizer.encode(data[i]['automatic_response'][-1], add_special_tokens=True))
                pos_docs_id.append(data[i]['automatic_response_new_id'][-1])
                #seen_neg_docs = set()
                #for doc in data[i]['neg_docs']:
                #    if doc in data[i]['pos_docs'] or doc in seen_neg_docs:
                #        continue
                #    seen_neg_docs.add(doc)
                #    neg_docs.append(passage_tokenizer.encode(doc, add_special_tokens=True))   
                #    neg_docs_id = data[i]['neg_docs_id']
                # But if no neg_docs, at least add one
                #if len(neg_docs) == 0:
                #    neg_docs.append(passage_tokenizer.encode(doc , add_special_tokens=True, max_length=args.max_doc_length))

            # For baseline test
            raw_query = cur_query
            
            # if "oracle_query" in data[i]: 
            if "rewrite" in data[i]: 
                oracle_query = query_tokenizer.encode(data[i]['rewrite'], add_special_tokens=True)
            else:
                oracle_query = None

            self.examples.append(ConvExample_cast(sample_id,
                                            conv_query, 
                                            history_context,
                                            pos_docs,
                                            neg_docs,
                                            raw_query = raw_query,
                                            oracle_query = oracle_query))          

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


    @staticmethod
    def get_collate_fn(args, add_doc_info:bool, mode:str):

        def collate_fn(batch: list):
            collated_dict = {
                "bt_sample_id": [],
                "bt_conv_query":[],
                "bt_conv_query_mask":[],
                "bt_history_context":[],
                "bt_history_context_mask":[],
                "bt_raw_query":[],
                "bt_raw_query_mask":[],
                "bt_oracle_query":[],
                "bt_oracle_query_mask":[],
                "bt_pos_docs":[],
                "bt_pos_docs_mask":[],
                "bt_neg_docs":[],
                "bt_neg_docs_mask":[],
            }
            
            bt_sample_id = [] 
            bt_conv_query = []
            bt_conv_query_mask = []
            bt_history_context = []
            bt_history_context_mask = []
            bt_raw_query = []
            bt_raw_query_mask = []
            bt_oracle_query = []
            bt_oracle_query_mask = []
            
            # for doc
            bt_pos_docs = []
            bt_pos_docs_mask = []
            bt_neg_docs = []
            bt_neg_docs_mask = []

            for example in batch:
                # padding
                conv_query, conv_query_mask = pad_seq_ids_with_mask(example.conv_query, max_length = args.max_concat_length)
                history_context, history_context_mask = pad_seq_ids_with_mask(example.history_context, max_length = args.max_concat_length)

                if example.raw_query:
                    raw_query, raw_query_mask = pad_seq_ids_with_mask(example.raw_query, max_length = args.max_query_length)
                if example.oracle_query:
                    oracle_query, oracle_query_mask = pad_seq_ids_with_mask(example.oracle_query, max_length = args.max_query_length)


                bt_sample_id.append(example.sample_id)
                bt_conv_query.append(conv_query)
                bt_conv_query_mask.append(conv_query_mask)
                bt_history_context.append(history_context)
                bt_history_context_mask.append(history_context_mask)

                if example.raw_query:
                    bt_raw_query.append(raw_query)
                    bt_raw_query_mask.append(raw_query_mask)
                if example.oracle_query:
                    bt_oracle_query.append(oracle_query)
                    bt_oracle_query_mask.append(oracle_query_mask)
                
                if add_doc_info:
                    assert len(example.pos_docs) > 0
                    assert len(example.neg_docs) > 0
                    pos_doc = random.sample(example.pos_docs, 1)[0]
                    neg_doc = random.sample(example.neg_docs, 1)[0] # only one neg
                    pos_doc, pos_doc_mask = pad_seq_ids_with_mask(pos_doc, max_length = args.max_doc_length)
                    neg_doc, neg_doc_mask = pad_seq_ids_with_mask(neg_doc, max_length = args.max_doc_length)
    
                    bt_pos_docs.append(pos_doc)
                    bt_pos_docs_mask.append(pos_doc_mask)
                    bt_neg_docs.append(neg_doc)
                    bt_neg_docs_mask.append(neg_doc_mask)

            collated_dict["bt_sample_id"] = bt_sample_id
            collated_dict["bt_conv_query"] = bt_conv_query
            collated_dict["bt_conv_query_mask"] = bt_conv_query_mask
            collated_dict["bt_history_context"] = bt_history_context
            collated_dict["bt_history_context_mask"] = bt_history_context_mask
            collated_dict["bt_raw_query"] = bt_raw_query
            collated_dict["bt_raw_query_mask"] = bt_raw_query_mask
            collated_dict["bt_oracle_query"] = bt_oracle_query
            collated_dict["bt_oracle_query_mask"] = bt_oracle_query_mask

            collated_dict["bt_pos_docs"] = bt_pos_docs
            collated_dict["bt_pos_docs_mask"] = bt_pos_docs_mask
            collated_dict["bt_neg_docs"] = bt_neg_docs
            collated_dict["bt_neg_docs_mask"] = bt_neg_docs_mask
            
            # change to tensor
            for key in collated_dict:
                if key not in ["bt_sample_id", "bt_conv_id", "bt_turn_id"]:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        def collate_fn_test(batch: list):
            collated_dict = {
                "bt_sample_id": [],
                "bt_conv_query":[],
                "bt_conv_query_mask":[],
                "bt_history_context":[],
                "bt_history_context_mask":[],
                "bt_raw_query":[],
                "bt_raw_query_mask":[],
                "bt_oracle_query":[],
                "bt_oracle_query_mask":[],
            }

            # for query
            bt_sample_id = [] 
            bt_conv_query = []
            bt_conv_query_mask = []
            bt_history_context = []
            bt_history_context_mask = []
            bt_raw_query = []
            bt_raw_query_mask = []
            bt_oracle_query = []
            bt_oracle_query_mask = []

            for example in batch:
                # padding
                conv_query, conv_query_mask = pad_seq_ids_with_mask(example.conv_query, max_length = args.max_concat_length)
                history_context, history_context_mask = pad_seq_ids_with_mask(example.history_context, max_length = args.max_concat_length)
                if example.raw_query:
                    raw_query, raw_query_mask = pad_seq_ids_with_mask(example.raw_query, max_length = args.max_query_length)
                if example.oracle_query:
                    oracle_query, oracle_query_mask = pad_seq_ids_with_mask(example.oracle_query, max_length = args.max_query_length)

                bt_sample_id.append(example.sample_id)
                bt_conv_query.append(conv_query)
                bt_conv_query_mask.append(conv_query_mask)
                bt_history_context.append(history_context)
                bt_history_context_mask.append(history_context_mask)
                if example.raw_query:
                    bt_raw_query.append(raw_query)
                    bt_raw_query_mask.append(raw_query_mask)
                if example.oracle_query:
                    bt_oracle_query.append(oracle_query)
                    bt_oracle_query_mask.append(oracle_query_mask)

                            
            collated_dict["bt_sample_id"] = bt_sample_id
            collated_dict["bt_conv_query"] = bt_conv_query
            collated_dict["bt_conv_query_mask"] = bt_conv_query_mask
            collated_dict["bt_history_context"] = bt_history_context
            collated_dict["bt_history_context_mask"] = bt_history_context_mask
            collated_dict["bt_raw_query"] = bt_raw_query
            collated_dict["bt_raw_query_mask"] = bt_raw_query_mask
            
            # change to tensor
            for key in collated_dict:
                if key not in ["bt_sample_id", "bt_conv_id", "bt_turn_id"]:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
              
            return collated_dict

        if mode == "test":
            return collate_fn_test
        elif mode == "train":
            return collate_fn
        else:
            raise ValueError

class ConvExample_qrecc_old:
    def __init__(self, sample_id,
                       conv_query, 
                       conv_query_ans,
                       pos_docs = None,
                       neg_docs = None,
                       answer = None,
                       raw_query = None,
                       oracle_query = None):
        self.sample_id = sample_id
        self.conv_query = conv_query
        self.conv_query_ans = conv_query_ans
        self.pos_docs = pos_docs
        self.neg_docs = neg_docs
        self.answer = answer
        self.raw_query = raw_query
        self.oracle_query = oracle_query

class ConvDataset_qrecc_old(Dataset):
    def __init__(self, args, query_tokenizer, passage_tokenizer, filename, add_doc_info=True):
        self.examples = []
        
        with open(filename, 'r') as f:
            data = f.readlines()

        if args.use_PRF:
            with open(args.PRF_file, 'r') as f:
                PRF = f.readlines()

        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for testing
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        logger.info("Loading {} data file...".format(filename))
        for i in trange(n):
            # basic
            data[i] = json.loads(data[i])
            sample_id = data[i]['sample_id']
            history_query = data[i]["context_queries"]
            history_answer = data[i]["context_answers"]
            query = data[i]["query"]
            answer = data[i]["answer"]
            oracle = data[i]["oracle_query"]
            last_response = data[i]["last_response"]
            source = data[i]["source"]

            if add_doc_info and len(data[i]['pos_docs']) == 0: # bad passage
                continue

            if args.use_PRF:
                PRF[i] = json.loads(PRF[i])
                rel_label = PRF[i]["rel_label"]

            # query
            conv_query = [] # q_i, q_i-1 ... q_2, q_1
            conv_query_ans = [] # q_i, q_i-1 ... a_2, q_1, a_1
                
            cur_query = query_tokenizer.encode(query, add_special_tokens=True)
            conv_query.extend(cur_query)
            conv_query_ans.extend(cur_query)

            if len(last_response) > 0 and args.use_last_response and args.mode != "convqa":
                lp = []
                lp.append(query_tokenizer.cls_token_id)
                lp.extend(query_tokenizer.convert_tokens_to_ids(["<response>"]))
                lp.extend(query_tokenizer.convert_tokens_to_ids(query_tokenizer.tokenize(last_response)))
                lp = lp[:args.max_doc_length]
                lp.append(query_tokenizer.sep_token_id)
                conv_query.extend(lp)

            assert len((history_query)) == len((history_answer))
            if not args.use_PRF:
                if len(history_query) > 0:
                    for j in range(len(history_query)-1, -1, -1):
                        conv_query.extend(query_tokenizer.encode(history_query[j], add_special_tokens=True, max_length=args.max_query_length))
                        #if j < len(history_answer)
            else:
                if len(rel_label) > 0:
                    for j in range(len(rel_label)-1, -1, -1):
                        if rel_label[j] == 1:
                            conv_query.extend(query_tokenizer.encode(history_query[j], add_special_tokens=True, max_length=args.max_query_length))
                            conv_query_ans.extend(query_tokenizer.encode(history_query[j], add_special_tokens=True, max_length=args.max_query_length))
            # doc 
            pos_docs = []
            neg_docs = []
            pos_docs_id = []
            neg_docs_id = []
            if add_doc_info:
                for doc in data[i]['pos_docs']:
                    pos_docs.append(passage_tokenizer.encode(doc, add_special_tokens=True))
                    #pos_docs_id.append(data[i]['pos_docs_id'])
                #seen_neg_docs = set()
                for doc in data[i]['neg_docs']:
                #    if doc in data[i]['pos_docs'] or doc in seen_neg_docs:
                #        continue
                #    seen_neg_docs.add(doc)
                    neg_docs.append(passage_tokenizer.encode(doc, add_special_tokens=True))   
                    #neg_docs_id = data[i]['neg_docs_id']
                # But if no neg_docs, at least add one
                #if len(neg_docs) == 0:
                #    neg_docs.append(passage_tokenizer.encode(doc , add_special_tokens=True, max_length=args.max_doc_length))

            # For baseline test
            raw_query = cur_query
            
            # if "oracle_query" in data[i]: 
            #if "oracle_query" in data[i]: 
            oracle_query = query_tokenizer.encode(oracle, add_special_tokens=True)
            #else:
            #    oracle_query = None
            #embed()
            #input()
            self.examples.append(ConvExample_qrecc_old(sample_id,
                                            conv_query, 
                                            conv_query_ans, 
                                            pos_docs, 
                                            neg_docs,
                                            raw_query = raw_query,
                                            oracle_query = oracle_query))          
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


    @staticmethod
    def get_collate_fn(args, add_doc_info:bool, mode:str):

        def collate_fn(batch: list):
            collated_dict = {
                "bt_sample_id": [],
                "bt_conv_query":[],
                "bt_conv_query_mask":[],
                "bt_conv_query_ans":[],
                "bt_conv_query_ans_mask":[],
                "bt_raw_query":[],
                "bt_raw_query_mask":[],
                "bt_oracle_query":[],
                "bt_oracle_query_mask":[],
                "bt_pos_docs":[],
                "bt_pos_docs_mask":[],
                "bt_neg_docs":[],
                "bt_neg_docs_mask":[],
            }
            
            bt_sample_id = [] 
            bt_conv_query = []
            bt_conv_query_mask = []
            bt_conv_query_ans = []
            bt_conv_query_ans_mask = []
            bt_raw_query = []
            bt_raw_query_mask = []
            bt_oracle_query = []
            bt_oracle_query_mask = []
            
            # for doc
            bt_pos_docs = []
            bt_pos_docs_mask = []
            bt_neg_docs = []
            bt_neg_docs_mask = []


            for example in batch:
                # padding
                conv_query, conv_query_mask = pad_seq_ids_with_mask(example.conv_query, max_length = args.max_concat_length)
                conv_query_ans, conv_query_ans_mask = pad_seq_ids_with_mask(example.conv_query_ans, max_length = args.max_concat_length)

                #if example.raw_query:
                raw_query, raw_query_mask = pad_seq_ids_with_mask(example.raw_query, max_length = args.max_query_length)
                #if example.oracle_query:
                oracle_query, oracle_query_mask = pad_seq_ids_with_mask(example.oracle_query, max_length = args.max_query_length)

                bt_sample_id.append(example.sample_id)
                bt_conv_query.append(conv_query)
                bt_conv_query_mask.append(conv_query_mask)
                bt_conv_query_ans.append(conv_query_ans)
                bt_conv_query_ans_mask.append(conv_query_ans_mask)

                #if example.raw_query:
                bt_raw_query.append(raw_query)
                bt_raw_query_mask.append(raw_query_mask)
                #if example.oracle_query:
                bt_oracle_query.append(oracle_query)
                bt_oracle_query_mask.append(oracle_query_mask)
                
                if add_doc_info:
                    assert len(example.pos_docs) > 0
                    assert len(example.neg_docs) > 0
                    pos_doc = random.sample(example.pos_docs, 1)[0]
                    neg_doc = random.sample(example.neg_docs, 1)[0] # only one neg
                    pos_doc, pos_doc_mask = pad_seq_ids_with_mask(pos_doc, max_length = args.max_doc_length)
                    neg_doc, neg_doc_mask = pad_seq_ids_with_mask(neg_doc, max_length = args.max_doc_length)
    
                    bt_pos_docs.append(pos_doc)
                    bt_pos_docs_mask.append(pos_doc_mask)
                    bt_neg_docs.append(neg_doc)
                    bt_neg_docs_mask.append(neg_doc_mask)

            collated_dict["bt_sample_id"] = bt_sample_id
            collated_dict["bt_conv_query"] = bt_conv_query
            collated_dict["bt_conv_query_mask"] = bt_conv_query_mask
            collated_dict["bt_conv_query_ans"] = bt_conv_query_ans
            collated_dict["bt_conv_query_ans_mask"] = bt_conv_query_ans_mask
            collated_dict["bt_raw_query"] = bt_raw_query
            collated_dict["bt_raw_query_mask"] = bt_raw_query_mask
            collated_dict["bt_oracle_query"] = bt_oracle_query
            collated_dict["bt_oracle_query_mask"] = bt_oracle_query_mask

            collated_dict["bt_pos_docs"] = bt_pos_docs
            collated_dict["bt_pos_docs_mask"] = bt_pos_docs_mask
            collated_dict["bt_neg_docs"] = bt_neg_docs
            collated_dict["bt_neg_docs_mask"] = bt_neg_docs_mask
            # change to tensor
            for key in collated_dict:
                if key not in ["bt_sample_id"]:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        def collate_fn_test(batch: list):
            collated_dict = {
                "bt_sample_id": [],
                "bt_conv_query":[],
                "bt_conv_query_mask":[],
                "bt_conv_query_ans":[],
                "bt_conv_query_ans_mask":[],
                "bt_raw_query":[],
                "bt_raw_query_mask":[],
                "bt_oracle_query":[],
                "bt_oracle_query_mask":[],
            }

            # for query
            bt_sample_id = [] 
            bt_conv_query = []
            bt_conv_query_mask = []
            bt_conv_query_ans = []
            bt_conv_query_ans_mask = []
            bt_raw_query = []
            bt_raw_query_mask = []
            bt_oracle_query = []
            bt_oracle_query_mask = []

            for example in batch:
                # padding
                conv_query, conv_query_mask = pad_seq_ids_with_mask(example.conv_query, max_length = args.max_concat_length)
                conv_query_ans, conv_query_ans_mask = pad_seq_ids_with_mask(example.conv_query_ans, max_length = args.max_concat_length)
                
                #if example.raw_query:
                raw_query, raw_query_mask = pad_seq_ids_with_mask(example.raw_query, max_length = args.max_query_length)
                #if example.oracle_query:
                oracle_query, oracle_query_mask = pad_seq_ids_with_mask(example.oracle_query, max_length = args.max_query_length)

                #if example.raw_query:
                bt_raw_query.append(raw_query)
                bt_raw_query_mask.append(raw_query_mask)
                #if example.oracle_query:
                bt_oracle_query.append(oracle_query)
                bt_oracle_query_mask.append(oracle_query_mask)

                bt_sample_id.append(example.sample_id)
                bt_conv_query.append(conv_query)
                bt_conv_query_mask.append(conv_query_mask)
                bt_conv_query_ans.append(conv_query_ans)
                bt_conv_query_ans_mask.append(conv_query_ans_mask)
                            
            collated_dict["bt_sample_id"] = bt_sample_id
            collated_dict["bt_conv_query"] = bt_conv_query
            collated_dict["bt_conv_query_mask"] = bt_conv_query_mask
            collated_dict["bt_conv_query_ans"] = bt_conv_query_ans
            collated_dict["bt_conv_query_ans_mask"] = bt_conv_query_ans_mask
            collated_dict["bt_raw_query"] = bt_raw_query
            collated_dict["bt_raw_query_mask"] = bt_raw_query_mask
            collated_dict["bt_oracle_query"] = bt_oracle_query
            collated_dict["bt_oracle_query_mask"] = bt_oracle_query_mask
            
            # change to tensor
            for key in collated_dict:
                if key not in ["bt_sample_id"]:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
              
            return collated_dict

        if mode == "test":
            return collate_fn_test
        elif mode == "train":
            return collate_fn
        else:
            raise ValueError

class ConvExample_qrecc:
    def __init__(self, sample_id,
                       query,
                       conv_query, 
                       conv_context,
                       pos_docs = None,
                       neg_docs = None,
                       answer = None,
                       oracle_query = None):
        self.sample_id = sample_id
        self.query = query
        self.conv_query = conv_query
        self.conv_context = conv_context
        self.pos_docs = pos_docs
        self.neg_docs = neg_docs
        self.answer = answer
        self.oracle_query = oracle_query

class ConvDataset_qrecc(Dataset):
    def __init__(self, args, query_tokenizer, passage_tokenizer, filename, add_doc_info=True):
        self.examples = []
        
        with open(filename, 'r') as f:
            data = f.readlines()

        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for testing
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        logger.info("Loading {} data file...".format(filename))
        for i in trange(n):
            # basic
            data[i] = json.loads(data[i])
            sample_id = data[i]['sample_id']
            source = data[i]["source"]
            history_context = data[i]["ctx_utts_text"]
            query = data[i]["cur_utt_text"]
            answer = data[i]["cur_response_text"]
            oracle = data[i]["oracle_utt_text"]
            pos_pid = data[i]["pos_docs_pids"]
            neg_pid = data[i]["random_neg_docs_pids"]
            if "pos_docs_text" in data[i]:
                pos_docs_text = data[i]["pos_docs_text"]
            if "random_neg_docs_text" in data[i]:
                random_neg_docs_text = data[i]["random_neg_docs_text"]

            if len(pos_pid) == 0 or len(pos_docs_text) == 0: # bad passage
                continue

            # query
            conv_query = [] # q_i, q_i-1 ... q_2, q_1  
            conv_context = [] #q_1, q_2    
            cur_query = query_tokenizer.encode(query, add_special_tokens=True)
            oracle = query_tokenizer.encode(oracle, add_special_tokens=True)
            answer = query_tokenizer.encode(answer, add_special_tokens=True)
            conv_query.extend(cur_query)
            if len(history_context) > 0:
                for j in range(len(history_context)-1, -1, -1):
                    conv_query.extend(query_tokenizer.encode(history_context[j], add_special_tokens=True, max_length=args.max_query_length))
                for j in range(len(history_context)):
                    conv_context.extend(query_tokenizer.encode(history_context[j], add_special_tokens=True, max_length=args.max_query_length))


            # doc 
            if add_doc_info:
                for idx in range(len(pos_docs_text)):
                    pos_docs = []
                    neg_docs = []
                    #pos_docs_id = []
                    #neg_docs_id = []
                    pos_docs.extend(passage_tokenizer.encode(pos_docs_text[idx], add_special_tokens=True))
                    neg_docs.extend(passage_tokenizer.encode(random_neg_docs_text, add_special_tokens=True))
                    #pos_docs_id.append(pos_pid[idx])
                    seen_neg_docs = set()
                    #while len(neg_docs_id) == 0:
                        #pid = random.choice(neg_pid)
                        #if pid in pos_docs_id or pid in seen_neg_docs:
                        #    continue
                        #seen_neg_docs.add(pid)
                        #neg_docs.append(passage_tokenizer.encode(doc, add_special_tokens=True))   
                    self.examples.append(ConvExample_qrecc(sample_id,
                                    cur_query,
                                    conv_query, 
                                    conv_context,
                                    pos_docs, 
                                    neg_docs,
                                    answer,
                                    oracle))          

            else:
                self.examples.append(ConvExample_qrecc(sample_id,
                                                cur_query,
                                                conv_query, 
                                                conv_context,
                                                pos_docs, 
                                                neg_docs,
                                                answer,
                                                oracle))          
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


    @staticmethod
    def get_collate_fn(args, add_doc_info:bool, mode:str):

        def collate_fn(batch: list):
            collated_dict = {
                "bt_sample_id": [],
                "bt_conv_query":[],
                "bt_conv_query_mask":[],
                "bt_conv_context":[],
                "bt_conv_context_mask":[],
                "bt_raw_query":[],
                "bt_raw_query_mask":[],
                "bt_oracle_query":[],
                "bt_oracle_query_mask":[],
                "bt_answer":[],
                "bt_answer_mask":[],
                "bt_pos_docs":[],
                "bt_pos_docs_mask":[],
                "bt_neg_docs":[],
                "bt_neg_docs_mask":[],
            }
            
            bt_sample_id = [] 
            bt_conv_query = []
            bt_conv_query_mask = []
            bt_conv_context = []
            bt_conv_context_mask = []
            bt_raw_query = []
            bt_raw_query_mask = []
            bt_oracle_query = []
            bt_oracle_query_mask = []
            bt_answer = []
            bt_answer_mask = []
            
            # for doc
            bt_pos_docs = []
            bt_pos_docs_mask = []
            bt_neg_docs = []
            bt_neg_docs_mask = []


            for example in batch:
                # padding
                conv_query, conv_query_mask = pad_seq_ids_with_mask(example.conv_query, max_length = args.max_concat_length)
                conv_context, conv_context_mask = pad_seq_ids_with_mask(example.conv_context, max_length = args.max_concat_length)
                raw_query, raw_query_mask = pad_seq_ids_with_mask(example.query, max_length = args.max_query_length)
                oracle_query, oracle_query_mask = pad_seq_ids_with_mask(example.oracle_query, max_length = args.max_query_length)
                answer, answer_mask = pad_seq_ids_with_mask(example.answer, max_length = args.max_doc_length)
                
                bt_sample_id.append(example.sample_id)
                bt_conv_query.append(conv_query)
                bt_conv_query_mask.append(conv_query_mask)
                bt_conv_context.append(conv_context)
                bt_conv_context_mask.append(conv_context_mask)
                bt_answer.append(answer)
                bt_answer_mask.append(answer_mask)

                bt_raw_query.append(raw_query)
                bt_raw_query_mask.append(raw_query_mask)
                bt_oracle_query.append(oracle_query)
                bt_oracle_query_mask.append(oracle_query_mask)
                
                if add_doc_info:
                    assert len(example.pos_docs) > 0
                    assert len(example.neg_docs) > 0
                    pos_doc, pos_doc_mask = pad_seq_ids_with_mask(example.pos_docs, max_length = args.max_doc_length)
                    neg_doc, neg_doc_mask = pad_seq_ids_with_mask(example.neg_docs, max_length = args.max_doc_length)
    
                    bt_pos_docs.append(pos_doc)
                    bt_pos_docs_mask.append(pos_doc_mask)
                    bt_neg_docs.append(neg_doc)
                    bt_neg_docs_mask.append(neg_doc_mask)

            collated_dict["bt_sample_id"] = bt_sample_id
            collated_dict["bt_conv_query"] = bt_conv_query
            collated_dict["bt_conv_query_mask"] = bt_conv_query_mask
            collated_dict["bt_conv_context"] = bt_conv_context
            collated_dict["bt_conv_context_mask"] = bt_conv_context_mask
            collated_dict["bt_answer"] = bt_answer
            collated_dict["bt_answer_mask"] = bt_answer_mask
            collated_dict["bt_raw_query"] = bt_raw_query
            collated_dict["bt_raw_query_mask"] = bt_raw_query_mask
            collated_dict["bt_oracle_query"] = bt_oracle_query
            collated_dict["bt_oracle_query_mask"] = bt_oracle_query_mask

            collated_dict["bt_pos_docs"] = bt_pos_docs
            collated_dict["bt_pos_docs_mask"] = bt_pos_docs_mask
            collated_dict["bt_neg_docs"] = bt_neg_docs
            collated_dict["bt_neg_docs_mask"] = bt_neg_docs_mask
            # change to tensor
            for key in collated_dict:
                if key not in ["bt_sample_id"]:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            #embed()
            #input()
            return collated_dict

        def collate_fn_test(batch: list):
            collated_dict = {
                "bt_sample_id": [],
                "bt_conv_query":[],
                "bt_conv_query_mask":[],
                "bt_conv_context":[],
                "bt_conv_context_mask":[],
                "bt_raw_query":[],
                "bt_raw_query_mask":[],
                "bt_oracle_query":[],
                "bt_oracle_query_mask":[],
                "bt_answer":[],
                "bt_answer_mask":[],
            }

            # for query
            bt_sample_id = [] 
            bt_conv_query = []
            bt_conv_query_mask = []
            bt_conv_context = []
            bt_conv_context_mask = []
            bt_raw_query = []
            bt_raw_query_mask = []
            bt_oracle_query = []
            bt_oracle_query_mask = []
            bt_answer = []
            bt_answer_mask = []

            for example in batch:
                # padding
                conv_query, conv_query_mask = pad_seq_ids_with_mask(example.conv_query, max_length = args.max_concat_length)
                conv_context, conv_context_mask = pad_seq_ids_with_mask(example.conv_context, max_length = args.max_concat_length)
                raw_query, raw_query_mask = pad_seq_ids_with_mask(example.query, max_length = args.max_query_length)
                oracle_query, oracle_query_mask = pad_seq_ids_with_mask(example.oracle_query, max_length = args.max_query_length)
                answer, answer_mask = pad_seq_ids_with_mask(example.answer, max_length = args.max_doc_length)

                bt_sample_id.append(example.sample_id)
                bt_conv_query.append(conv_query)
                bt_conv_query_mask.append(conv_query_mask)
                bt_conv_context.append(conv_context)
                bt_conv_context_mask.append(conv_context_mask)
                bt_answer.append(answer)
                bt_answer_mask.append(answer_mask)

                bt_raw_query.append(raw_query)
                bt_raw_query_mask.append(raw_query_mask)
                bt_oracle_query.append(oracle_query)
                bt_oracle_query_mask.append(oracle_query_mask)

            collated_dict["bt_sample_id"] = bt_sample_id
            collated_dict["bt_conv_query"] = bt_conv_query
            collated_dict["bt_conv_query_mask"] = bt_conv_query_mask
            collated_dict["bt_conv_context"] = bt_conv_context
            collated_dict["bt_conv_context_mask"] = bt_conv_context_mask
            collated_dict["bt_answer"] = bt_answer
            collated_dict["bt_answer_mask"] = bt_answer_mask
            collated_dict["bt_raw_query"] = bt_raw_query
            collated_dict["bt_raw_query_mask"] = bt_raw_query_mask
            collated_dict["bt_oracle_query"] = bt_oracle_query
            collated_dict["bt_oracle_query_mask"] = bt_oracle_query_mask
            
            # change to tensor
            for key in collated_dict:
                if key not in ["bt_sample_id"]:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
              
            return collated_dict

        if mode == "test":
            return collate_fn_test
        elif mode == "train":
            return collate_fn
        else:
            raise ValueError

class T5RewriterIRDataset_qrecc(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for deugging
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        i = 0

        for line in tqdm(data):
            record = json.loads(line)
            flat_concat = []
            cur_utt_text = record['cur_utt_text']
            ctx_utts_text = record['ctx_utts_text']
            cur_response_text = record["cur_response_text"]
            oracle_utt_text = record["oracle_utt_text"]
            
            if "pos_docs_text" in record and "random_neg_docs_text" in record:
                pos_docs_text = record["pos_docs_text"]
                random_neg_docs_text = record["random_neg_docs_text"]
            else:
                continue
            
            if args.use_prefix:
                cur_utt_text = "question: " + cur_utt_text
                first_context = True
                
            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens = True, max_length = args.max_query_length)
            flat_concat.extend(cur_utt)
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if j % 2 == 1:
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length
                if args.use_prefix and first_context:
                    ctx_utts_text[j] = "context: " + ctx_utts_text[j]
                    first_context = False
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=max_length, truncation=True) # not remove [CLS]
                if len(flat_concat) + len(utt) > args.max_concat_length:
                    flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt) 

            flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat, max_pad_length = args.max_concat_length)

            if args.collate_fn_type == "flat_concat_for_train":
                if args.decode_type == "oracle":
                    target_seq = oracle_utt_text
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_query_length, truncation=True)    
                elif args.decode_type == "answer":
                    target_seq = cur_response_text
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_response_length, truncation=True)    
                elif args.decode_type == "next_q":
                    if (i + 1) != n:
                        next_record = json.loads(data[i + 1])
                        next_turn_id = str(next_record['sample_id'].strip().split('_')[-1])
                        if next_turn_id != '1':
                            next_query_text = next_record['cur_utt_text']
                        else:
                            next_query_text = cur_utt_text
                    else:
                        next_query_text = cur_utt_text
                    target_seq = next_query_text
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_query_length, truncation=True)    

                labels = target_encoding.input_ids
                labels = torch.tensor(labels)
                labels[labels == tokenizer.pad_token_id] = -100
                labels = labels.tolist()
                
                for idx in range(len(pos_docs_text)):
                    pos_docs = []
                    neg_docs = []
                    pos_docs.extend(tokenizer.encode(pos_docs_text[idx], add_special_tokens=True, max_length = args.max_doc_length))
                    neg_docs.extend(tokenizer.encode(random_neg_docs_text[0], add_special_tokens=True, max_length = args.max_doc_length))
                    pos_docs, pos_docs_mask = padding_seq_to_same_length(pos_docs, max_pad_length = args.max_doc_length)
                    neg_docs, neg_docs_mask = padding_seq_to_same_length(neg_docs, max_pad_length = args.max_doc_length)
                
                    self.examples.append([record['sample_id'], 
                                    flat_concat,
                                    flat_concat_mask,
                                    labels,
                                    cur_utt_text,
                                    oracle_utt_text,
                                    pos_docs,
                                    pos_docs_mask,
                                    neg_docs,
                                    neg_docs_mask])
                i += 1
            else:
                labels = []
                pos_docs = []
                neg_docs = []
                self.examples.append([record['sample_id'], 
                                        flat_concat,
                                        flat_concat_mask,
                                        labels,
                                        cur_utt_text,
                                        oracle_utt_text,
                                        pos_docs,
                                        pos_docs_mask,
                                        neg_docs,
                                        neg_docs_mask])

    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_input_ids": [],
                             "bt_attention_mask": [],
                             "bt_labels": [],
                             "bt_cur_utt_text": [],
                             "bt_oracle_utt_text": [],
                             "bt_pos_docs": [],
                             "bt_pos_docs_mask": [],
                             "bt_neg_docs": [],
                             "bt_neg_docs_mask": [],
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_input_ids"].append(example[1])
                collated_dict["bt_attention_mask"].append(example[2])
                collated_dict["bt_labels"].append(example[3])
                collated_dict["bt_cur_utt_text"].append(example[4])
                collated_dict["bt_oracle_utt_text"].append(example[5])
                collated_dict["bt_pos_docs"].append(example[6])
                collated_dict["bt_pos_docs_mask"].append(example[7])
                collated_dict["bt_neg_docs"].append(example[8])
                collated_dict["bt_neg_docs_mask"].append(example[9])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_oracle_utt_text"])
            if args.collate_fn_type == "flat_concat_for_test":
                not_need_to_tensor_keys.add("bt_labels")

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn

class T5RewriterDataset_qrecc(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()

        #with open(args.addtional_file, encoding="utf-8") as f:
        #    data_2 = f.readlines()

        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for deugging
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        last_record = None
        i = 0
        for line in tqdm(data):
            record = json.loads(line)
            # <CUR> oracle_query <CTX> cq_{k-1} <SEP> ... cq_{1} (<BOS>)
            flat_concat = []
            cur_utt_text = record['cur_utt_text']
            ctx_utts_text = record['ctx_utts_text']
            cur_response_text = record["cur_response_text"]
            oracle_utt_text = record["oracle_utt_text"]

            #record_2 = json.loads(data_2[i])
            #answer_utt_text_2 = record_2["answer_utt_text"][:-1]

            if args.use_prefix:
                cur_utt_text = "question: " + cur_utt_text
                first_context = True

            '''
            if args.use_last_response:
                turn_id = str(record['sample_id'].strip().split('_')[-1])
                if last_record != None and turn_id != '1':
                    last_response_text = last_record["cur_response_text"]
                else:
                    last_response_text = ""
                ora_utt = tokenizer.encode(oracle_utt_text, add_special_tokens = True, max_length = args.max_query_length)
                flat_concat.extend(ora_utt)
                if len(last_response_text) > 0:
                    last_response = tokenizer.encode(last_response_text, add_special_tokens = True, max_length = args.max_response_length)
                    if len(flat_concat) + len(last_response) > args.max_concat_length:
                        flat_concat += last_response[:args.max_concat_length - len(flat_concat) - 1] + [last_response[-1]] 
                    else:
                        flat_concat.extend(last_response)
            '''
            
            #ans_utt = tokenizer.encode(answer_utt_text_2, add_special_tokens = True, max_length = args.max_query_length)
            #flat_concat.extend(ans_utt)
            #ora_utt = tokenizer.encode(oracle_utt_text, add_special_tokens = True, max_length = args.max_query_length)
            #flat_concat.extend(ora_utt)
            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens = True, max_length = args.max_query_length)
            flat_concat.extend(cur_utt)
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if j % 2 == 1:
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length
                if args.use_prefix and first_context:
                    ctx_utts_text[j] = "context: " + ctx_utts_text[j]
                    first_context = False
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=max_length, truncation=True) # not remove [CLS]
                if len(flat_concat) + len(utt) > args.max_concat_length:
                    flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt) 
            
            flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat, max_pad_length = args.max_concat_length)

            if args.collate_fn_type == "flat_concat_for_train":
                if args.decode_type == "next_q":
                    if (i + 1) != n:
                        next_record = json.loads(data[i + 1])
                        next_turn_id = str(next_record['sample_id'].strip().split('_')[-1])
                        if next_turn_id != '1':
                            next_query_text = next_record['cur_utt_text']
                        else:
                            next_query_text = cur_utt_text
                    else:
                        next_query_text = cur_utt_text
                    oracle_target_seq = next_query_text
                    i += 1
                #if args.decode_type == "oracle":
                else:
                    oracle_target_seq = oracle_utt_text
                oracle_target_encoding = tokenizer(oracle_target_seq, padding="max_length", max_length=args.max_query_length, truncation=True)    
                #elif args.decode_type == "answer":
                answer_target_seq = cur_response_text
                answer_target_encoding = tokenizer(answer_target_seq, padding="max_length", max_length=args.max_query_length, truncation=True)    
                

                oracle_labels = oracle_target_encoding.input_ids
                oracle_labels_mask = oracle_target_encoding.attention_mask
                oracle_labels = torch.tensor(oracle_labels)
                oracle_labels[oracle_labels == tokenizer.pad_token_id] = -100
                oracle_labels = oracle_labels.tolist()

                answer_labels = answer_target_encoding.input_ids
                answer_labels_mask = answer_target_encoding.attention_mask
                answer_labels = torch.tensor(answer_labels)
                answer_labels[answer_labels == tokenizer.pad_token_id] = -100
                answer_labels = answer_labels.tolist()


                self.examples.append([record['sample_id'], 
                                flat_concat,
                                flat_concat_mask,
                                oracle_labels,
                                oracle_labels_mask,
                                answer_labels,
                                answer_labels_mask,
                                cur_utt_text,
                                oracle_utt_text,
                                ctx_utts_text
                                ])
            else:
                oracle_labels = []
                oracle_labels_mask = []
                answer_labels = []
                answer_labels_mask = []
                self.examples.append([record['sample_id'], 
                                flat_concat,
                                flat_concat_mask,
                                oracle_labels,
                                oracle_labels_mask,
                                answer_labels,
                                answer_labels_mask,
                                cur_utt_text,
                                oracle_utt_text,
                                ctx_utts_text])

            last_record = record
            i += 1

    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_input_ids": [],
                             "bt_attention_mask": [],
                             "bt_oracle_labels": [],
                             "bt_oracle_labels_mask": [],
                             "bt_answer_labels": [],
                             "bt_answer_labels_mask": [],
                             "bt_cur_utt_text": [],
                             "bt_oracle_utt_text": [],
                             "bt_ctx_utts_text":[],
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_input_ids"].append(example[1])
                collated_dict["bt_attention_mask"].append(example[2])
                collated_dict["bt_oracle_labels"].append(example[3])
                collated_dict["bt_oracle_labels_mask"].append(example[4])
                collated_dict["bt_answer_labels"].append(example[5])
                collated_dict["bt_answer_labels_mask"].append(example[6])
                collated_dict["bt_cur_utt_text"].append(example[7])
                collated_dict["bt_oracle_utt_text"].append(example[8])
                collated_dict["bt_ctx_utts_text"].append(example[9])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_oracle_utt_text", "bt_ctx_utts_text"])
            if args.collate_fn_type == "flat_concat_for_test":
                not_need_to_tensor_keys.add("bt_oracle_labels")
                not_need_to_tensor_keys.add("bt_oracle_labels_mask")
                not_need_to_tensor_keys.add("bt_answer_labels")
                not_need_to_tensor_keys.add("bt_answer_labels_mask")


            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn

class T5RewriterIRDataset_topiocqa(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for deugging
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        i = 0

        for line in tqdm(data):
            record = json.loads(line)
            flat_concat = []

            ctx_utts_text = []
            cur_utt_text = record['query']
            history_query = record['history_query']
            history_answer = record['history_answer']
            for i in range(len(history_query)):
                ctx_utts_text.append(history_query[i])
                ctx_utts_text.append(history_answer[i])
            cur_response_text = record["answer"]
            oracle_utt_text = record["rewrite"]

            #if "pos_docs" in record and len(record["pos_docs"]) != 0:
            #    pos_docs_text = record["pos_docs"]
            #    neg_docs_text = record["neg_docs"]
            #else:
            #    continue
            
            if args.use_prefix:
                cur_utt_text = "question: " + cur_utt_text
                first_context = True
                
            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens = True, max_length = args.max_query_length)
            flat_concat.extend(cur_utt)
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if j % 2 == 1:
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length
                if args.use_prefix and first_context:
                    ctx_utts_text[j] = "context: " + ctx_utts_text[j]
                    first_context = False
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=max_length, truncation=True) # not remove [CLS]
                if len(flat_concat) + len(utt) > args.max_concat_length:
                    flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt) 

            flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat, max_pad_length = args.max_concat_length)

            if args.collate_fn_type == "flat_concat_for_train":
                if args.decode_type == "oracle":
                    target_seq = oracle_utt_text
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_query_length, truncation=True)    
                elif args.decode_type == "answer":
                    target_seq = cur_response_text
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_response_length, truncation=True)    

                labels = target_encoding.input_ids
                labels = torch.tensor(labels)
                labels[labels == tokenizer.pad_token_id] = -100
                labels = labels.tolist()
                
                pos_docs = []
                neg_docs = []
                pos_docs.extend(tokenizer.encode(record["pos_docs"], add_special_tokens=True, max_length = args.max_doc_length))
                neg_docs.extend(tokenizer.encode(record["neg_docs"], add_special_tokens=True, max_length = args.max_doc_length))
                pos_docs, pos_docs_mask = padding_seq_to_same_length(pos_docs, max_pad_length = args.max_doc_length)
                neg_docs, neg_docs_mask = padding_seq_to_same_length(neg_docs, max_pad_length = args.max_doc_length)
            
                self.examples.append([record['id'], 
                                flat_concat,
                                flat_concat_mask,
                                labels,
                                cur_utt_text,
                                oracle_utt_text,
                                pos_docs,
                                pos_docs_mask,
                                neg_docs,
                                neg_docs_mask])
                i += 1
            else:
                labels = []
                pos_docs = []
                neg_docs = []
                self.examples.append([record['id'], 
                                        flat_concat,
                                        flat_concat_mask,
                                        labels,
                                        cur_utt_text,
                                        oracle_utt_text,
                                        pos_docs,
                                        pos_docs_mask,
                                        neg_docs,
                                        neg_docs_mask])

    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_input_ids": [],
                             "bt_attention_mask": [],
                             "bt_labels": [],
                             "bt_cur_utt_text": [],
                             "bt_oracle_utt_text": [],
                             "bt_pos_docs": [],
                             "bt_pos_docs_mask": [],
                             "bt_neg_docs": [],
                             "bt_neg_docs_mask": [],
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_input_ids"].append(example[1])
                collated_dict["bt_attention_mask"].append(example[2])
                collated_dict["bt_labels"].append(example[3])
                collated_dict["bt_cur_utt_text"].append(example[4])
                collated_dict["bt_oracle_utt_text"].append(example[5])
                collated_dict["bt_pos_docs"].append(example[6])
                collated_dict["bt_pos_docs_mask"].append(example[7])
                collated_dict["bt_neg_docs"].append(example[8])
                collated_dict["bt_neg_docs_mask"].append(example[9])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_oracle_utt_text"])
            if args.collate_fn_type == "flat_concat_for_test":
                not_need_to_tensor_keys.add("bt_labels")

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn

class T5RewriterDataset_topiocqa(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for deugging
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        for line in tqdm(data):
            record = json.loads(line)
            # <CUR> oracle_query <CTX> cq_{k-1} <SEP> ... cq_{1} (<BOS>)
            flat_concat = []
            ctx_utts_text = []
            cur_utt_text = record['query']
            history_query = record['history_query']
            history_answer = record['history_answer']
            for i in range(len(history_query)):
                ctx_utts_text.append(history_query[i])
                ctx_utts_text.append(history_answer[i])
            cur_response_text = record["answer"]
            oracle_utt_text = record["rewrite"]

            if args.use_prefix:
                cur_utt_text = "question: " + cur_utt_text
                first_context = True

            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens = True, max_length = args.max_query_length)
            flat_concat.extend(cur_utt)
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if j % 2 == 1:
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length
                if args.use_prefix and first_context:
                    ctx_utts_text[j] = "context: " + ctx_utts_text[j]
                    first_context = False
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length= max_length, truncation=True) # not remove [CLS]
                if len(flat_concat) + len(utt) > args.max_concat_length:
                    flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt) 

            flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat, max_pad_length = args.max_concat_length)

            if args.collate_fn_type == "flat_concat_for_train":
                if args.decode_type == "oracle":
                    target_seq = oracle_utt_text
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_query_length, truncation=True)    
                elif args.decode_type == "answer":
                    target_seq = cur_response_text
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_response_length, truncation=True)    
                labels = target_encoding.input_ids
                labels = torch.tensor(labels)
                labels[labels == tokenizer.pad_token_id] = -100
                labels = labels.tolist()
                '''
                for idx in range(len(pos_docs_text)):
                    pos_docs = []
                    neg_docs = []
                    pos_docs.extend(tokenizer.encode(pos_docs_text[idx], add_special_tokens=True, max_length = args.max_doc_length))
                    neg_docs.extend(tokenizer.encode(random_neg_docs_text[0], add_special_tokens=True, max_length = args.max_doc_length))
                '''
                self.examples.append([record['id'], 
                                flat_concat,
                                flat_concat_mask,
                                labels,
                                cur_utt_text,
                                oracle_utt_text])
            else:
                labels = []
                pos_docs = []
                neg_docs = []
                self.examples.append([record['id'], 
                                        flat_concat,
                                        flat_concat_mask,
                                        labels,
                                        cur_utt_text,
                                        oracle_utt_text])

    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_input_ids": [],
                             "bt_attention_mask": [],
                             "bt_labels": [],
                             "bt_cur_utt_text": [],
                             "bt_oracle_utt_text": []
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_input_ids"].append(example[1])
                collated_dict["bt_attention_mask"].append(example[2])
                collated_dict["bt_labels"].append(example[3])
                collated_dict["bt_cur_utt_text"].append(example[4])
                collated_dict["bt_oracle_utt_text"].append(example[5])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_oracle_utt_text"])
            if args.collate_fn_type == "flat_concat_for_test":
                not_need_to_tensor_keys.add("bt_labels")

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn

class T5RewriterDataset_cast(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for deugging
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        for line in tqdm(data):
            record = json.loads(line)
            # <CUR> oracle_query <CTX> cq_{k-1} <SEP> ... cq_{1} (<BOS>)
            flat_concat = []
            ctx_utts_text = record["input"][:-1]
            cur_utt_text = record["input"][-1]
            oracle_utt_text = record["target"]

            if args.use_prefix:
                cur_utt_text = "question: " + cur_utt_text
                first_context = True
            
            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens = True, max_length=args.max_query_length)
            flat_concat.extend(cur_utt)
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if args.use_prefix and first_context:
                    ctx_utts_text[j] = "context: " + ctx_utts_text[j]
                    first_context = False
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=args.max_query_length, truncation=True) # not remove [CLS]
                if len(flat_concat) + len(utt) > args.max_concat_length:
                    flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt) # QR and retrieval

            flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat, max_pad_length=args.max_concat_length)

            if args.collate_fn_type == "flat_concat_for_train":
                if args.decode_type == "oracle":
                    target_seq = oracle_utt_text
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_query_length, truncation=True)    
                elif args.decode_type == "answer":
                    target_seq = cur_response_text
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_response_length, truncation=True)    
                labels = target_encoding.input_ids
                labels = torch.tensor(labels)
                labels[labels == tokenizer.pad_token_id] = -100
                labels = labels.tolist()
                
            else:
                labels = []
                
                self.examples.append([record['id'], 
                                        flat_concat,
                                        flat_concat_mask,
                                        labels,
                                        cur_utt_text,
                                        oracle_utt_text])
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_input_ids": [],
                             "bt_attention_mask": [],
                             "bt_labels": [],
                             "bt_cur_utt_text": [],
                             "bt_oracle_utt_text": []
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_input_ids"].append(example[1])
                collated_dict["bt_attention_mask"].append(example[2])
                collated_dict["bt_labels"].append(example[3])
                collated_dict["bt_cur_utt_text"].append(example[4])
                collated_dict["bt_oracle_utt_text"].append(example[5])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_oracle_utt_text"])
            if args.collate_fn_type == "flat_concat_for_test":
                not_need_to_tensor_keys.add("bt_labels")

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn

class FiD_qrecc(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for deugging
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        question_prefix='question:'
        passage_prefix='context:'

        for line in tqdm(data):
            record = json.loads(line)
            # <CUR> oracle_query <CTX> cq_{k-1} <SEP> ... cq_{1} (<BOS>)
            flat_concat = []
            ctx_utts_text = record["input"][:-1]
            cur_utt_text = record["input"][-1]
            oracle_utt_text = record["target"]
            
            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens = True, max_length=args.max_query_length)
            flat_concat.extend(cur_utt)
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=args.max_query_length, truncation=True) # not remove [CLS]
                if len(flat_concat) + len(utt) > args.max_concat_length:
                    flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt) # QR and retrieval

            flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat, max_pad_length=args.max_concat_length)

            if args.collate_fn_type == "flat_concat_for_train":
                if args.decode_type == "oracle":
                    target_seq = oracle_utt_text
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_query_length, truncation=True)    
                elif args.decode_type == "answer":
                    target_seq = cur_response_text
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_response_length, truncation=True)    
                labels = target_encoding.input_ids
                labels = torch.tensor(labels)
                labels[labels == tokenizer.pad_token_id] = -100
                labels = labels.tolist()
                
            else:
                labels = []
                
                self.examples.append([record['id'], 
                                        flat_concat,
                                        flat_concat_mask,
                                        labels,
                                        cur_utt_text,
                                        oracle_utt_text])
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_input_ids": [],
                             "bt_attention_mask": [],
                             "bt_labels": [],
                             "bt_cur_utt_text": [],
                             "bt_oracle_utt_text": []
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_input_ids"].append(example[1])
                collated_dict["bt_attention_mask"].append(example[2])
                collated_dict["bt_labels"].append(example[3])
                collated_dict["bt_cur_utt_text"].append(example[4])
                collated_dict["bt_oracle_utt_text"].append(example[5])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_oracle_utt_text"])
            if args.collate_fn_type == "flat_concat_for_test":
                not_need_to_tensor_keys.add("bt_labels")

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn


class ConvDataset_QRExample:
    def __init__(self, sample_id, query, context, rewrite, pred_begin_pos):
        self.sample_id = sample_id
        self.query = query
        self.context = context
        self.rewrite = rewrite
        self.pred_begin_pos = pred_begin_pos

class ConvDataset_QR(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []

        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for deugging
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)


        for line in tqdm(data):
            record = json.loads(line)
            sample_id = record["sample_id"]
            query = record['cur_utt_text']
            input_sents = record['ctx_utts_text']
            target_sent = record["oracle_utt_text"]
            this_example = []
            this_example_labels = []

            for sent in input_sents:
                this_example.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)))
                this_example.append(tokenizer.sep_token_id)
            this_example.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(query)))
            #this_example.pop()
            this_example.append(tokenizer.bos_token_id)

            begin_pos = len(this_example)
            this_example_labels.extend([100] * begin_pos)

            this_example.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_sent)))
            this_example_labels.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_sent)))

            this_example.append(tokenizer.eos_token_id)
            this_example_labels.append(tokenizer.eos_token_id)

            if len(this_example) > args.block_size:
                this_example = this_example[:args.block_size]
                this_example_labels = this_example_labels[:args.block_size]
            else:
                pad_num = args.block_size - len(this_example)
                this_example.extend([tokenizer.pad_token_id] * pad_num)
                this_example_labels.extend([100] * pad_num)
            assert len(this_example) == args.block_size
            assert len(this_example_labels) == args.block_size
            self.examples.append(ConvDataset_QRExample(sample_id, query, this_example, this_example_labels, begin_pos))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

class ConvExample_cast_rel:
    def __init__(self, sample_id,
                        conv_id,
                        turn_id,
                        pair_query = None,
                        query_passage = None,
                        cur_query = None
                        ):
        self.sample_id = sample_id
        self.conv_id = conv_id
        self.turn_id = turn_id
        self.cur_query = cur_query
        self.pair_query = pair_query
        self.query_passage = query_passage


class ConvDataset_cast_rel(Dataset):
    def __init__(self, args, query_tokenizer, passage_tokenizer, filename, add_doc_info=True):
        self.examples = []
        
        with open(filename, 'r') as f:
            data = f.readlines()

        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for testing
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        logger.info("Loading {} data file...".format(filename))
        for i in trange(n):
            # basic
            data[i] = json.loads(data[i])
            sample_id = data[i]['id']
            conv_id = data[i]['conv_id']
            turn_id = data[i]['turn_id']
            query = data[i]["query"] # str
            #last_response = data[i]["last_response"]
            query_pair = data[i]["query_pair"] # str

            # query
            pair_query = []
            query_passage = []
            cur_query = query_tokenizer.encode(query, add_special_tokens=True)
            pair_query.extend(cur_query)
            query_passage.extend(cur_query)
            '''
            if args.use_last_response and len(last_response) > 0:
                lp = []
                lp.append(query_tokenizer.cls_token_id)
                lp.extend(query_tokenizer.convert_tokens_to_ids(["<response>"]))
                lp.extend(query_tokenizer.convert_tokens_to_ids(query_tokenizer.tokenize(last_response)))
                lp = lp[:args.max_doc_length]
                lp.append(query_tokenizer.sep_token_id)
                #lp = query_tokenizer.encode(last_response, add_special_tokens=True)
                pair_query.extend(lp)
                query_passage.extend(lp)
            '''
            if len(query_pair) > 0:
                turn_query = query_tokenizer.encode(query_pair, add_special_tokens=True)
                pair_query.extend(turn_query)

            self.examples.append(ConvExample_cast_rel(sample_id,
                                            conv_id,
                                            turn_id,
                                            pair_query,
                                            query_passage,
                                            cur_query,
                                            ))                     

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


    @staticmethod
    def get_collate_fn(args, add_doc_info:bool):

        def collate_fn(batch: list):
            collated_dict = {
                "bt_sample_id": [],
                "bt_conv_id": [],
                "bt_turn_id": [],
                "bt_query":[],
                "bt_query_mask":[],
                "bt_pair_query":[],
                "bt_pair_query_mask":[],
                "bt_query_passage":[],
                "bt_query_passage_mask":[],
            }
            
            bt_sample_id = [] 
            bt_conv_id = [] 
            bt_turn_id = [] 
            #bt_query = []
            #bt_query_mask = []
            bt_pair_query = []
            bt_pair_query_mask = []
            bt_query_passage = []
            bt_query_passage_mask = []

            for example in batch:
                # padding
                pair_query, pair_query_mask = pad_seq_ids_with_mask(example.pair_query, max_length = args.max_concat_length)
                #query_passage, query_passage_mask = pad_seq_ids_with_mask(example.query_passage, max_length = args.max_concat_length)
                bt_sample_id.append(example.sample_id)
                bt_conv_id.append(example.conv_id)
                bt_turn_id.append(example.turn_id)
                bt_pair_query.append(pair_query)
                bt_pair_query_mask.append(pair_query_mask)
                #bt_query_passage.append(query_passage)
                #bt_query_passage_mask.append(query_passage_mask)
                

            collated_dict["bt_sample_id"] = bt_sample_id
            collated_dict["bt_conv_id"] = bt_conv_id
            collated_dict["bt_turn_id"] = bt_turn_id
            collated_dict["bt_pair_query"] = bt_pair_query
            collated_dict["bt_pair_query_mask"] = bt_pair_query_mask
            #collated_dict["bt_query_passage"] = bt_query_passage
            #collated_dict["bt_query_passage_mask"] = bt_query_passage_mask
            
            # change to tensor
            for key in collated_dict:
                if key not in ["bt_sample_id", "bt_conv_id", "bt_turn_id"]:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        return collate_fn

class ConvExample_topiocqa_rel:
    def __init__(self, sample_id,
                        conv_id,
                        turn_id, 
                        pair_query = None,
                        query_passage = None,
                        cur_query = None
                        ):
        self.sample_id = sample_id
        self.conv_id = conv_id
        self.turn_id = turn_id
        self.cur_query = cur_query
        self.pair_query = pair_query
        self.query_passage = query_passage


class ConvDataset_topiocqa_rel(Dataset):
    def __init__(self, args, query_tokenizer, passage_tokenizer, filename, add_doc_info=True):
        self.examples = []
        
        with open(filename, 'r') as f:
            data = f.readlines()

        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for testing
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        logger.info("Loading {} data file...".format(filename))
        for i in trange(n):
            # basic
            data[i] = json.loads(data[i])
            sample_id = data[i]['id']
            conv_id = data[i]['conv_id']
            turn_id = data[i]['turn_id']
            #topic = data[i]["topic"]
            #sub_topic = data[i]["sub_topic"]
            query = data[i]["query"] # str
            #history_answer = data[i]["history_answer"]
            #last_response = data[i]["last_response"]
            #answer = data[i]["answer"]
            #turn_pair_id = data[i]['turn_pair_id']
            query_pair = data[i]["query_pair"] # str

            # query
            pair_query = []
            query_passage = []
            cur_query = query_tokenizer.encode(query, add_special_tokens=True)
            pair_query.extend(cur_query)
            query_passage.extend(cur_query)
            if args.use_last_response and len(last_response) > 0:
                lp = []
                lp.append(query_tokenizer.cls_token_id)
                lp.extend(query_tokenizer.convert_tokens_to_ids(["<response>"]))
                lp.extend(query_tokenizer.convert_tokens_to_ids(query_tokenizer.tokenize(last_response)))
                lp = lp[:args.max_doc_length]
                lp.append(query_tokenizer.sep_token_id)
                #lp = query_tokenizer.encode(last_response, add_special_tokens=True)
                pair_query.extend(lp)
                query_passage.extend(lp)
            if args.use_answer and len(history_answer) > 0:
                last_answer = query_tokenizer.encode(history_answer[-1], add_special_tokens=True)
                pair_query.extend(last_answer)
            if len(query_pair) > 0:
                turn_query = query_tokenizer.encode(query_pair, add_special_tokens=True)
                pair_query.extend(turn_query)

            self.examples.append(ConvExample_topiocqa_rel(sample_id,
                                            conv_id,
                                            turn_id, 
                                            pair_query,
                                            query_passage,
                                            cur_query,
                                            )) 

                                                     

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


    @staticmethod
    def get_collate_fn(args, add_doc_info:bool):

        def collate_fn(batch: list):
            collated_dict = {
                "bt_sample_id": [],
                "bt_conv_id": [],
                "bt_turn_id": [],
                "bt_query":[],
                "bt_query_mask":[],
                "bt_pair_query":[],
                "bt_pair_query_mask":[],
                "bt_query_passage":[],
                "bt_query_passage_mask":[],
            }
            
            bt_sample_id = [] 
            bt_conv_id = [] 
            bt_turn_id = [] 
            #bt_query = []
            #bt_query_mask = []
            bt_pair_query = []
            bt_pair_query_mask = []
            bt_query_passage = []
            bt_query_passage_mask = []

            for example in batch:
                # padding
                pair_query, pair_query_mask = pad_seq_ids_with_mask(example.pair_query, max_length = args.max_concat_length)
                query_passage, query_passage_mask = pad_seq_ids_with_mask(example.query_passage, max_length = args.max_concat_length)
                bt_sample_id.append(example.sample_id)
                bt_conv_id.append(example.conv_id)
                bt_turn_id.append(example.turn_id)
                bt_pair_query.append(pair_query)
                bt_pair_query_mask.append(pair_query_mask)
                bt_query_passage.append(query_passage)
                bt_query_passage_mask.append(query_passage_mask)
                

            collated_dict["bt_sample_id"] = bt_sample_id
            collated_dict["bt_conv_id"] = bt_conv_id
            collated_dict["bt_turn_id"] = bt_turn_id
            collated_dict["bt_pair_query"] = bt_pair_query
            collated_dict["bt_pair_query_mask"] = bt_pair_query_mask
            collated_dict["bt_query_passage"] = bt_query_passage
            collated_dict["bt_query_passage_mask"] = bt_query_passage_mask
            
            # change to tensor
            for key in collated_dict:
                if key not in ["bt_sample_id", "bt_conv_id", "bt_turn_id"]:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        return collate_fn

class ConvExample_qrecc_rel:
    def __init__(self, sample_id,
                        conv_id,
                        turn_id, 
                        pair_query = None,
                        cur_query = None
                        ):
        self.sample_id = sample_id
        self.conv_id = conv_id
        self.turn_id = turn_id
        self.cur_query = cur_query
        self.pair_query = pair_query


class ConvDataset_qrecc_rel(Dataset):
    def __init__(self, args, query_tokenizer, passage_tokenizer, filename, add_doc_info=True):
        self.examples = []
        
        with open(filename, 'r') as f:
            data = f.readlines()

        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for testing
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        logger.info("Loading {} data file...".format(filename))
        for i in trange(n):
            # basic
            data[i] = json.loads(data[i])
            sample_id = data[i]['id']
            conv_id = data[i]['conv_id']
            turn_id = data[i]['turn_id']
            query = data[i]["query"] # str
            last_response = data[i]["last_response"]
            query_pair = data[i]["query_pair"] # str

            # query
            pair_query = []
            cur_query = query_tokenizer.encode(query, add_special_tokens=True)
            pair_query.extend(cur_query)
            if args.use_last_response and len(last_response) > 0:
                lp = []
                lp.append(query_tokenizer.cls_token_id)
                lp.extend(query_tokenizer.convert_tokens_to_ids(["<response>"]))
                lp.extend(query_tokenizer.convert_tokens_to_ids(query_tokenizer.tokenize(last_response)))
                lp = lp[:args.max_doc_length]
                lp.append(query_tokenizer.sep_token_id)
                #lp = query_tokenizer.encode(last_response, add_special_tokens=True)
                pair_query.extend(lp)
            if len(query_pair) > 0:
                turn_query = query_tokenizer.encode(query_pair, add_special_tokens=True)
                pair_query.extend(turn_query)

            self.examples.append(ConvExample_topiocqa_rel(sample_id,
                                            conv_id,
                                            turn_id, 
                                            pair_query,
                                            cur_query,
                                            )) 

                                                     

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


    @staticmethod
    def get_collate_fn(args, add_doc_info:bool):

        def collate_fn(batch: list):
            collated_dict = {
                "bt_sample_id": [],
                "bt_conv_id": [],
                "bt_turn_id": [],
                "bt_query":[],
                "bt_query_mask":[],
                "bt_pair_query":[],
                "bt_pair_query_mask":[],
                "bt_query_passage":[],
                "bt_query_passage_mask":[],
            }
            
            bt_sample_id = [] 
            bt_conv_id = [] 
            bt_turn_id = [] 
            #bt_query = []
            #bt_query_mask = []
            bt_pair_query = []
            bt_pair_query_mask = []
            #bt_query_passage = []
            #bt_query_passage_mask = []

            for example in batch:
                # padding
                pair_query, pair_query_mask = pad_seq_ids_with_mask(example.pair_query, max_length = args.max_concat_length)
                #query_passage, query_passage_mask = pad_seq_ids_with_mask(example.query_passage, max_length = args.max_concat_length)
                bt_sample_id.append(example.sample_id)
                bt_conv_id.append(example.conv_id)
                bt_turn_id.append(example.turn_id)
                bt_pair_query.append(pair_query)
                bt_pair_query_mask.append(pair_query_mask)
                #bt_query_passage.append(query_passage)
                #bt_query_passage_mask.append(query_passage_mask)
                

            collated_dict["bt_sample_id"] = bt_sample_id
            collated_dict["bt_conv_id"] = bt_conv_id
            collated_dict["bt_turn_id"] = bt_turn_id
            collated_dict["bt_pair_query"] = bt_pair_query
            collated_dict["bt_pair_query_mask"] = bt_pair_query_mask
            #collated_dict["bt_query_passage"] = bt_query_passage
            #collated_dict["bt_query_passage_mask"] = bt_query_passage_mask
            
            # change to tensor
            for key in collated_dict:
                if key not in ["bt_sample_id", "bt_conv_id", "bt_turn_id"]:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        return collate_fn

class ConvExample:
    def __init__(self, sample_id, 
                       conv_query, 
                       cur_query_position = -1, 
                       pos_docs = None,
                       neg_docs = None,
                       raw_query = None,
                       oracle_query = None):
        self.sample_id = sample_id
        self.conv_query = conv_query
        self.cur_query_position = cur_query_position
        self.pos_docs = pos_docs
        self.neg_docs = neg_docs

        self.raw_query = raw_query
        self.oracle_query = oracle_query

class ConvDataset(Dataset):
    def __init__(self, args, query_tokenizer, passage_tokenizer, filename, add_doc_info=True):
        self.examples = []
        
        with open(filename, 'r') as f:
            data = f.readlines()

        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for testing
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        logger.info("Loading {} data file...".format(filename))
        for i in trange(n):
            # basic
            data[i] = json.loads(data[i])
            #sample_id = data[i]['sample_id']
            sample_id = data[i]['qid']
            conv_query = []

            
            # context
            #context_queries = data[i]['context_queries']
            context_queries = data[i]['input'][:-1] # a list
            #if args.use_gold_answer and data[i]["manual_response"]:
            #    gold_answer = data[i]["manual_response"]
            #    idx = 0
            for cq in context_queries:           
                conv_query.extend(query_tokenizer.encode(cq, add_special_tokens=True))
                #conv_query.extend(query_tokenizer.encode(gold_answer[idx], add_special_tokens=True))
                #idx += 1

            # last_response
            '''
            if args.last_response and data[i]["response"]:
                last_response = []
                last_response.append(query_tokenizer.cls_token_id)
                last_response.extend(query_tokenizer.convert_tokens_to_ids(["<response>"]))
                last_response.extend(query_tokenizer.convert_tokens_to_ids(query_tokenizer.tokenize(data[i]['last_response'])))
                last_response.append(query_tokenizer.sep_token_id)
    
                conv_query.extend(last_response)
            '''
            
            cur_query_position = len(conv_query)

            # query
            #query = query_tokenizer.encode(data[i]['query'] , add_special_tokens=True, max_length=args.max_query_length)
            query = query_tokenizer.encode(data[i]['input'][-1] , add_special_tokens=True, max_length=args.max_query_length)
            conv_query.extend(query)

            # doc info for ranking loss
            pos_docs = []
            neg_docs = []
            if add_doc_info:
                for doc in data[i]['pos_docs']:
                    pos_docs.append(passage_tokenizer.encode(doc , add_special_tokens=True, max_length=args.max_doc_length))
                seen_neg_docs = set()
                for doc in data[i]['neg_docs']:
                    if doc in data[i]['pos_docs'] or doc in seen_neg_docs:
                        continue
                    seen_neg_docs.add(doc)
                    neg_docs.append(passage_tokenizer.encode(doc , add_special_tokens=True, max_length=args.max_doc_length))   
                # But if no neg_docs, at least add one
                if len(neg_docs) == 0:
                    neg_docs.append(passage_tokenizer.encode(doc , add_special_tokens=True, max_length=args.max_doc_length))
            
            # For baseline test
            raw_query = query
            #if "oracle_query" in data[i]: 
            if "target" in data[i]: 
                oracle_query = query_tokenizer.encode(data[i]['target'][-1], add_special_tokens=True, max_length=args.max_query_length)
            else:
                oracle_query = None

            self.examples.append(ConvExample(sample_id, 
                                             conv_query, 
                                             cur_query_position, 
                                             pos_docs, 
                                             neg_docs,
                                             raw_query = raw_query,
                                             oracle_query = oracle_query))            

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

        
    @staticmethod
    def get_collate_fn(args, add_doc_info:bool, mode:str):

        def collate_fn(batch: list):
            collated_dict = {
                "bt_sample_id": [],
                "bt_conv_query":[],
                "bt_conv_query_mask":[],
                "bt_cur_query_position":[],
                "bt_raw_query":[],
                "bt_raw_query_mask":[],
                "bt_oracle_query":[],
                "bt_oracle_query_mask":[],
                "bt_pos_docs":[],
                "bt_pos_docs_mask":[],
                "bt_neg_docs":[],
                "bt_neg_docs_mask":[],
            }
            
            bt_sample_id = [] 
            bt_conv_query = []
            bt_conv_query_mask = []
            bt_cur_query_position = []

            bt_raw_query = []
            bt_raw_query_mask = []
            bt_oracle_query = []
            bt_oracle_query_mask = []
            
            # for doc
            bt_pos_docs = []
            bt_pos_docs_mask = []
            bt_neg_docs = []
            bt_neg_docs_mask = []


            for example in batch:
                # padding
                conv_query, conv_query_mask = pad_seq_ids_with_mask(example.conv_query, max_length = args.max_concat_length)
                if example.raw_query:
                    raw_query, raw_query_mask = pad_seq_ids_with_mask(example.raw_query, max_length = args.max_query_length)
                if example.oracle_query:
                    oracle_query, oracle_query_mask = pad_seq_ids_with_mask(example.oracle_query, max_length = args.max_query_length)
     
                bt_sample_id.append(example.sample_id)
                bt_conv_query.append(conv_query)
                bt_conv_query_mask.append(conv_query_mask)
                bt_cur_query_position.append(example.cur_query_position)
                
                if example.raw_query:
                    bt_raw_query.append(raw_query)
                    bt_raw_query_mask.append(raw_query_mask)
                if example.oracle_query:
                    bt_oracle_query.append(oracle_query)
                    bt_oracle_query_mask.append(oracle_query_mask)
                
                if add_doc_info:
                    assert len(example.pos_docs) > 0
                    assert len(example.neg_docs) > 0
                    pos_doc = random.sample(example.pos_docs, 1)[0]
                    neg_doc = random.sample(example.neg_docs, 1)[0] # BM25 hard negative
                    pos_doc, pos_doc_mask = pad_seq_ids_with_mask(pos_doc, max_length = max_doc_length)
                    neg_doc, neg_doc_mask = pad_seq_ids_with_mask(neg_doc, max_length = max_doc_length)
    
                    bt_pos_docs.append(pos_doc)
                    bt_pos_docs_mask.append(pos_doc_mask)
                    bt_neg_docs.append(neg_doc)
                    bt_neg_docs_mask.append(neg_doc_mask)

                
            collated_dict["bt_sample_id"] = bt_sample_id
            collated_dict["bt_conv_query"] = bt_conv_query
            collated_dict["bt_conv_query_mask"] = bt_conv_query_mask
            collated_dict["bt_cur_query_position"] = bt_cur_query_position
            collated_dict["bt_raw_query"] = bt_raw_query
            collated_dict["bt_raw_query_mask"] = bt_raw_query_mask
            collated_dict["bt_oracle_query"] = bt_oracle_query
            collated_dict["bt_oracle_query_mask"] = bt_oracle_query_mask

            collated_dict["bt_pos_docs"] = bt_pos_docs
            collated_dict["bt_pos_docs_mask"] = bt_pos_docs_mask
            collated_dict["bt_neg_docs"] = bt_neg_docs
            collated_dict["bt_neg_docs_mask"] = bt_neg_docs_mask
            
            # change to tensor
            for key in collated_dict:
                if key not in ["bt_sample_id"]:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
             
            return collated_dict

        def collate_fn_test_orquac(batch: list):
            collated_dict = {
                "bt_sample_id": [],
                "bt_conv_query":[],
                "bt_conv_query_mask":[],
                "bt_cur_query_position":[],
                "bt_doc":[],
                "bt_doc_mask":[],
                #"bt_pos_label_num":[],
                #"bt_retrieval_score_mask":[],
            }

            # for query
            bt_sample_id = [] 
            bt_conv_query = []
            bt_conv_query_mask = []
            bt_cur_query_position = []


            # for doc
            bt_doc = []
            bt_doc_mask = []
            #bt_pos_label_num = []
            #bt_retrieval_score_mask = []

            for example in batch:
                # padding
                conv_query, conv_query_mask = pad_seq_ids_with_mask(example.conv_query, max_length = args.max_concat_length)

                bt_sample_id.append(example.sample_id)
                bt_conv_query.append(conv_query)
                bt_conv_query_mask.append(conv_query_mask)
                bt_cur_query_position.append(example.cur_query_position)
                
                docs = []
                docs_mask = []
                '''
                pos_label_num = 0
                for pos_doc in example.pos_docs:
                    pos_doc, pos_doc_mask = pad_seq_ids_with_mask(pos_doc, max_length = args.max_concat_length)
                    docs.append(pos_doc)
                    docs_mask.append(pos_doc_mask)
                    pos_label_num += 1
                for neg_doc in example.neg_docs:
                    neg_doc, neg_doc_mask = pad_seq_ids_with_mask(neg_doc, max_length = args.max_concat_length)
                    docs.append(neg_doc)
                    docs_mask.append(neg_doc_mask)
                '''
                
                bt_doc.append(torch.tensor(docs))
                bt_doc_mask.append(torch.tensor(docs_mask))
                #bt_pos_label_num.append(pos_label_num)
                #bt_retrieval_score_mask.append(torch.zeros((len(docs), 1)))
            
            # pad doc number
    
            bt_doc = pad_sequence(bt_doc, batch_first = True)   # B * max_doc_num * seq_len
            bt_doc_mask = pad_sequence(bt_doc_mask, batch_first = True)
            #bt_retrieval_score_mask = pad_sequence(bt_retrieval_score_mask, batch_first=True, padding_value=-np.inf)
            
            collated_dict["bt_sample_id"] = bt_sample_id
            collated_dict["bt_conv_query"] = bt_conv_query
            collated_dict["bt_conv_query_mask"] = bt_conv_query_mask
            collated_dict["bt_cur_query_position"] = bt_cur_query_position

            collated_dict["bt_doc"] = bt_doc
            collated_dict["bt_doc_mask"] = bt_doc_mask
            #collated_dict["bt_retrieval_score_mask"] = bt_retrieval_score_mask
            #collated_dict["bt_pos_label_num"] = bt_pos_label_num
            
            # change to tensor
            for key in collated_dict:
                #if key not in ["bt_sample_id", "bt_doc", "bt_doc_mask", "bt_pos_label_num", "bt_retrieval_score_mask"]:
                if key not in ["bt_sample_id", "bt_doc", "bt_doc_mask"]:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
              
            return collated_dict

        if mode == "test_orquac":
            return collate_fn_test_orquac
        elif mode in ["train_orquac", "train_cast", "test_cast"]:
            return collate_fn
        else:
            raise ValueError


def pad_seq_ids_with_mask(input_ids,
                            max_length,
                            pad_on_left=False,
                            pad_token=0):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length

    attention_mask = []

    if padding_length <= 0:
        input_ids = input_ids[-max_length:]
        attention_mask = [1] * max_length
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
        else:
            attention_mask = [1] * len(input_ids) + [0] * padding_length
            input_ids = input_ids + padding_id

    assert len(input_ids) == max_length
    assert len(attention_mask) == max_length

    return input_ids, attention_mask

def padding_seq_to_same_length(input_ids, max_pad_length, pad_token = 0):
    padding_length = max_pad_length - len(input_ids)
    padding_ids = [pad_token] * padding_length
    attention_mask = []

    if padding_length <= 0:
        attention_mask = [1] * max_pad_length
        input_ids = input_ids[:max_pad_length]
    else:
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids = input_ids + padding_ids
            
    assert len(input_ids) == max_pad_length
    assert len(attention_mask) == max_pad_length
  
    return input_ids, attention_mask

from re import T
from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')
import argparse
import os
from utils import check_dir_exist_or_build
from os import path
from os.path import join as oj
import toml
import numpy as np
import json
from pyserini.search.lucene import LuceneSearcher
import pytrec_eval

def main():
    args = get_args()
    
    query_list = []
    qid_list = []
    with open(args.input_query_path, "r") as f:
        data = f.readlines()
        total_nums = len(data)
    query_times = 0
    for i, line in enumerate(data):
        line = json.loads(line)
        if len(line["query_pair"]) > 0 and int(line["id"][-1]) > 0: # query_pair
            query_times += 1
            if args.query_type == "raw":
                query = line["query"] + ' ' + line['query_pair']
            elif args.query_type == "rewrite":
                query = line['rewrite'] + ' ' + line['rewrite_query_pair']
        else: # one query
            query_times += 1
            if args.query_type == "raw":
                query = line["query"]
            elif args.query_type == "rewrite":
                query = line['rewrite'] 
        
        if args.use_last_response:
            query = query + ' ' + line['last_response']

        if args.use_answer:
            turn_id = int(line["turn_id"])
            if turn_id > 2:
                query = query + ' ' + line['history_answer'][-1]         
        query_list.append(query)
        qid_list.append(line['id'])
    assert total_nums == query_times
   
    # pyserini search
    searcher = LuceneSearcher(args.index_dir_path)
    searcher.set_bm25(args.bm25_k1, args.bm25_b)
    hits = searcher.batch_search(query_list, qid_list, k = args.top_k, threads = 20)

    merged_data = []
    with open(oj(args.output_dir_path, "test_rel_res.trec"), "w") as f:
        for qid, query in zip(qid_list, query_list):
            #if int(qid[-1]) > 0:
            #    input_type = "pair"
            #else:
            #    input_type = "center"
            rank_list = []
            for i, item in enumerate(hits[qid]):
                f.write("{} {} {} {} {} {}".format(qid,
                                                "Q0",
                                                item.docid[3:],
                                                i+1,
                                                -i - 1 + 200, #item.score,
                                                #input_type,
                                                "bm25"
                                                ))
                f.write('\n')

    logger.info("output file write ok")

    res = print_res(oj(args.output_dir_path, "test_rel_res.trec"), args.gold_qrel_file_path, args.rel_threshold, args.input_query_path, args.ori_qrel_file_path)
    return res  


def print_res(run_file, qrel_file, rel_threshold, input_query_file, ori_qrel_file):
    with open(run_file, 'r' )as f:
        run_data = f.readlines()
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    
    qrels = {}
    qrels_ndcg = {}
    runs = {}
    
    for line in qrel_data:
        line = line.strip().split()
        query = line[0]
        passage = line[2]
        rel = int(line[3])
        if query not in qrels:
            qrels[query] = {}
        if query not in qrels_ndcg:
            qrels_ndcg[query] = {}

        # for NDCG
        qrels_ndcg[query][passage] = rel
        # for MAP, MRR, Recall
        if rel >= rel_threshold:
            rel = 1
        else:
            rel = 0
        qrels[query][passage] = rel
    
    for line in run_data:
        line = line.split(" ")
        query = line[0]
        passage = line[2]
        rel = int(line[4])
        if query not in runs:
            runs[query] = {}
        runs[query][passage] = rel

    # pytrec_eval eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.5", "recall.20", "recall.100"})
    #evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.5"})

    res = evaluator.evaluate(runs)
    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    recall_100_list = [v['recall_100'] for v in res.values()]
    recall_20_list = [v['recall_20'] for v in res.values()]
    recall_5_list = [v['recall_5'] for v in res.values()]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3"})
    res = evaluator.evaluate(runs)
    ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]

    mrr_one_nums = 0
    mrr_zero_nums = 0
    res_mrr_dict = improve_judge(input_query_file, mrr_list, ori_qrel_file)
    for key, value in res_mrr_dict.items():
        if (len(value) > 0 and 1 in value[1:]) or len(value) == 1:
            mrr_one_nums += 1
        elif len(value) > 0 and 1 not in value[1:]:
            mrr_zero_nums += 1
    print(mrr_one_nums) #  
    print(mrr_zero_nums) # 

    with open(oj("output/qrecc/bm25", "test_rel_label_rawq_1.json"), "w") as f:
        for key, value in res_mrr_dict.items():
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

    return res_mrr_dict

def improve_judge(input_query_file, score_list, ori_qrel_file):
    qrel_id = []
    with open(ori_qrel_file, "r") as f:
        qrel_data = f.readlines()
        for line in qrel_data:
            line = line.strip().split()
            query = line[0]
            if query not in qrel_id:
                qrel_id.append(query) 

    with open(input_query_file, "r") as f:
        data = f.readlines()
    rel_label = {}
    rel_list = []
    base_score = 0
    for i, line in enumerate(data):
        line = json.loads(line)
        id_list = line["id"].split('-')
        conv_id = int(id_list[0])
        turn_id = int(id_list[1])
        type_id = int(id_list[-1])

        if (i + 1) != len(data):
            next_turn_id = int(json.loads(data[i + 1])["id"].split('-')[1])
            next_conv_id = int(json.loads(data[i + 1])["id"].split('-')[0])

        #if type_id == 0 and turn_id == 1:
        #    rel_list = []
        if type_id == 0 and turn_id > 1: 
            base_score = score_list[i]
        elif type_id > 0 and turn_id > 1: 
            if score_list[i] > base_score:
                rel_list.append(1)
            else:
                rel_list.append(0)
        
        if (i + 1) == len(data) or turn_id != next_turn_id or (turn_id == next_turn_id and conv_id != next_conv_id):
            if (str(conv_id) + '-1') in qrel_id:
                rel_label[id_list[0] + '-1'] = []
            #rel_label[id_list[0] + '-2'] = [1]
            #rel_list.insert(0, 1)
            rel_label[id_list[0] + '-' + id_list[1]] = rel_list
            rel_list = []
            base_score = 0

    return rel_label
            
        
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config",
                        type = str,
                        required = True,
                        help = "Config file path.")

    args = parser.parse_args()
    config = toml.load(args.config)
    args = argparse.Namespace(**config)

    check_dir_exist_or_build([args.output_dir_path])
    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)
    return args

    
if __name__ == '__main__':
    main()

# python bm25_rel_qrecc.py --config Config/bm25_rel_qrecc.toml
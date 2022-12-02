import argparse
import os
import json
from tqdm import tqdm
import pickle
from IPython import embed
import csv
import random
from collections import defaultdict


dev_19 = "datasets/cast19/eval_topics.jsonl"
dev_rel_19 = "datasets/cast19/dev_rel.json"
dev_trec_gold_19 = "datasets/cast19/qrels.tsv"
dev_rel_gold_19 = "datasets/cast19/dev_rel_gold.trec"
dev_20 = "datasets/cast20/eval_topics.jsonl"
dev_rel_20 = "datasets/cast20/dev_rel.json"
dev_rel_gold_20 = "datasets/cast20/dev_rel_gold.trec"
dev_trec_gold_20 = "datasets/cast20/qrels.tsv"

dev_19_pos = "datasets/cast19/eval_topics_with_doc.jsonl"
dev_20_pos = "datasets/cast20/eval_topics_with_doc.jsonl"
collection = "datasets/cast20/collection.tsv"



def create_label_rel_turn(inputs, output):
    with open(inputs, "r") as f, open(output, "w") as g:
        obj = f.readlines()
        total_nums = len(obj)
        query_pair_nums = 0
        for i in range(total_nums):
            obj[i] = json.loads(obj[i])
            sample_id = obj[i]['id']
            conv_id = obj[i]['topic_number']
            turn_id = obj[i]['query_number']
            history_query = obj[i]["input"][:-1]
            query = obj[i]["input"][-1]
            rewrite = obj[i]["target"]
            last_response = ""
            pos_docs = []
            pos_docs_id = []
            if int(turn_id) > 1 and int(conv_id) > 80:
                last_response = obj[i - 1]["manual_response"][-1]
            if int(conv_id) > 80:
                pos_docs.append(obj[i]['manual_response'][-1])
                pos_docs_id.append(obj[i]['manual_response_new_id'][-1])

            if int(turn_id) > 1: 
                # if first turn
                g.write(
                    json.dumps({
                        "id": str(conv_id) + '-' + str(turn_id) + '-0',
                        "conv_id": conv_id,
                        "turn_id": turn_id,
                        "query": query,
                        "query_pair": "",
                        #"last_response": last_response,
                        #"pos_docs": pos_docs,
                        #"pos_docs_id": pos_docs_id,
                    }) + "\n")
                query_pair_nums += 1

                for tid in range(0, int(turn_id) - 1):
                    query_pair = history_query[tid]
                    g.write(
                        json.dumps({
                            "id": str(conv_id) + '-' + str(turn_id) + '-' + str(tid + 1),
                            "conv_id": conv_id,
                            "turn_id": turn_id,
                            "query": query,
                            "query_pair": query_pair,
                            #"last_response": last_response,
                            #"pos_docs": pos_docs,
                            #"pos_docs_id": pos_docs_id,
                        }) + "\n")
                    query_pair_nums += 1
        print(query_pair_nums)

def create_rel_gold_trec(query_file, rel_file, gold_trec_file, rel_gole_trec_file, rel_threshold):
    with open(query_file, "r") as f, open(gold_trec_file, "r") as g, open(rel_gole_trec_file, "w") as h:
        query_data = f.readlines()
        for line in query_data:
            line = json.loads(line)
            conv_id = line["topic_number"]
            turn_id = line["query_number"]

        gold_trec_data = g.readlines()
        seen_id = []
        for line in gold_trec_data:
            line_list = line.strip().split('\t')
            qid, doc_id, qrel = line_list[0], line_list[2], line_list[3]
            conv_id = qid.split('-')[0]
            turn_id = qid.split('-')[1]
            #if turn_id == "1" or int(qrel) < rel_threshold:
            if turn_id == "1":
                continue
            for i in range(int(turn_id)):
                sample_id = qid + '-' + str(i)
                h.write("{} {} {} {}".format(sample_id,
                                        "Q0",
                                        doc_id,
                                        qrel,
                                        ))
                h.write('\n')
                if sample_id not in seen_id:
                    seen_id.append(sample_id)
        print(len(seen_id))
        
        with open(rel_file, 'r') as f:
            file_id = []
            for line in f:
                line = json.loads(line)
                sample_id = line["id"]
                file_id.append(sample_id)
            for qid in file_id:
                if qid not in seen_id:
                    h.write("{} {} {} {}".format(qid,
                                            "Q0",
                                            12734590, # random
                                            0,
                                            ))
                    h.write('\n')
                if qid not in seen_id:
                    print(qid)
                    seen_id.append(qid)
        
        print(len(seen_id))
    print("finish")

def create_filter_cast20(label_file, query_file, output):
    with open(label_file, "r") as f1, open(query_file, "r") as f2, open(output, "w") as g:
        obj_1 = f1.readlines()
        obj_2 = f2.readlines()
        one = 0
        zero = 0
        total_nums = len(obj_1)
        assert len(obj_1) == len(obj_2)
        for i in range(total_nums):
            obj_1[i] = json.loads(obj_1[i])
            obj_2[i] = json.loads(obj_2[i])
            sample_id = obj_2[i]['id']
            conv_id = obj_1[i]['conv_id']
            turn_id = obj_1[i]['turn_id']
            rel_label = obj_1[i]['rel_label']
            cur_query = obj_2[i]['input'][-1]
            history_query = obj_2[i]["input"][:-1]
            last_response = obj_2[i]["last_response"]
            assert len(history_query) == len(rel_label)
            if len(history_query) > 0:
                for idx in range(len(history_query)):
                    #if rel_label[idx] == 1:
                    #    one += 1
                    #else:
                    #    zero += 1
                    g.write(
                        json.dumps({
                            "id": sample_id + '-' + str(idx + 1),
                            "conv_id": conv_id,
                            "turn_id": turn_id,
                            "query": cur_query,
                            "rel_query": history_query[idx],
                            "rel_label": rel_label[idx],
                            "last_response": last_response
                        }) + "\n")
        #print("one", one)
        #print("zero", zero)

def add_doc(collection, input_file, gold_file, output_file, rel_threshold):
    with open(input_file, "r") as f:
        data = f.readlines()

    with open(gold_file, "r") as g:
        gold = g.readlines()


    qid2goldpid = defaultdict(list)
    for line in gold:
        qid, _, pid, rel = line.strip().split()
        if int(rel) >= int(rel_threshold):
            qid2goldpid[qid].append(int(pid))
    g.close()

    pid2doc = {}
    for line in tqdm(open(collection, "r")):
        try:
            pid, doc = line.strip().split('\t')
            pid = int(pid)
        except:
            pid = int(line.strip().split('\t')[0])
            doc = ""
            continue
        #    bad_doc_set.add(pid)
        #if pid in needed_pids_set:
        pid2doc[pid] = doc

    with open(output_file, 'w') as fw:
        for line in data:
            line = json.loads(line)
            sample_id = line["id"]
            pos_docs_id = []
            if sample_id in qid2goldpid:
                pos_docs_id = qid2goldpid[sample_id]
            else:
                pos_docs_id = []

            pos_docs_text = []
            for pid in pos_docs_id:
                if pid in pid2doc:
                    pos_docs_text.append(pid2doc[int(pid)])
            #embed()
            #input()
            #pos_docs_text = modify_pos_docs(line, pos_docs_text)
            line["pos_docs_id"] = pos_docs_id
            line["pos_docs_text"] = pos_docs_text
            fw.write(json.dumps(line))
            fw.write('\n')

if __name__ == "__main__":
    #create_label_rel_turn(dev_19, dev_rel_19)
    #create_label_rel_turn(dev_20, dev_rel_20)
    #create_rel_gold_trec(dev_19, dev_rel_19, dev_trec_gold_19, dev_rel_gold_19, 1)
    #create_rel_gold_trec(dev_20, dev_rel_20, dev_trec_gold_20, dev_rel_gold_20, 2)
    #create_filter_cast20("output/cast20/dense_rel/dev_rel_label_rawq.json", dev_20, "filter/data/cast20_dev_q.json")
    #add_doc(collection, dev_19, dev_trec_gold_19, dev_19_pos, 1)
    add_doc(collection, dev_20, dev_trec_gold_20, dev_20_pos, 2)
    '''
    with open(dev_rel_20, 'r') as f:
        file_id = []
        for line in f:
            line = json.loads(line)
            sample_id = line["id"]
            file_id.append(sample_id)
        id_list = ['81-2-0', '81-2-1', '81-3-0', '81-3-1', '81-3-2', '81-4-0', '81-4-1', '81-4-2', '81-4-3', '81-5-0', '81-5-1', '81-5-2', '81-5-3', '81-5-4', '81-6-0', '81-6-1', '81-6-2', '81-6-3', '81-6-4', '81-6-5', '81-7-0', '81-7-1', '81-7-2', '81-7-3', '81-7-4', '81-7-5', '81-7-6', '81-8-0', '81-8-1', '81-8-2', '81-8-3', '81-8-4', '81-8-5', '81-8-6', '81-8-7', '82-2-0', '82-2-1', '82-3-0', '82-3-1', '82-3-2', '82-4-0', '82-4-1', '82-4-2', '82-4-3', '82-5-0', '82-5-1', '82-5-2', '82-5-3', '82-5-4', '82-6-0', '82-6-1', '82-6-2', '82-6-3', '82-6-4', '82-6-5', '82-7-0', '82-7-1', '82-7-2', '82-7-3', '82-7-4', '82-7-5', '82-7-6', '82-8-0', '82-8-1', '82-8-2', '82-8-3', '82-8-4', '82-8-5', '82-8-6', '82-8-7', '82-9-0', '82-9-1', '82-9-2', '82-9-3', '82-9-4', '82-9-5', '82-9-6', '82-9-7', '82-9-8', '82-10-0', '82-10-1', '82-10-2', '82-10-3', '82-10-4', '82-10-5', '82-10-6', '82-10-7', '82-10-8', '82-10-9', '83-2-0', '83-2-1', '83-3-0', '83-3-1', '83-3-2', '83-4-0', '83-4-1', '83-4-2', '83-4-3', '83-5-0', '83-5-1', '83-5-2', '83-5-3', '83-5-4', '83-6-0', '83-6-1', '83-6-2', '83-6-3', '83-6-4', '83-6-5', '83-7-0', '83-7-1', '83-7-2', '83-7-3', '83-7-4', '83-7-5', '83-7-6', '83-8-0', '83-8-1', '83-8-2', '83-8-3', '83-8-4', '83-8-5', '83-8-6', '83-8-7', '84-2-0', '84-2-1', '84-3-0', '84-3-1', '84-3-2', '84-4-0', '84-4-1', '84-4-2', '84-4-3', '84-5-0', '84-5-1', '84-5-2', '84-5-3', '84-5-4', '84-6-0', '84-6-1', '84-6-2', '84-6-3', '84-6-4', '84-6-5', '85-2-0', '85-2-1', '85-3-0', '85-3-1', '85-3-2', '85-4-0', '85-4-1', '85-4-2', '85-4-3', '85-5-0', '85-5-1', '85-5-2', '85-5-3', '85-5-4', '85-6-0', '85-6-1', '85-6-2', '85-6-3', '85-6-4', '85-6-5', '85-7-0', '85-7-1', '85-7-2', '85-7-3', '85-7-4', '85-7-5', '85-7-6', '85-8-0', '85-8-1', '85-8-2', '85-8-3', '85-8-4', '85-8-5', '85-8-6', '85-8-7', '85-9-0', '85-9-1', '85-9-2', '85-9-3', '85-9-4', '85-9-5', '85-9-6', '85-9-7', '85-9-8', '86-2-0', '86-2-1', '86-3-0', '86-3-1', '86-3-2', '86-4-0', '86-4-1', '86-4-2', '86-4-3', '86-5-0', '86-5-1', '86-5-2', '86-5-3', '86-5-4', '86-6-0', '86-6-1', '86-6-2', '86-6-3', '86-6-4', '86-6-5', '86-7-0', '86-7-1', '86-7-2', '86-7-3', '86-7-4', '86-7-5', '86-7-6', '87-2-0', '87-2-1', '87-3-0', '87-3-1', '87-3-2', '87-4-0', '87-4-1', '87-4-2', '87-4-3', '87-5-0', '87-5-1', '87-5-2', '87-5-3', '87-5-4', '87-7-0', '87-7-1', '87-7-2', '87-7-3', '87-7-4', '87-7-5', '87-7-6', '87-8-0', '87-8-1', '87-8-2', '87-8-3', '87-8-4', '87-8-5', '87-8-6', '87-8-7', '87-9-0', '87-9-1', '87-9-2', '87-9-3', '87-9-4', '87-9-5', '87-9-6', '87-9-7', '87-9-8', '88-2-0', '88-2-1', '88-3-0', '88-3-1', '88-3-2', '88-4-0', '88-4-1', '88-4-2', '88-4-3', '88-5-0', '88-5-1', '88-5-2', '88-5-3', '88-5-4', '88-6-0', '88-6-1', '88-6-2', '88-6-3', '88-6-4', '88-6-5', '88-7-0', '88-7-1', '88-7-2', '88-7-3', '88-7-4', '88-7-5', '88-7-6', '88-8-0', '88-8-1', '88-8-2', '88-8-3', '88-8-4', '88-8-5', '88-8-6', '88-8-7', '88-9-0', '88-9-1', '88-9-2', '88-9-3', '88-9-4', '88-9-5', '88-9-6', '88-9-7', '88-9-8', '88-10-0', '88-10-1', '88-10-2', '88-10-3', '88-10-4', '88-10-5', '88-10-6', '88-10-7', '88-10-8', '88-10-9', '89-2-0', '89-2-1', '89-3-0', '89-3-1', '89-3-2', '89-4-0', '89-4-1', '89-4-2', '89-4-3', '89-5-0', '89-5-1', '89-5-2', '89-5-3', '89-5-4', '89-6-0', '89-6-1', '89-6-2', '89-6-3', '89-6-4', '89-6-5', '89-7-0', '89-7-1', '89-7-2', '89-7-3', '89-7-4', '89-7-5', '89-7-6', '89-8-0', '89-8-1', '89-8-2', '89-8-3', '89-8-4', '89-8-5', '89-8-6', '89-8-7', '89-9-0', '89-9-1', '89-9-2', '89-9-3', '89-9-4', '89-9-5', '89-9-6', '89-9-7', '89-9-8', '89-10-0', '89-10-1', '89-10-2', '89-10-3', '89-10-4', '89-10-5', '89-10-6', '89-10-7', '89-10-8', '89-10-9', '89-11-0', '89-11-1', '89-11-2', '89-11-3', '89-11-4', '89-11-5', '89-11-6', '89-11-7', '89-11-8', '89-11-9', '89-11-10', '90-2-0', '90-2-1', '90-3-0', '90-3-1', '90-3-2', '90-4-0', '90-4-1', '90-4-2', '90-4-3', '90-5-0', '90-5-1', '90-5-2', '90-5-3', '90-5-4', '90-6-0', '90-6-1', '90-6-2', '90-6-3', '90-6-4', '90-6-5', '90-7-0', '90-7-1', '90-7-2', '90-7-3', '90-7-4', '90-7-5', '90-7-6', '90-8-0', '90-8-1', '90-8-2', '90-8-3', '90-8-4', '90-8-5', '90-8-6', '90-8-7', '91-2-0', '91-2-1', '91-3-0', '91-3-1', '91-3-2', '91-4-0', '91-4-1', '91-4-2', '91-4-3', '91-5-0', '91-5-1', '91-5-2', '91-5-3', '91-5-4', '91-6-0', '91-6-1', '91-6-2', '91-6-3', '91-6-4', '91-6-5', '91-7-0', '91-7-1', '91-7-2', '91-7-3', '91-7-4', '91-7-5', '91-7-6', '91-8-0', '91-8-1', '91-8-2', '91-8-3', '91-8-4', '91-8-5', '91-8-6', '91-8-7', '92-2-0', '92-2-1', '92-3-0', '92-3-1', '92-3-2', '92-4-0', '92-4-1', '92-4-2', '92-4-3', '92-5-0', '92-5-1', '92-5-2', '92-5-3', '92-5-4', '92-6-0', '92-6-1', '92-6-2', '92-6-3', '92-6-4', '92-6-5', '92-7-0', '92-7-1', '92-7-2', '92-7-3', '92-7-4', '92-7-5', '92-7-6', '93-2-0', '93-2-1', '93-3-0', '93-3-1', '93-3-2', '93-4-0', '93-4-1', '93-4-2', '93-4-3', '93-5-0', '93-5-1', '93-5-2', '93-5-3', '93-5-4', '93-6-0', '93-6-1', '93-6-2', '93-6-3', '93-6-4', '93-6-5', '94-2-0', '94-2-1', '94-3-0', '94-3-1', '94-3-2', '94-4-0', '94-4-1', '94-4-2', '94-4-3', '94-5-0', '94-5-1', '94-5-2', '94-5-3', '94-5-4', '94-6-0', '94-6-1', '94-6-2', '94-6-3', '94-6-4', '94-6-5', '94-7-0', '94-7-1', '94-7-2', '94-7-3', '94-7-4', '94-7-5', '94-7-6', '94-8-0', '94-8-1', '94-8-2', '94-8-3', '94-8-4', '94-8-5', '94-8-6', '94-8-7', '95-2-0', '95-2-1', '95-3-0', '95-3-1', '95-3-2', '95-4-0', '95-4-1', '95-4-2', '95-4-3', '95-5-0', '95-5-1', '95-5-2', '95-5-3', '95-5-4', '95-6-0', '95-6-1', '95-6-2', '95-6-3', '95-6-4', '95-6-5', '95-7-0', '95-7-1', '95-7-2', '95-7-3', '95-7-4', '95-7-5', '95-7-6', '95-8-0', '95-8-1', '95-8-2', '95-8-3', '95-8-4', '95-8-5', '95-8-6', '95-8-7', '96-3-0', '96-3-1', '96-3-2', '96-4-0', '96-4-1', '96-4-2', '96-4-3', '96-5-0', '96-5-1', '96-5-2', '96-5-3', '96-5-4', '96-6-0', '96-6-1', '96-6-2', '96-6-3', '96-6-4', '96-6-5', '96-7-0', '96-7-1', '96-7-2', '96-7-3', '96-7-4', '96-7-5', '96-7-6', '96-8-0', '96-8-1', '96-8-2', '96-8-3', '96-8-4', '96-8-5', '96-8-6', '96-8-7', '97-2-0', '97-2-1', '97-3-0', '97-3-1', '97-3-2', '97-4-0', '97-4-1', '97-4-2', '97-4-3', '97-5-0', '97-5-1', '97-5-2', '97-5-3', '97-5-4', '97-6-0', '97-6-1', '97-6-2', '97-6-3', '97-6-4', '97-6-5', '97-7-0', '97-7-1', '97-7-2', '97-7-3', '97-7-4', '97-7-5', '97-7-6', '97-8-0', '97-8-1', '97-8-2', '97-8-3', '97-8-4', '97-8-5', '97-8-6', '97-8-7', '98-2-0', '98-2-1', '98-3-0', '98-3-1', '98-3-2', '98-4-0', '98-4-1', '98-4-2', '98-4-3', '98-5-0', '98-5-1', '98-5-2', '98-5-3', '98-5-4', '98-6-0', '98-6-1', '98-6-2', '98-6-3', '98-6-4', '98-6-5', '98-7-0', '98-7-1', '98-7-2', '98-7-3', '98-7-4', '98-7-5', '98-7-6', '98-8-0', '98-8-1', '98-8-2', '98-8-3', '98-8-4', '98-8-5', '98-8-6', '98-8-7', '99-2-0', '99-2-1', '99-3-0', '99-3-1', '99-3-2', '99-4-0', '99-4-1', '99-4-2', '99-4-3', '99-5-0', '99-5-1', '99-5-2', '99-5-3', '99-5-4', '99-6-0', '99-6-1', '99-6-2', '99-6-3', '99-6-4', '99-6-5', '99-7-0', '99-7-1', '99-7-2', '99-7-3', '99-7-4', '99-7-5', '99-7-6', '99-8-0', '99-8-1', '99-8-2', '99-8-3', '99-8-4', '99-8-5', '99-8-6', '99-8-7', '100-2-0', '100-2-1', '100-3-0', '100-3-1', '100-3-2', '100-4-0', '100-4-1', '100-4-2', '100-4-3', '100-5-0', '100-5-1', '100-5-2', '100-5-3', '100-5-4', '100-6-0', '100-6-1', '100-6-2', '100-6-3', '100-6-4', '100-6-5', '100-7-0', '100-7-1', '100-7-2', '100-7-3', '100-7-4', '100-7-5', '100-7-6', '100-8-0', '100-8-1', '100-8-2', '100-8-3', '100-8-4', '100-8-5', '100-8-6', '100-8-7', '101-2-0', '101-2-1', '101-3-0', '101-3-1', '101-3-2', '101-4-0', '101-4-1', '101-4-2', '101-4-3', '101-5-0', '101-5-1', '101-5-2', '101-5-3', '101-5-4', '101-6-0', '101-6-1', '101-6-2', '101-6-3', '101-6-4', '101-6-5', '101-7-0', '101-7-1', '101-7-2', '101-7-3', '101-7-4', '101-7-5', '101-7-6', '101-8-0', '101-8-1', '101-8-2', '101-8-3', '101-8-4', '101-8-5', '101-8-6', '101-8-7', '101-9-0', '101-9-1', '101-9-2', '101-9-3', '101-9-4', '101-9-5', '101-9-6', '101-9-7', '101-9-8', '101-10-0', '101-10-1', '101-10-2', '101-10-3', '101-10-4', '101-10-5', '101-10-6', '101-10-7', '101-10-8', '101-10-9', '102-2-0', '102-2-1', '102-3-0', '102-3-1', '102-3-2', '102-4-0', '102-4-1', '102-4-2', '102-4-3', '102-5-0', '102-5-1', '102-5-2', '102-5-3', '102-5-4', '102-6-0', '102-6-1', '102-6-2', '102-6-3', '102-6-4', '102-6-5', '102-7-0', '102-7-1', '102-7-2', '102-7-3', '102-7-4', '102-7-5', '102-7-6', '102-8-0', '102-8-1', '102-8-2', '102-8-3', '102-8-4', '102-8-5', '102-8-6', '102-8-7', '102-9-0', '102-9-1', '102-9-2', '102-9-3', '102-9-4', '102-9-5', '102-9-6', '102-9-7', '102-9-8', '103-2-0', '103-2-1', '103-3-0', '103-3-1', '103-3-2', '103-4-0', '103-4-1', '103-4-2', '103-4-3', '103-5-0', '103-5-1', '103-5-2', '103-5-3', '103-5-4', '103-6-0', '103-6-1', '103-6-2', '103-6-3', '103-6-4', '103-6-5', '103-8-0', '103-8-1', '103-8-2', '103-8-3', '103-8-4', '103-8-5', '103-8-6', '103-8-7', '103-9-0', '103-9-1', '103-9-2', '103-9-3', '103-9-4', '103-9-5', '103-9-6', '103-9-7', '103-9-8', '103-10-0', '103-10-1', '103-10-2', '103-10-3', '103-10-4', '103-10-5', '103-10-6', '103-10-7', '103-10-8', '103-10-9', '104-3-0', '104-3-1', '104-3-2', '104-4-0', '104-4-1', '104-4-2', '104-4-3', '104-6-0', '104-6-1', '104-6-2', '104-6-3', '104-6-4', '104-6-5', '104-7-0', '104-7-1', '104-7-2', '104-7-3', '104-7-4', '104-7-5', '104-7-6', '104-8-0', '104-8-1', '104-8-2', '104-8-3', '104-8-4', '104-8-5', '104-8-6', '104-8-7', '104-9-0', '104-9-1', '104-9-2', '104-9-3', '104-9-4', '104-9-5', '104-9-6', '104-9-7', '104-9-8', '104-10-0', '104-10-1', '104-10-2', '104-10-3', '104-10-4', '104-10-5', '104-10-6', '104-10-7', '104-10-8', '104-10-9', '104-12-0', '104-12-1', '104-12-2', '104-12-3', '104-12-4', '104-12-5', '104-12-6', '104-12-7', '104-12-8', '104-12-9', '104-12-10', '104-12-11', '104-13-0', '104-13-1', '104-13-2', '104-13-3', '104-13-4', '104-13-5', '104-13-6', '104-13-7', '104-13-8', '104-13-9', '104-13-10', '104-13-11', '104-13-12', '105-2-0', '105-2-1', '105-3-0', '105-3-1', '105-3-2', '105-4-0', '105-4-1', '105-4-2', '105-4-3', '105-5-0', '105-5-1', '105-5-2', 
        '105-5-3', '105-5-4', '105-6-0', '105-6-1', '105-6-2', '105-6-3', '105-6-4', '105-6-5', '105-7-0', '105-7-1', '105-7-2', '105-7-3', '105-7-4', '105-7-5', '105-7-6', '105-8-0', '105-8-1', '105-8-2', '105-8-3', '105-8-4', '105-8-5', '105-8-6', '105-8-7', '105-9-0', '105-9-1', '105-9-2', '105-9-3', '105-9-4', '105-9-5', '105-9-6', '105-9-7', '105-9-8']
        for key in file_id:
            if key not in id_list:
                print(key)
    '''
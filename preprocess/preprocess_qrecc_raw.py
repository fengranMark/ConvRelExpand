import argparse
import os
import json
from tqdm import tqdm, trange
import pickle
from IPython import embed
import csv
import random
import gc


def process_dir(dir_path, pid, pid2rawpid, fw):
    filenames = os.listdir(dir_path)
    for filename in tqdm(filenames):
        with open(os.path.join(dir_path, filename), "r") as f:
            data = f.readlines()
        for line in data:
            line = json.loads(line)
            raw_pid = line["id"]
            passage = line["contents"]
            pid2rawpid.append(raw_pid)
            fw.write("{}\t{}".format(pid, passage))
            fw.write("\n")
            
            pid += 1
    
    return pid, pid2rawpid



def extract_passage_form_directory():
    passage_dir = "collection-paragraph"
    pdir1 = os.path.join(passage_dir, "commoncrawl")
    pdir2 = os.path.join(passage_dir, "wayback")
    pdir3 = os.path.join(passage_dir, "wayback-backfill")
    
    pid = 0
    pid2rawpid = []

    with open("qrecc_collection.tsv", "w") as fw:
        pid, pid2rawpid = process_dir(pdir1, pid, pid2rawpid, fw)
        print("{} process ok!".format(pdir1))
        pid, pid2rawpid = process_dir(pdir2, pid, pid2rawpid, fw)
        print("{} process ok!".format(pdir2))
        pid, pid2rawpid = process_dir(pdir3, pid, pid2rawpid, fw)
        print("{} process ok!".format(pdir3))


    print("#totoal passages = {}".format(pid))

    with open("pid2rawpid.pkl", "wb") as f:
        pickle.dump(pid2rawpid, f, protocol=pickle.HIGHEST_PROTOCOL) 
    print("store pid2rawpid ok!")




def gen_qrecc_qrel():
    with open("scai-qrecc21-test-turns.json", "r") as f:
        data = json.load(f)

    with open("pid2rawpid.pkl", "rb") as f:
        pid2rawpid = pickle.load(f)
    print("load pid2rawpid ok!")

    d = {}
    for i in trange(len(pid2rawpid)):
        d[pid2rawpid[i]] = i

    with open("qrecc_qrel.tsv", "w") as f:
        for line in tqdm(data):
            sample_id = "{}-{}".format(line['Conversation_no'], line['Turn_no'])
            for rawpid in line['Truth_passages']:
                f.write("{}\t{}\t{}\t{}".format(sample_id, 0, d[rawpid], 1))
                f.write('\n') 

def gen_train_test_file():

    with open("pid2rawpid.pkl", "rb") as f:
        pid2rawpid = pickle.load(f)
    print("load pid2rawpid ok!")

    d = {}
    for i in trange(len(pid2rawpid)):
        d[pid2rawpid[i]] = i  # d = {rawid: pid}

    need_pids = set()
    last_conv_id = 1
    last_response = ""
    
    # train file
    with open("scai-qrecc21-training-turns.json", "r") as f:
        data = json.load(f)
        total_nums = len(data)
        print(total_nums)
    with open("train.json", "w") as f:
        for i in range(total_nums):
        #for line in tqdm(data):
            #sample_id = "{}_{}".format(line['Conversation_no'], line['Turn_no'])
            sample_id = "{}-{}".format(data[i]['Conversation_no'], data[i]['Turn_no'])
            context_queries = []
            context_answers = []
            if i > 0 and data[i]['Turn_no'] > 1:
                for j in range(1, data[i]['Turn_no']):
                    context_queries.append(data[i - j]['Question'])
                    context_answers.append(data[i - j]['Truth_answer'])
            #for i in range(0, len(line['Context']), 2):
            #    context_queries.append(line['Context'][i])
                #if len(line['Context'][i + 1]) > 0 and len(line['Context']) > 0:
                #    context_answers.append(line['Context'][i + 1])
                #else:
                #    context_answers.append("")
            # last_answer = ""
            #if len(line['Context']) > 0:
                # last_answer = line['Context'][-1]
            #    for i in range(1, len(line['Context']), 2):
            #        context_answers.append(line['Context'][i])
            oracle_query = data[i]["Truth_rewrite"]
            pos_docs = [] 
            for rawpid in data[i]['Truth_passages']:
                need_pids.add(d[rawpid])
                pos_docs.append(d[rawpid])

            if last_conv_id != data[i]['Conversation_no']:
                last_response = ""

            record = {}
            record["sample_id"] = sample_id
            record["query"] = data[i]["Question"]
            record["context_queries"] = context_queries
            record["context_answers"] = context_answers
            #record["last_answer"] = last_answer
            record["last_response"] = last_response
            record["oracle_query"] = oracle_query
            record["source"] = data[i]["Conversation_source"]
            record["pos_docs"] = pos_docs
            record["neg_docs"] = []

            last_conv_id = data[i]['Conversation_no']
            if len(data[i]['Truth_passages']) > 0:
                last_response = d[data[i]['Truth_passages'][0]]
            else:
                last_response = ""
        
            f.write(json.dumps(record))
            f.write('\n')
    print("train file ok")


    # test file
    with open("scai-qrecc21-test-turns.json", "r") as f:
        data = json.load(f)
        total_nums = len(data)
        print(total_nums)

    last_conv_id = 1
    last_response = ""

    with open("test.json", "w") as f:
        for i in range(total_nums):
        #for line in tqdm(data):
            #sample_id = "{}_{}".format(line['Conversation_no'], line['Turn_no'])
            sample_id = "{}-{}".format(data[i]['Conversation_no'], data[i]['Turn_no'])
            context_queries = []
            context_answers = []
            if i > 0 and data[i]['Turn_no'] > 1:
                for j in range(1, data[i]['Turn_no']):
                    context_queries.append(data[i - j]['Question'])
                    context_answers.append(data[i - j]['Truth_answer'])
            #for i in range(0, len(line['Context']), 2):
            #    context_queries.append(line['Context'][i])
            #last_answer = ""
            #if len(line['Context']) > 0:
                #last_answer = line['Context'][-1]
            #    for i in range(1, len(line['Context']), 2):
            #        context_answers.append(line['Context'][i])
            oracle_query = data[i]["Truth_rewrite"]

            if last_conv_id != data[i]['Conversation_no']:
                last_response = ""

            pos_docs = []   # actually pos_docs_id
            for rawpid in data[i]['Truth_passages']:
                need_pids.add(d[rawpid])
                pos_docs.append(d[rawpid])

            record = {}
            record["sample_id"] = sample_id
            record["query"] = data[i]["Question"]
            record["context_queries"] = context_queries
            record["context_answers"] = context_answers
            #record["last_answer"] = last_answer
            record["last_response"] = last_response
            record["oracle_query"] = oracle_query
            record["source"] = data[i]["Conversation_source"]
            record["pos_docs"] = pos_docs
            
            last_conv_id = data[i]['Conversation_no']
            if len(data[i]['Truth_passages']) > 0:
                last_response = d[data[i]['Truth_passages'][0]]
            else:
                last_response = ""

            f.write(json.dumps(record))
            f.write('\n')
    print("test file ok")
    
    with open("need_pids.pkl", "wb") as f:
        pickle.dump(need_pids, f, protocol=pickle.HIGHEST_PROTOCOL) 
    print("need_pids dumped!")


    del d, pid2rawpid, data
    gc.collect()

    '''
    # load collection.tsv
    need_pid2passage = {}
    num_total_passage = 54573064
    bad_passage_set = set()
    print("total 54M passages...")
    for line in tqdm(open("qrecc_collection.tsv", "r"), total=num_total_passage):
        try:
            pid, passage = line.strip().split('\t')
        except:
            pid = int(line.strip().split('\t')[0])
            passage = ""
            bad_passage_set.add(pid)
        
        pid = int(pid)
        if pid in need_pids:
            need_pid2passage[pid] = passage

    print("total bad passages = {}".format(len(bad_passage_set)))
    with open("bad_passage_set.pkl", "wb") as f:
        pickle.dump(bad_passage_set, f, protocol=pickle.HIGHEST_PROTOCOL) 
    print("bad_passage_set dumped!")

    print("total need passages = {}".format(len(need_pid2passage)))
    with open("need_pid2passage.pkl", "wb") as f:
        pickle.dump(need_pid2passage, f, protocol=pickle.HIGHEST_PROTOCOL) 
    print("need_pid2passage dumped!")


    # reset last_response and pos_docs in train.json and test.json
    with open("train.json", "r") as f:
        data = f.readlines()
    with open("train.json", "w") as f:
        for line in data:
            line = json.loads(line)
            last_response_pid = line['last_response']
            if last_response_pid != "":
                last_response = need_pid2passage[last_response_pid]
                line['last_response'] = last_response
            
            pos_docs = []
            for pid in line['pos_docs_id']:
                pos_docs.append(need_pid2passage[pid])
            line['pos_docs'] = pos_docs
            f.write(json.dumps(line))
            f.write('\n')

    print("reset last_response and pos_docs in train.json ok!")

    with open("test.json", "r") as f:
        data = f.readlines()
    with open("test.json", "w") as f:
        for line in data:
            line = json.loads(line)
            last_response_pid = line['last_response']
            if last_response_pid != "":
                last_response = need_pid2passage[last_response_pid]
                line['last_response'] = last_response
            
            f.write(json.dumps(line))
            f.write('\n')
    print("reset last_response in test.json ok")
    '''

def gen_bm25_oracle_and_raw_query_file():
    with open("bm25_raw_query.tsv", "w") as f_raw:
        with open("bm25_oralce_query.tsv", "w") as f_oracle:
            with open("test.json", "r") as f:
                data = f.readlines()
            for line in tqdm(data):
                line = json.loads(line)
                sample_id = line["sample_id"]
                raw = line["query"]
                oracle = line["oracle_query"]
                f_raw.write("{}\t{}".format(sample_id, raw))
                f_raw.write('\n')

                f_oracle.write("{}\t{}".format(sample_id, oracle))
                f_oracle.write('\n')


def add_neg_docs():
    with open("qrecc_collection.tsv", "r") as f:
        passages = f.readlines()

    print("load qrecc passages ok")

    num_passage = len(passages)
    num_neg = 20
    with open("train.json", "r") as f:
        train_data = f.readlines()

    with open("new_train.json", "w") as f:
        for line in tqdm(train_data):
            line = json.loads(line)
            neg_docs = []
            seen_neg_ids = set()
            while len(neg_docs) < num_neg:
                neg_id = random.randint(0, num_passage - 1)
                if neg_id in seen_neg_ids:
                    continue
                try:
                    passage = passages[neg_id].strip().split('\t')[1]
                    neg_docs.append(passage)
                except:
                    continue
                seen_neg_ids.add(neg_id)
            assert len(neg_docs) == num_neg

            line['neg_docs'] = neg_docs
            f.write(json.dumps(line))
            f.write('\n') 

def gen_new_test():
    qrel_id = []
    '''
    with open("qrecc_qrel.tsv", "r") as f:
        qrel_data = f.readlines()
        for line in qrel_data:
            line = line.strip().split()
            query = line[0]
            if query not in qrel_id:
                qrel_id.append(query)
    print(len(qrel_id))  
    ''' 
    with open("../../output/qrecc/bm25/test_rel_label_rawq_1.json") as f:
        for line in f:
            line = json.loads(line)
            query_id = line["id"]
            if query_id not in qrel_id:
                qrel_id.append(query_id)
    print(len(qrel_id))  

    with open("test.json", "r") as f:
        test_data = f.readlines() 
    print(len(test_data))

    with open("new_test.json", "w") as f:
        num = 0
        for line in tqdm(test_data):
            line = json.loads(line)
            sample_id = line["sample_id"]
            if sample_id in qrel_id:
                num += 1
                f.write(json.dumps(line))
                f.write('\n')
        print(num)

def gen_new_train():
    qrel_id = []
    '''
    with open("qrecc_qrel.tsv", "r") as f:
        qrel_data = f.readlines()
        for line in qrel_data:
            line = line.strip().split()
            query = line[0]
            if query not in qrel_id:
                qrel_id.append(query)
    print(len(qrel_id))  
    
    with open("../../output/qrecc/bm25/test_rel_label_rawq_1.json") as f:
        for line in f:
            line = json.loads(line)
            query_id = line["id"]
            if query_id not in qrel_id:
                qrel_id.append(query_id)
    print(len(qrel_id))  
    '''
    with open("train.json", "r") as f:
        train_data = f.readlines() 
    print(len(train_data))

    with open("new_train.json", "w") as f:
        num = 0
        for line in tqdm(train_data):
            line = json.loads(line)
            sample_id = line["sample_id"]
            pos_docs = line["pos_docs"]
            if len(pos_docs) > 0:
                num += 1
                f.write(json.dumps(line))
                f.write('\n')
        print(num)



if __name__ == "__main__":
    extract_passage_form_directory()
    gen_qrecc_qrel()
    gen_train_test_file()
    gen_bm25_oracle_and_raw_query_file()
    add_neg_docs()
    gen_new_test()
    gen_new_train()

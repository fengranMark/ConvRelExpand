import csv
import json
import os
from argparse import ArgumentParser

from tqdm import tqdm

WIKI_FILE = "qrecc_collection.tsv"
OUTPUT_FILE = "bm25_collection/qrecc_collection.jsonl"

id_col= 0
text_col= 1
title_col = 2

def main(wiki_file, output_file):

    '''
    # for TopiOCQA
    with open(wiki_file, 'r') as input:
        reader = csv.reader(input, delimiter="\t")
        with open(output_file, 'w') as output:
            for i, row in enumerate(tqdm(reader)):
                if len(row[text_col]) == 0:
                    continue
                text = row[title_col] + ' ' + row[text_col]
                obj = {"id": str(i)", "contents": text}
                output.write(json.dumps(obj, ensure_ascii=False) + '\n')
    '''
    
    # for QReCC
    with open(output_file, 'w') as output:
        for line in tqdm(open(WIKI_FILE, "r")):
            try:
                pid, passage = line.strip().split('\t')
                obj = {"id": str(pid), "contents": passage}
                output.write(json.dumps(obj, ensure_ascii=False) + '\n')
            except:
                continue
            #pid = int(line.strip().split('\t')[0])
            #passage = ""
            #bad_passage_set.add(pid)

if __name__ == "__main__":
    main(WIKI_FILE, OUTPUT_FILE)


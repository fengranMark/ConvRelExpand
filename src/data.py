import torch
from torch.utils.data import Dataset
import json
from transformers import BertTokenizer


def padding(input_dict: dict, max_pad_len: int, pad_token=0):
    input_ids = input_dict['input_ids'].reshape(-1,)
    attention_mask = input_dict['attention_mask'].reshape(-1,)
    padding_len = max_pad_len - len(input_ids)
    padding_ids = torch.tensor([pad_token] * padding_len)
    input_ids = torch.cat((input_ids, padding_ids), 0)
    attention_mask = torch.cat((attention_mask, padding_ids), 0)
    return {'input_ids': input_ids.long(), 'attention_mask': attention_mask.long()}


class QRDataset(Dataset):
    
    def __init__(self, fpath: str, bpath: str):
        """ 
        fpath: dataset file path
        bpath: path stored BertModel and BertTokenizer
        """
        tokened = []
        max_token_len_q, max_token_len_r = 0, 0
        tokenizer = BertTokenizer.from_pretrained(bpath)
        with open(fpath, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            l_dict = json.loads(line)
            """
            {'id': , 'query': str, 'rel_query': str, 'rel_label': int}
            """
            if l_dict['rel_label'] == 1:
                q = tokenizer(l_dict['query'], return_tensors='pt')
                r = tokenizer(l_dict['rel_query'], return_tensors='pt')
                q_len = q['input_ids'].size(1)
                r_len = r['input_ids'].size(1)
                tokened.append({
                    'query': q,
                    'rel_query': r
                })
                max_token_len_q = q_len if max_token_len_q < q_len else max_token_len_q
                max_token_len_r = r_len if max_token_len_r < r_len else max_token_len_r
        
        self.q_data, self.r_data = [], []
        for each in tokened:
            self.q_data.append(padding(each['query'], max_token_len_q))
            self.r_data.append(padding(each['rel_query'], max_token_len_r))

    def __len__(self):
        return len(self.q_data)

    def __getitem__(self, idx):
        return self.q_data[idx], self.r_data[idx]    
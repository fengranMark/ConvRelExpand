import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from data import *


parser = argparse.ArgumentParser(description='Questions Relevance Pre-Training')
parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of training epochs')
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--fpath_train', default='../dataset/topiocqa_train_rel_label.json', type=str, help='file path for training data')
parser.add_argument('--bpath', default='../../bertmodel', type=str, help='file path for downloaded Bert model and tokenizer')

def main():
    global args
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_data = QRDataset(args.fpath_train, args.bpath)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    model = BertModel.from_pretrained(args.bpath).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(args.epochs):
        trainbar = tqdm(train_dataloader)
        for data in trainbar:
            q, r = data[0], data[1]
            q['input_ids'] = q['input_ids'].to(device)
            q['attention_mask'] = q['attention_mask'].to(device)
            r['input_ids'] = r['input_ids'].to(device)
            r['attention_mask'] = r['attention_mask'].to(device)
            feat_q, feat_r = model(**q).last_hidden_state.mean(dim=1), model(**r).last_hidden_state.mean(dim=1)
            feat_q, feat_r = F.normalize(feat_q, dim=1), F.normalize(feat_r, dim=1)
            sim_matrix = feat_q @ feat_r.T
            labels = torch.arange(feat_q.size(0)).to(sim_matrix.device)
            loss = F.cross_entropy(sim_matrix, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trainbar.set_description(f'Train Epoch: [{epoch}/{args.epochs}], Loss: {loss.item():.4f}')
    
    saved_path = "../saved/Finetuned_Bert.pth"
    torch.save(model.state_dict(), saved_path)


if __name__ == '__main__':
    main()
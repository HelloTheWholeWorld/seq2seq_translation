import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torchtext import data, datasets
from torchtext.vocab import FastText
from utils import load_data
from config import MODEL_PATH
import os

class Seq2Seq(nn.Module):
    def __init__(self,
                 source_field,
                 target_field,
                 embedding_dim=128,
                 use_gpu=False,
                 layers=4, 
                 hidden_size=256, 
                 dropout=0.2
                 ):
        super(Seq2Seq, self).__init__()
        self.source_embdding = nn.Embedding(len(source_field.vocab), embedding_dim)
        self.target_embedding = nn.Embedding(len(target_field.vocab), embedding_dim)

        self.encoder = nn.LSTM(
            embedding_dim, # 词嵌入维度
            hidden_size, # 隐藏状态维度
            num_layers=layers, # 层数
            dropout=dropout
        )
        self.decoder = nn.LSTM(
            embedding_dim, # 词嵌入维度
            hidden_size, 
            num_layers=layers,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, len(target_field.vocab)) #输出维度是目标语言词典大小， 映射成one-hot vector
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, src, trg):
        src_emb = self.source_embdding(src)
        enc_out, (enc_hidden, enc_cell) = self.encoder(src_emb)  # 默认初始化输入的hidden state是全零的，除非初始化过参数
        trg_emb = self.target_embedding(trg)
        dec_out, (dec_hidden, dec_cell) = self.decoder(trg_emb, (enc_hidden, enc_cell))

        pred = self.fc(dec_out)
        return pred

    # 初始化参数
    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08) 

# 单步训练
def train_step(model, iterator, optimizer, criterion, CLIP):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src, trg)
        output = output[1:].view(-1, output.shape[-1])  #TODO
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()
        epoch_loss += loss.item()
        if i % 100 == 0:
            print('batch {0}'.format(i))
    return epoch_loss/len(iterator)

def eval(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg)
            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def test_method(model, iterator, src_field, trg_field, batch_size):
    print('start to test')
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            pred = model(src, trg)
            src_tensor = batch.src.transpose(0, 1)
            trg_tensor = batch.trg.transpose(0, 1)
            pred_tensor = pred.topk(1)[1].data.transpose(0, 1)
            for j in range(batch_size):
                src_sent = src_tensor[j]
                trg_sent = trg_tensor[j]
                pred_sent = pred_tensor[j]
                print('source sentence:', ' '.join([src_field.vocab.itos[num] for num in src_sent.data][::-1][1:-1]))
                print('target sentence:', ' '.join([trg_field.vocab.itos[num] for num in trg_sent.data][1:-1]))
                print('predicted sentence:', ' '.join([trg_field.vocab.itos[num] for num in pred_sent.data][1:-1]))
                print()
            

def train(device, epoch=10, batch_size=64):
    eng_field, fren_field, (train, val, test) = load_data()
    model = Seq2Seq(eng_field, fren_field)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        model.init_weights()
    optimizer = optim.Adam(model.parameters())
    
    # 忽略<PAD>
    fren_idx = fren_field.vocab[fren_field.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=fren_idx)

    train_iterator, val_iterator, test_iterator = data.BucketIterator.splits((train, val, test), batch_size=batch_size, device=device)
    print('start to train!!!')
    best_loss = float('inf')
    for ep in range(epoch):
        train_loss = train_step(model, train_iterator, optimizer, criterion, 1)
        val_loss = eval(model, val_iterator, criterion)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
        print('epoch {0}'.format(ep))
        print('train loss {0}'.format(train_loss))
        print('val loss {0}\n'.format(val_loss))

if __name__ == "__main__":
    use_gpu=False
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    if use_gpu:
        torch.backends.cudnn.deterministic = True

    #train
    # train(device=device)

    #test
    eng_field, fren_field, (train, val, test) = load_data()
    model = Seq2Seq(eng_field, fren_field)
    if os.path.exists(MODEL_PATH):
        if not use_gpu:
            model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        else:
            model.load_state_dict(torch.load(MODEL_PATH, map_location='cuda'))
    else:
        print('no model')
        exit(0)
    train_iterator, val_iterator, test_iterator = data.BucketIterator.splits((train, val, test), batch_size=16, device=device)
    test_method(model, test_iterator, eng_field, fren_field, 16)
    
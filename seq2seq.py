import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchtext import data, datasets
from torchtext.vocab import FastText
from utils import load_data

class Seq2Seq(nn.Module):
    def __init__(self,
                 source_field,
                 target_field,
                 embedding_dim=128,
                 use_gpu=False,
                 layers=4, 
                 hidden_size=256, 
                 dropout=0.2,
                 ):
        super(Seq2Seq, self).__init__()
        self.source_embdding = nn.Embedding(len(source_field.vocab), embedding_dim)
        self.target_embedding = nn.Embedding(len(target_field.vocab), embedding_dim)

        self.encoder = nn.LSTM(
            embedding_dim, # 词嵌入维度
            hidden_size, # 隐藏状态特征数？有待考察
            num_layers=layers, # 层数
            batch_first=True,
            dropout=dropout
        )
        self.decoder = nn.LSTM(
            embedding_dim, # 词嵌入维度
            hidden_size, 
            num_layers=layers,
            batch_first=True,
            dropout=dropout
        )
        self.out = nn.Linear(hidden_size, len(fren_field.vocab)) #TODO
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_, hidden):
        src_emb = self.source_embdding(input_)
        output, hidden = self.encoder(src_emb)
        output = self.target_embedding(output)
        output = F.relu(output)
        output, hidden = self.decoder(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)


def train():
    pass

def test():
    pass

def eval():
    pass
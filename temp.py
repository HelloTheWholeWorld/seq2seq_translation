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

# writer = SummaryWriter()
use_pretrained_embeddings = False
cuda = torch.cuda.is_available()
print(cuda)


# def tokenizer(lang):
#     return lambda text: [token.text for token in lang.tokenizer(text)]

# DE = data.Field(eos_token="<eos>",
#                 include_lengths=True, batch_first=True)
# EN = data.Field(init_token="<sos>",
#                 eos_token="<eos>", include_lengths=True, batch_first=True)

# train, val, test = datasets.Multi30k.splits(exts=('.de','.en'), fields=(DE, EN))
# print(len(train), len(val), len(test))

# # Optionally use pretrained word vectors from FastText
# DE.build_vocab(train.src, vectors=FastText('de') if use_pretrained_embeddings else None)
# EN.build_vocab(train.trg, vectors=FastText('en') if use_pretrained_embeddings else None)
# print(len(DE.vocab), len(EN.vocab))

DE, EN, (train, val, test) = load_data()

class Seq2Seq(nn.Module):
    def __init__(self, src, trg):
        super(Seq2Seq, self).__init__()
        SRC_EMB_SIZE = 128
        TRG_EMB_SIZE = 128
        H_SIZE = 256
        LAYERS = 4
        
        self.src_emb = nn.Embedding(len(src), SRC_EMB_SIZE)  
        self.trg_emb = nn.Embedding(len(trg), TRG_EMB_SIZE)
        
        self.encoder = nn.LSTM(SRC_EMB_SIZE, H_SIZE, LAYERS//2, bidirectional=True, dropout=0.2, batch_first=True)
        self.decoder = nn.LSTM(TRG_EMB_SIZE, H_SIZE, LAYERS, dropout=0.2, batch_first=True)
        self.to_trg = nn.Linear(H_SIZE, len(trg))
    
    def forward(self, src_sen_ids, src_lens, trg_sen_ids):
        src_sen_emb = self.src_emb(src_sen_ids)
        src_sen_emb = pack_padded_sequence(src_sen_emb, src_lens, batch_first=True)
        enc_output, enc_hidden = self.encoder(src_sen_emb)
        
        # Always use teacher forcing
        trg_sen_emb = self.trg_emb(trg_sen_ids)
        dec_output, dec_hidden = self.decoder(trg_sen_emb, enc_hidden)

        preds = F.log_softmax(self.to_trg(dec_output), dim=2) 
        return preds

model = Seq2Seq(DE.vocab, EN.vocab)
if cuda: model.cuda()

trg_mask = torch.ones(len(EN.vocab))
trg_mask[EN.vocab.stoi["<pad>"]] = 0
if cuda: trg_mask = trg_mask.cuda()
criterion = nn.NLLLoss(weight=trg_mask)

optimizer = optim.Adam(model.parameters(), lr=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, 15)

train_iter = data.BucketIterator(train, batch_size=64, sort_key=lambda ex: len(ex.src), sort_within_batch=True)
examples = iter(data.BucketIterator(val, batch_size=1, train=False, shuffle=True, repeat=True))

def compare_prediction(src_sen, trg_sen, pred_sen):
    print(">", ' '.join([DE.vocab.itos[num] for num in src_sen.data[0]]))
    print("=", ' '.join([EN.vocab.itos[num] for num in trg_sen.data[0]]))
    print("<", ' '.join([EN.vocab.itos[num[0]] for num in pred_sen]))

def batch_forward(batch):
    src_sen = batch.src[0]
    trg_sen_in = batch.trg[0][:,:-1] # skip eos
    trg_sen = batch.trg[0][:,1:] # skip sos
    preds = model(src_sen, batch.src[1].cpu().numpy(), trg_sen_in)
    return src_sen, trg_sen, preds
    
def sample_prediction(data_iter):
    batch = next(data_iter)
    src_sen, trg_sen, preds = batch_forward(batch)
    pred_sen = preds.topk(1)[1].data[0]
    compare_prediction(src_sen, trg_sen, pred_sen)

# Quick sanity check
# sample_prediction(examples)
for epoch in range(20):    
    scheduler.step()
    # Training loop
    model.train()
    for i, batch in enumerate(train_iter):
        src_sen, trg_sen, preds = batch_forward(batch)
        loss = criterion(preds.contiguous().view(-1,preds.size(2)), trg_sen.contiguous().view(-1))
        # writer.add_scalar('data/train_loss', loss.data[0], len(train_iter)*epoch + i)
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), 5.0)
        optimizer.step()
        if i == len(train_iter)-1:
            break
        print(epoch)
    
    # Validation loop
    model.eval()
    val_iter = data.BucketIterator(val, batch_size=1, sort_key=lambda ex: len(ex.src), sort_within_batch=True, train=False)
    val_loss = val_acc = 0
    for batch in val_iter:
        src_sen, trg_sen, preds = batch_forward(batch)
        val_acc += preds.topk(1)[1].data[0].view(1, -1).eq(trg_sen.data).sum() / trg_sen.size(1)
        val_loss += criterion(preds.contiguous().view(-1,preds.size(2)), trg_sen.contiguous().view(-1))
    # writer.add_scalar('data/val_loss', val_loss/len(val_iter), epoch)
    # writer.add_scalar('data/val_acc', val_acc/len(val_iter), epoch)



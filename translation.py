import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils import clip_grad_norm
from torch import optim
from utils import load_data
from torchtext import data


class Seq2Seq_Translation(nn.Module):
    def __init__(self,
                 eng_field,
                 fren_field,
                 embedding_dim=128,
                 use_gpu=False,
                 layers=4, #可调整
                 hidden_size=256, # 可调整
                 dropout=0.2,
                 ):
        super(Seq2Seq_Translation, self).__init__()

        self.src_emb = nn.Embedding(len(eng_field.vocab), embedding_dim)
        self.trg_emb = nn.Embedding(len(fren_field.vocab), embedding_dim)

        self.encoder = nn.GRU(embedding_dim, hidden_size, num_layers=layers//2, batch_first=True, dropout=dropout, bidirectional=True)
        self.decoder = nn.GRU(embedding_dim, hidden_size, num_layers=layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.to_trg = nn.Linear(hidden_size, len(fren_field.vocab))

    def forward(self, src_sen_ids, src_lens, trg_sen_ids):
        src_sen_emb = self.src_emb(src_sen_ids)
        src_sen_emb = pack_padded_sequence(src_sen_emb, src_lens, batch_first=True)
        enc_output, enc_hidden = self.encoder(src_sen_emb)
        
        # Always use teacher forcing
        trg_sen_emb = self.trg_emb(trg_sen_ids)
        dec_output, dec_hidden = self.decoder(trg_sen_emb, enc_hidden)

        preds = F.log_softmax(self.to_trg(dec_output), dim=2) 
        return preds

def batch_forward(batch):
    src_sen = batch.src[0]
    trg_sen_in = batch.trg[0][:,:-1] # skip eos
    trg_sen = batch.trg[0][:,1:] # skip sos
    preds = model(src_sen, batch.src[1].cpu().numpy(), trg_sen_in)
    return src_sen, trg_sen, preds

if __name__ == "__main__":
    eng_field, fren_field, (train, val, test) = load_data()
    model = Seq2Seq_Translation(eng_field, fren_field)
    trg_mask = torch.ones(len(eng_field.vocab))
    trg_mask[eng_field.vocab.stoi["<pad>"]] = 0
    criterion = nn.NLLLoss(weight=trg_mask)

    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 15)

    train_iter = data.BucketIterator(train, batch_size=64, sort_key=lambda ex: len(ex.src), sort_within_batch=True)
    examples = iter(data.BucketIterator(val, batch_size=1, train=False, shuffle=True, repeat=True))

    for epoch in range(20):
        scheduler.step()
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
        model.eval()
        val_iter = data.BucketIterator(val, batch_size=1, sort_key=lambda ex: len(ex.src), sort_within_batch=True, train=False)
        val_loss = val_acc = 0
        for batch in val_iter:
            src_sen, trg_sen, preds = batch_forward(batch)
            val_acc += preds.topk(1)[1].data[0].view(1, -1).eq(trg_sen.data).sum() / trg_sen.size(1)
            val_loss += criterion(preds.contiguous().view(-1,preds.size(2)), trg_sen.contiguous().view(-1))

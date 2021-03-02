import torch
import config
from torch import optim
import torch.nn as nn
from encoder import NumEncoder
from decoder import NumDecoder
from seq2seq import Seq2Seq
# from dataset import data_loader as train_dataloader
from word_sequence import num_sequence
from data import Corpus

from torch.utils.data import Dataset,DataLoader
import torch.utils.data as Data
corpus = Corpus('./data')


encoder = NumEncoder(vocab_size=len(corpus.dictionary))
decoder = NumDecoder(vocab_size=len(corpus.dictionary))
model = Seq2Seq(encoder,decoder)
print(model)
print("\nInitializing weights...")
for name, param in model.named_parameters():
    if 'bias' in name:
        torch.nn.init.constant_(param, 0.0)
    elif 'weight' in name:
        torch.nn.init.xavier_normal_(param)

# model.load_state_dict(torch.load("model/seq2seq_model.pkl"))
optimizer =  optim.Adam(model.parameters())
# optimizer.load_state_dict(torch.load("model/seq2seq_optimizer.pkl"))
criterion= nn.NLLLoss(ignore_index=num_sequence.PAD,reduction="mean")


train_source = torch.LongTensor(corpus.train[0])
train_target = torch.LongTensor(corpus.train[1])

torch_dataset = Data.TensorDataset(train_source,train_target)
data_loader = DataLoader(dataset=torch_dataset,batch_size=config.batch_size,shuffle=True,drop_last=True)

def get_loss(decoder_outputs,target):
    target = target.view(-1) #[batch_size*max_len]
    decoder_outputs = decoder_outputs.view(config.batch_size*config.max_len,-1)
    return criterion(decoder_outputs,target)


def train(epoch):
    total_loss = 0
    cur_loss = 0
    for idx,(input,target) in enumerate(data_loader):
        optimizer.zero_grad()
        ##[seq_len,batch_size,vocab_size] [batch_size,seq_len]
        # print('input',input)
        # print('target',target)
        decoder_outputs,decoder_hidden = model(input,target,torch.LongTensor(corpus.train_max_length+1),torch.LongTensor(3))
        # print(decoder_outputs.size())
        # print(target.size())
        loss = get_loss(decoder_outputs,target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if idx % config.inter_val == 0 and idx > 0:
            cur_loss = total_loss / config.inter_val
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,idx * len(input), len(data_loader.dataset),
                   100. * idx / len(data_loader), loss.item()))

            torch.save(model.state_dict(), "model/seq2seq_model_test.pkl")
            torch.save(optimizer.state_dict(), 'model/seq2seq_optimizer_test.pkl')

if __name__ == '__main__':
    for i in range(2):
        train(i)
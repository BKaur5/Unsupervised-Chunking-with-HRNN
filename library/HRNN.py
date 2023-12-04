import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchtext.data import BucketIterator
from library.logger import timing_logger
import transformers
from subprocess import run, PIPE
import numpy as np
from library.labelled_entry import LabelledEntry


BATCH_SIZE = 1

def make_bucket_iterator(
    data,
    device: torch.device,
):
    bucket_iterator = BucketIterator(
        data, 
        batch_size=BATCH_SIZE,
        sort_key=lambda x: np.count_nonzero(x[0]),
        sort=False, 
        shuffle=False,
        sort_within_batch=False,
        device=device,
    )
    bucket_iterator.create_batches()
    return bucket_iterator



class HRNNtagger(nn.ModuleList):
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        tagset_size: int,
        device: torch.device,
        train_embeddings: bool = False,
        vocab_size: int = None,
    ) -> None:

        super(HRNNtagger, self).__init__()
        self.device = device
        self.criterion = nn.NLLLoss().to(device)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.train_embeddings = train_embeddings
        if train_embeddings:
            self.vocab_size = vocab_size
            self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=padding_idx).to(self.device)
        self.rnn11 = nn.RNNCell(self.embedding_dim, self.hidden_dim).to(self.device)
        self.rnn12 = nn.RNNCell(self.embedding_dim, self.hidden_dim).to(self.device)
        self.rnn21 = nn.RNNCell(self.hidden_dim, self.hidden_dim).to(self.device)
        self.hidden2tag = nn.Linear(self.hidden_dim+self.hidden_dim+self.embedding_dim, self.tagset_size).to(self.device)
        self.soft = nn.Softmax(dim=1).to(self.device)
        

    def forward(
        self,
        h_init: torch.Tensor,
        x: torch.Tensor,
        seqlens: int,
        mask_ind = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        output_seq = torch.zeros((seqlens, self.tagset_size)).to(self.device)

        h11 = h_init.to(self.device)
        h12 = h_init.to(self.device)
        h1_actual = h_init.to(self.device)

        h21 = h_init.to(self.device)
        h22 = h_init.to(self.device)
        h2_actual = h_init.to(self.device)

        embeddings = self.word_embeddings(x) if self.train_embeddings else x
        for t in range(seqlens):
            entry = torch.unsqueeze(embeddings[t], 0).to(self.device)
            next_entry = torch.unsqueeze(embeddings[t], 0).to(self.device) \
                        if t == seqlens-1 else \
                        torch.unsqueeze(embeddings[t+1], 0).to(self.device)
            h11 = self.rnn11(entry, h1_actual)
            h12 = self.rnn12(entry, h_init)
            h22 = h2_actual
            h21 = self.rnn21(h1_actual, h2_actual)

            if t == 0:
                h1_actual = mask_ind*h12 + (1-mask_ind)*h11
                h2_actual = mask_ind*h21 + (1-mask_ind)*h22
                h_init = h1_actual
            else:
                h1_actual = torch.mul(h11, output[0]) + torch.mul(h12, output[1])
                h2_actual = torch.mul(h22, output[0]) + torch.mul(h21, output[1])

            tag_rep = self.hidden2tag(torch.cat((h1_actual, h2_actual, next_entry), dim=1)).to(self.device)
            output = torch.squeeze(self.soft(tag_rep))
            output_seq[t] = output

        return output_seq, h2_actual

    def proceed(
        self,
        batch: torch.Tensor,
        hc: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = batch[0][0].to(device)
        tag = torch.as_tensor(batch[0][1].get_boolean_tags(), dtype=torch.bool, device=device)
        seqlens = torch.as_tensor(torch.count_nonzero(tag, dim=-1), dtype=torch.int64, device='cpu')+2
        # WHAT DOES THIS DO?
        # was previously (tag-1)1:seqlens - 1]
        tag = ~tag[1:seqlens - 1]
        # again does this work? does doing self work the same way as when we were doing it for model before?
        self.zero_grad()
        tag_scores,_ = self(hc, embedding, seqlens)
        tag_scores = torch.log(tag_scores[1:seqlens-1])
        # how do i change this for boolean
        selection = (tag != 2)
        tag_long = tag[selection].to(torch.long)
        loss = self.criterion(tag_scores[selection], tag_long)
        return tag_scores, loss


    def train(
        self,
        data,
        optimizer,
        scheduler,
        device,
    ) -> float:
        self.train()
        loss_sum = 0.
        bucket_iterator = make_bucket_iterator(data, device=device)
        for batch in tqdm(bucket_iterator.batches, total=len(bucket_iterator)):
            hc = self.init_hidden().to(device)
            _, loss = self.proceed(batch, hc, device)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        scheduler.step()
        return loss_sum / len(bucket_iterator)

    def predict(
        self,
        data,
        sentences_words,
        device,
    ) -> tuple[float, str]:
        # does not work gives an error, do we need it? 
        # self.eval()
        hc = self.init_hidden().to(device)
        loss_sum = 0.
        bucket_iterator = make_bucket_iterator(data, device=device)
        output_entries = []
        with torch.no_grad():
            for i, (batch, sentence_words) in tqdm(enumerate(zip(bucket_iterator.batches, sentences_words)), total=len(bucket_iterator)):
                tag_scores, loss = self.proceed(batch, hc, device)
                loss_sum += loss.item()
                ind = torch.argmax(tag_scores, dim=1)
                output_entries.append(LabelledEntry.load_from_boolean_format(sentence_words, ind))
        return loss_sum / len(bucket_iterator), output_entries



    def init_hidden(self):
        return (torch.zeros(BATCH_SIZE, self.hidden_dim))


def get_training_equipments(
	model: HRNNtagger,
    lr: float,
    num_iter: int,
    warmup: int,
) -> tuple[torch.optim.Optimizer, transformers.SchedulerType]:
    optimizer = optim.Adam(model.parameters(), lr=lr*(num_iter+1)/num_iter, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup,
        num_training_steps=num_iter+1,
        num_cycles=0.5,
        last_epoch=-1
    )
    scheduler.step()
    return optimizer, scheduler




# add it to HRNN: _proceed -> proceed
 # predicted probs, true tags, loss



# add to HRNN: model -> self, _proceed -> self.proceed




# remove
# chnage this to return an entry
# def validation_output(
#     ind,
#     true_tag,
# ) -> str:
#     output = "x y B B\n"
#     for i, pred in enumerate(ind[:-1]):
#         if i + 1 >= len(true_tag):
#             break  # Exit the loop if accessing true_tag[i+1] is out of bounds
#         true_label = true_tag[i+1]
#         if true_label not in ["B", "I"]:
#             continue
#         predicted_label = "B" if pred else "I"

#         output += f"x y {true_label} {predicted_label}\n"
#     return output


# add to HRNN, model -> self, validate (function's name) -> predict, _proceed -> self.proceed


# remove
# i think this needs to be updated?
# def eval_conll2000(
#     pairs: str,
#     eval_conll_path: str = 'library/eval_conll.pl',
# ) -> tuple[float, float]: # F1, Acc
#     pipe = run(["perl", eval_conll_path], stdout=PIPE, input=pairs, encoding='ascii')
#     output = pipe.stdout.split('\n')[1]
#     tag_acc = float(output.split()[1].split('%')[0])
#     phrase_f1 = float(output.split()[-1])
#     return phrase_f1, tag_acc
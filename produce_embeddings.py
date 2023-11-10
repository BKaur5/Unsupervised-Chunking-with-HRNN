import torch
import sys
import tqdm
import os
import yaml
from datetime import datetime
import shutil
from library.labelled_entry import LabelledEntry
from transformers import AutoModel, AutoTokenizer

EXPERIMENT_PATH = "embeddings"


def main():
    # first argument is path to config file
    config_path = sys.argv[1]
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # second argument is path to folder containing input files
    # read this from config, it is a list of files and not a folder
    input_files_path = sys.argv[2]
    with open(input_files_path, 'r') as file:
    # Read each line (assuming each line contains a file name)
        for line in file:
            # Strip any leading or trailing whitespaces
            file_name = line.strip()
            # Open and read the content of the file
            with open(file_name, 'r') as current_file:
                data = map(
                lambda line: LabelledEntry.load_from_bracket_format(line).sentence.rstrip(']'),
                open(file_name).readlines()
            )
            
            #create_embeddings(data,config)

    data_padding(data,None)               
           
    
    # creating datetime folder and copying  config + input_folder
    date_time = str(datetime.now()).replace(' ','_')
    new_folder_path = os.path.join(EXPERIMENT_PATH,date_time)
    # If the folder already exists, remove it
    if os.path.exists(new_folder_path):
        shutil.rmtree(new_folder_path)
    os.mkdir(new_folder_path)
    # copying files
    shutil.copy(config_path,new_folder_path)
   
            
def data_padding(data,device):
    data = list(data)
    split_data = [sentence.split() for sentence in data]
    data_lengths = [len(sentence) for sentence in split_data]
    ind = data_lengths.index(max(data_lengths))
    max_data_len = max(data_lengths) + 2
    padded_data = torch.empty(len(split_data), max_data_len, dtype=torch.long).to(device)
    padded_data.fill_(0.)
    #word_to_ix = build_vocab(split_data)

    for i,sentence_len in enumerate(data_lengths):
        sequence = split_data[i]
        sequence = ['<SOS>'] + sequence + ['<EOS>']
        while len(sequence) != max_data_len:
            sequence.append('<PAD>')
        # do we want this as a list or string ???

        # sequence = prepare_sequence(sequence, word_to_ix)
        # sequence = torch.tensor(sequence, dtype=torch.long)
        to_ix = word_to_index(sequence)
        idxs = [to_ix[w] for w in sequence]
        sequence = torch.tensor(idxs, dtype=torch.long)
        padded_data[i, :len(sequence)] = sequence[0:len(sequence)]
    
    return padded_data
    

def prepare_sequence(seq, to_ix):
	idxs = [to_ix[w] for w in seq]
	return torch.tensor(idxs, dtype=torch.long)

def word_to_index(data):
    word_to_ix = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
    for word in data:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

    return word_to_ix

def select_indices(tokens, raw_tokens, model, mode):
    mask = []
    raw_i = 0
    collapsed = ''
    model = model.split('/')[-1].split('-')[0]
    special = specials[model]

    for i in range(len(tokens)):
        token = tokens[i]

        while len(token) > 0 and token[0] == special:
            token = token[1:]
        if collapsed == '' and len(token) > 0:
            start_idx = i
        collapsed += token
        if collapsed == raw_tokens[raw_i]:
            if mode == 'first':
                mask.append(start_idx)
            elif mode == 'last':
                mask.append(i)
            else:
                raise NotImplementedError
            raw_i += 1
            collapsed = ''
    if raw_i != len(raw_tokens):
        raise Exception(f'Token mismatch: \n{tokens}\n{raw_tokens}')
    return mask

specials = {'bert': '#', 'gpt2': 'Ġ', 'xlnet': '▁', 'roberta': 'Ġ'}

def group_indices(tokens, raw_tokens, model):
    # print(tokens)
    # print(raw_tokens)
    # tokens, raw_tokens = persian_preprocess(tokens, raw_tokens)

    mask = []
    raw_i = 0
    model = model.split('/')[-1].split('-')[0]
    special = specials[model]

    collapsed = ''
    options = [raw_tokens[raw_i]]
    skip = 0
    collapsed_cnt = 0
    for i in range(len(tokens)):
        token = tokens[i]

        while len(token) > 0 and token[0] == special:
            token = token[1:]

        collapsed_cnt += 1
        if token != '[UNK]':
            collapsed += token
            # print(options, collapsed)
            if collapsed in options:
                raw_tokens_cnt = options.index(collapsed)
                for j in range(raw_tokens_cnt+1):
                    mask.append(raw_i)
                    raw_i += 1
                for j in range(collapsed_cnt-raw_tokens_cnt-1):
                    mask.append(raw_i-1)
                if raw_i >= len(raw_tokens):
                    if i != len(tokens)-1:
                        raise Exception("Tokens more that tags.")
                    break
                options = [raw_tokens[raw_i]]
                collapsed = ''
                collapsed_cnt = 0
                skip = 0
        else:
            if collapsed:
                print(options, collapsed)
                raise Exception("Invalid token-tags!")
            skip += 1
            options.append(raw_tokens[raw_i+skip])

    if raw_i != len(raw_tokens):
        print(options, collapsed)
        return 
    return torch.tensor(mask)

    # function that receives a list of sentences in string format, return a list of tensors with added padding
def compute_emb_by_bert(
    data: list[list],
    ix_to_word: dict,
    config: dict,
    device: torch.device,
) -> torch.Tensor:
    
    token_heuristic = config['embedding_token_heuristic']
    model_class = AutoModel
    tokenizer_class = AutoTokenizer
    pretrained_weights = config['bert_pretrained_weights']

    tokenizer = tokenizer_class.from_pretrained(pretrained_weights, cache_dir='LM/cache')
    model = model_class.from_pretrained(pretrained_weights, cache_dir='LM/cache', 
        output_hidden_states=True, output_attentions=True).to(device)

    with torch.no_grad():
        test_sent = tokenizer.encode('test', add_special_tokens=False)
        token_ids = torch.tensor([test_sent]).to(device)
        all_hidden, all_att = model(token_ids)[-2:]
        
        n_layers = len(all_att)
    
    feat_sents = torch.zeros([len(data), len(data[0]), config['embedding_dim']]) 

    for idx, s_tokens in enumerate(tqdm(data)):
        #################### read words and extract ##############
        s_tokens = [ix_to_word[ix] for ix in s_tokens.cpu().numpy()]

        raw_tokens = s_tokens
        s = ' '.join(s_tokens)
        tokens = tokenizer.tokenize(s)

        token_ids = tokenizer.encode(s, add_special_tokens=False)
        token_ids_tensor = torch.tensor([token_ids]).to(device)
        with torch.no_grad():
            all_hidden, all_att = model(token_ids_tensor)[-2:]
        all_hidden = list(all_hidden[1:])
        
        # (n_layers, seq_len, hidden_dim)
        all_hidden = torch.cat([all_hidden[n] for n in range(n_layers)], dim=0)
        
        #################### further pre processing ##############
        # try to only use last layer of all_hidden
        if len(tokens) > len(raw_tokens):
            th = token_heuristic
            if th == 'first' or th == 'last':
                mask = select_indices(tokens, raw_tokens, pretrained_weights, th)
                assert len(mask) == len(raw_tokens)
                all_hidden = all_hidden[:, mask]

            else:
                mask = group_indices(tokens, raw_tokens, pretrained_weights)
                raw_seq_len = len(raw_tokens)
                all_hidden = torch.stack(
                    [all_hidden[:, mask == i].mean(dim=1)
                     for i in range(raw_seq_len)], dim=1)
        
        all_hidden = all_hidden[n_layers - 1]
        feat_sents[idx] = all_hidden	
    
    return feat_sents
    

if __name__ == "__main__":
    main()
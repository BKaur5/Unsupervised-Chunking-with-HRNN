import torch
import sys
from tqdm import tqdm
import yaml
from library.labelled_entry import LabelledEntry
from transformers import AutoModel, AutoTokenizer
from library.utils import create_datetime_folder,copy_files,get_torch_device
import pickle
import os


EXPERIMENT_PATH = "embeddings"


def main():
    # first argument is path to config file
    config_path = sys.argv[1]
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # second argument is path to folder containing input files
    # read this from config, it is a list of files and not a folder
    sentences_files = config['sentences-files']

    for file in sentences_files:
        # Strip any leading or trailing whitespaces
        file_name = file.strip()
        # Open and read the content of the file
        with open(file_name, 'r') as current_file:
            sentences = list(map(
            lambda line: LabelledEntry.load_from_bracket_format(line).sentence.rstrip(']'),
            open(file_name).readlines()
        ))
            
          
    new_folder_path = create_datetime_folder(EXPERIMENT_PATH)
    # copying files
    files_to_copy = [config_path]
    copy_files(files_to_copy,new_folder_path)
    sentences_words = [sentence.split() for sentence in sentences]
    padded_sentences_words = data_padding(sentences_words)
    word_to_index_dict = word_to_index(padded_sentences_words)
    # word_list = []
    # for word,index in word_to_index_dict.items():
    #     word_list.append(index)
    word_list = [[word_to_index_dict[w] for w in sentence] for sentence in padded_sentences_words]
    indexes = torch.tensor(word_list,dtype=torch.long)
    #print(word_to_index('<PAD>')[0])
    index_to_words = {v: k for k, v in word_to_index_dict.items()}
    #print(index_to_words)
    device = get_torch_device(config)
    embeddings = induce_embeddings(indexes,index_to_words,config,device) 
    embd_file_path = os.path.join(new_folder_path,'bert_embeddings.ebd.pt')

    # Write the list of tensors to the .pkl file
    torch.save(embeddings,embd_file_path)
    
    # instead of a list of tensors, use a 2-d tensor and use torch.save. file extension:  .ebd.pt
 
def data_padding(sentences):
    data_lengths = [len(sentence) for sentence in sentences]
    max_data_len = max(data_lengths) + 2
    padded_data = []
    for sentence in sentences:
        padded_sentence = ['<SOS>'] + sentence + ['<EOS>']
        while len(padded_sentence) != max_data_len:
            padded_sentence.append('<PAD>')
        padded_data.append(padded_sentence)
    return padded_data
    
def word_to_index(sentences):
    word_to_ix = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
    for sentence in sentences:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    
    return word_to_ix

    # function that receives a list of sentences in string format, return a list of tensors with added padding
def induce_embeddings(
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

if __name__ == "__main__":
    main()

import torch
import sys
import pickle
import os
import yaml
from datetime import datetime
import shutil

EXPERIMENT_PATH = "experiments"
# function that receives a list of sentences in string format, return a list of tensors with added padding
def create_embeddings(sentences,config):
    for sentence in sentences:
        pass

def compute_bert_embeddings():
    pass

def main():
    # first argument is path to config file
    config_path = sys.argv[1]
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # second argument is path to folder containing input files
    input_folder_path = sys.argv[2]
    # third argument is path to output file 
    output_folder_path = sys.argv[3]
    # creating datetime folder and copying  config + input_folder
    date_time = str(datetime.now()).replace(' ','_')
    new_folder_path = os.path.join(EXPERIMENT_PATH,date_time)
    # If the folder already exists, remove it
    if os.path.exists(new_folder_path):
        shutil.rmtree(new_folder_path)
    os.mkdir(new_folder_path)
    # copying files
    shutil.copy(config_path,new_folder_path)
    if os.path.exists(input_folder_path):
        files = os.listdir(input_folder_path)
        for file in files:
            file_path = os.path.join(input_folder_path, file)
            data = pickle.load(open(file_path, 'rb'))
            create_embeddings(data,config)
    
def data_padding(data, word_to_ix, tag_to_ix, device, max_seq_len=20):
	data_lengths = [len(sentence)+2 for sentence,tags in data]
	
	max_seq_len = max(data_lengths)
	padded_data = torch.empty(len(data), max_seq_len, dtype=torch.long).to(device)
	padded_data.fill_(0.)
	# copy over the actual sequences
	for i, x_len in enumerate(data_lengths): 
		sequence,tags = data[i]
		sequence.insert(0,'SOS')
		sequence.append('EOS')
		
		sequence = prepare_sequence(sequence, word_to_ix)
		padded_data[i, 0:x_len] = sequence[0:x_len]

	tag_lengths = [len(tags)+2 for sentence, tags in data]
	padded_tags = torch.empty(len(data), max_seq_len, dtype=torch.long).to(device)
	padded_tags.fill_(0.)

	for i, y_len in enumerate(tag_lengths):
		sequence,tags = data[i]
		tags.insert(0,'<pad>')  # for SOS
		tags.append('<pad>')  ## for EOS

		tags = prepare_sequence(tags, tag_to_ix)
		padded_tags[i, 0:y_len] = tags[:y_len]

	return padded_data, padded_tags, max_seq_len    

def prepare_sequence(seq, to_ix):
	idxs = [to_ix[w] for w in seq]
	return torch.tensor(idxs, dtype=torch.long)


if __name__ == "__main__":
    main()
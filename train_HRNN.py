# CUDA_VISIBLE_DEVICES=0 python3 train_script.py experiments/???/config.yml

import yaml
import sys
import pickle
import torch
from library.utils import get_torch_device,create_experiment_csv
from library.HRNN import HRNNtagger, get_training_equipments, train, validate, eval_conll2000
from produce_embeddings import data_padding  #we are reading from the file right?
import numpy as np
import os
import csv


def _train(model, data, optimizer, scheduler, train_csv_file, name, device):
    loss = train(model, data, optimizer, scheduler, device=device)
    with open(train_csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(name, loss)
    return loss


def _validate(model, data, true_tags, config, name, losses, fscores, accs,validate_csv_file,device):
    if config['validation_mode'].lower() == 'enforced':
        enforced_tags = pickle.load(open(config['enforced_validation_tags'], "rb"))
    else:
        enforced_tags = None
    loss, validation_output = validate(
        model,
        data,
        true_tags,
        device=device,
        enforced_tags=enforced_tags,
        enforced_mode=config['enforced_mode'].lower(),
    )
    fscore, acc = eval_conll2000(validation_output)
    # if config['validation_checkpoints_path']:
    #     pred_path = config['home']+config['validation_checkpoints_path'] + 'validation-' + str(name) + '.out'
    #     with open(pred_path, 'w') as f:
    #         f.write(validation_output)
    # print( " __________________________________")
    # print(f"| Validation {name}:")
    # print(f"|     Loss:     {loss}")
    # print(f"|     F1:       {fscore}")
    # print(f"|     Accuracy: {acc}")
    # print( "|__________________________________")
    losses.append(loss)
    fscores.append(fscore)
    accs.append(acc)
    with open(validate_csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(name, loss,fscore,acc)

    # do we need this plot stuff
    if config['validation_metrics']:
        plot_vars = {
            'loss': losses,
            'fscore': fscores,
            'acc': accs,
        }
        with open(config['home']+config['validation_metrics'], 'wb') as f:
            pickle.dump(plot_vars, f)
    return fscore
    

def main():
    with open(sys.argv[1]) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = get_torch_device(config)
    if device != 'cpu' and not torch.cuda.is_available():
        raise Exception('THERE IS NO CUDA AVAILABLE!')
    else:
        device = torch.device(device)
  


    training_data = pickle.load(open(config['train_data'], "rb"))
    validation_data = pickle.load(open(config['validation_data'], "rb"))
    validation_true_tags = pickle.load(open(config['validation_true_tags'], "rb"))
    # word_to_ix, ix_to_word, tag_to_ix, ix_to_tag = build_vocab(training_data + validation_data)
    # train_tokens, train_tags, train_msl = data_padding(training_data, word_to_ix, tag_to_ix, device=device)
    # validation_tokens, validation_tags, validation_msl = data_padding(validation_data, word_to_ix, tag_to_ix, device=device)
    
    # update path
    #if config['load_last_embeddings'] and os.path.exists(config['home']+config['validation_embeddings']):
    validation_embeddings = torch.load(config['validation_embeddings'], map_location=device)
    training_embeddings = torch.load(config['train_embeddings'], map_location=device)

    # do we need these tags or do we just read them in?
    training_data = list(zip(training_embeddings, train_tags))
    validation_data = list(zip(validation_embeddings, validation_tags))


    hrnn_model = HRNNtagger(
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        tagset_size=2,
        device=device,
    ).to(device)
    optimizer, scheduler = get_training_equipments(hrnn_model, lr=config['learning_rate'], num_iter=config['epocs'], warmup=config['warmup'])

    train_loss_vec = []
    validation_records = [], [], []
    # validation_loss_vec, validation_fscore_vec, validation_acc_vec = validation_records
    best_fscore = 0.

    if config['pretrained_model']:
        hrnn_model.load_state_dict(torch.load(config['home']+config['pretrained_model'], map_location=torch.device(device)))
    _validate(
        hrnn_model,
        validation_data,
        validation_true_tags,
        config,
        'pre-trained' if config['pretrained_model'] else 'init model',
        *validation_records,
        device=device
    )

    train_file = create_experiment_csv(config,"train_results.csv",["Epoch","Loss"])
    validate_file = create_experiment_csv(config,"validate_results.csv",["Epoch","Loss","F1","Accuracy"])
    for epoch in range(config['epocs']):

        _train(hrnn_model, training_data, optimizer, scheduler, train_file,epoch, device=device)
        fscore = _validate(hrnn_model, validation_data, validation_true_tags, config, epoch, *validation_records, validate_file,device=device)

        if fscore > best_fscore:
            best_fscore = fscore
            torch.save(hrnn_model.state_dict(), config['home']+config['best_model_path'])


if __name__ == "__main__":
	main()

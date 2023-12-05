# CUDA_VISIBLE_DEVICES=0 python3 train_script.py experiments/???/config.yml

import yaml
import sys
import torch
from library.utils import get_torch_device,create_experiment_csv,read_entries,create_datetime_folder,validate
from library.HRNN import HRNNtagger, get_training_equipments# change based on new HRNN file
import csv

def _train(model, data, optimizer, scheduler, train_csv_file, name, device):
    loss = model.train_from(data, optimizer, scheduler, device=device)
    with open(train_csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, loss])
    return loss
    
def main():
    with open(sys.argv[1]) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = get_torch_device(config)
   
    training_entries = read_entries(config['train_data'])
    validation_entries = read_entries(config['validation_data'])

    # update path
    # if config['load_last_embeddings'] and os.path.exists(config['home']+config['validation_embeddings']):
    validation_embeddings = torch.load(config['validation_embeddings'], map_location=device)
    training_embeddings = torch.load(config['train_embeddings'], map_location=device)

    training_data = list(zip(training_embeddings, training_entries))
    validation_data = list(zip(validation_embeddings, validation_entries))

    hrnn_model = HRNNtagger(
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        tagset_size=2,
        device=device,
    ).to(device)
    optimizer, scheduler = get_training_equipments(hrnn_model, lr=config['learning_rate'], num_iter=config['epocs'], warmup=config['warmup'])

    best_fscore = 0.

    if config['pretrained_model']:
        hrnn_model.load_state_dict(torch.load(config['home']+config['pretrained_model'], map_location=torch.device(device)))
    
    results_path = config["experiments-path"]
    results_folder = create_datetime_folder(results_path)
    train_file = create_experiment_csv(results_folder,config,"train_results.csv",["Epoch","Loss"])
    validate_file = create_experiment_csv(results_folder,config,"validate_results.csv",["Epoch","Loss","F1","Accuracy"])

    validate(
        hrnn_model,
        validation_data,
        validation_entries,
        'pre-trained' if config['pretrained_model'] else 'initial model',
        validate_file,
        device=device
    )
    
    for epoch in range(config['epocs']):
        print("EPOCH: {}".format(epoch))
        _train(hrnn_model, training_data, optimizer, scheduler, train_file,epoch, device=device)
        _, fscore, _ = validate(hrnn_model, validation_data, validation_entries, epoch, validate_file,device=device)

        if fscore > best_fscore:
            best_fscore = fscore
            torch.save(hrnn_model.state_dict(), results_folder)

if __name__ == "__main__":
	main()
from library.utils import validate, get_torch_device, read_entries,create_experiment_csv
from library.HRNN import HRNNtagger
import os
import yaml
import torch
import sys

def main():
    experiment_path = sys.argv[1]
    with open(os.path.join(experiment_path, 'config.yml')) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = get_torch_device(config)
   
    test_entries = read_entries(config['train_data'])
    test_embeddings = torch.load(config['test_embeddings'], map_location=device)
    test_data = list(zip(test_embeddings, test_entries))

    hrnn_model = HRNNtagger(
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        tagset_size=2,
        device=device,
    ).to(device)
    hrnn_model.load_state_dict(torch.load(config['home']+config['best_model_path'], map_location=torch.device(device)))
    
    test_file = create_experiment_csv(experiment_path,config,"test_results.csv",["Loss","F1","Accuracy"])

    validate(
        hrnn_model,
        test_data,
        test_entries,
        'trained',
        test_file,
        device=device
    )

    
if __name__ == "__main__":
	main()

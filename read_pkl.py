import torch

# Specify the path to your .pt file
pt_file_path = 'embeddings/2023-11-28_18:53:36.098279/bert_embeddings.ebd.pt'

# Load the content from the .pt file
data = torch.load(pt_file_path, map_location=torch.device('cpu'))

# Print the content
print(data)
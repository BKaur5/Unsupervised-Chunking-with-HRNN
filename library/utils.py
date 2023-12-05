from datetime import datetime
import os
import shutil
import csv
import torch
from library.labelled_entry import LabelledEntry
from eval_heuristic import eval

def create_datetime_folder(path):
    date_time = str(datetime.now()).replace(' ','_')
    new_folder_path = os.path.join(path,date_time)
    # If the folder already exists, remove it
    if os.path.exists(new_folder_path):
        shutil.rmtree(new_folder_path)
    os.mkdir(new_folder_path)
    return new_folder_path

def copy_files(filenames,destination):
    for filename in filenames:
        shutil.copy(filename,destination)

def get_torch_device(config):
    device = config['device']
    if device != 'cpu' and not torch.cuda.is_available():
        raise Exception('THERE IS NO CUDA AVAILABLE!')
    else:
        device = torch.device(device)
        return device

def create_experiment_csv(results_folder,config,csv_name,headers):
    csv_path = os.path.join(results_folder,csv_name)
    with open(csv_path, mode='w', newline='') as file:
    # Create a CSV writer object
        writer = csv.writer(file)

        # Write the headers to the CSV file
        writer.writerow(headers)

    return csv_path

def read_entries(file_path):
    entries = []
    with(open(file_path, 'r')) as f:
        sentences = f.readlines()
        for sentence in sentences:
            entry = LabelledEntry.load_from_bracket_format(sentence)
            entries.append(entry)
    return entries 

def validate(model, data, entries, name,validate_csv_file,device):

    loss, validation_entries = model.predict(
        data,
        [entry.get_words() for entry in entries],
        device=device,
    )

    fscore, acc = eval(entries,validation_entries)
    with open(validate_csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name,loss,fscore,acc])

    return loss, fscore, acc
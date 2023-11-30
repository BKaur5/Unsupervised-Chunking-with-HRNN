from datetime import datetime
import os
import shutil
import csv
from library.labelled_entry import LabelledEntry

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
    return config['device']


def create_experiment_csv(config,csv_name,headers):
    results_path = config["experiments"]
    results_folder = create_datetime_folder(results_path)
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


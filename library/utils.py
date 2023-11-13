from datetime import datetime
import os
import shutil

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
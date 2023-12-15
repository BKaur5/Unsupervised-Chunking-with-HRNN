
from eval_heuristic import eval
import csv

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
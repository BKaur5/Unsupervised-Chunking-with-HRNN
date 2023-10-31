"""
This python script is used to convert the data files to new notation to identify chunks.
The new notations means using square brackets to identify chunk boundaries instead of using B and I 
"""
import pickle

def convert_to_new_notation(words, labels):
    new_notation = []
    current_chunk = []

    for word, label in zip(words, labels):
        if label == '2':  # Word boundary
            if current_chunk:
                new_notation.append(current_chunk)
            current_chunk = [word]
        else:
            current_chunk.append(word)

    if current_chunk:
        new_notation.append(current_chunk)

    return new_notation



def convert_save_to_new_file(data, new_file_path):
    # Convert each row in the loaded data to the new notation
    new_data = []

    for words, labels in data:
        new_notation = convert_to_new_notation(words, labels)
        new_data.append(new_notation)

    with open(new_file_path, 'wb') as new_file:
        pickle.dump(new_data, new_file)

def main():
    # Defining paths for all the train, test, and val data files and loading them to get the data
    train_data_path = "../data/separated_data_en/train.pkl"
    train_data = pickle.load(open(train_data_path, 'rb'))
    val_data_path = "../data/separated_data_en/val.pkl"
    val_data = pickle.load(open(val_data_path, 'rb'))
    test_data_path = "../data/separated_data_en/test.pkl"
    test_data = pickle.load(open(test_data_path, 'rb'))

    new_train_path = "../data/new_notation_separated_data_en/train.pkl"
    new_val_path = "../data/new_notation_separated_data_en/val.pkl"
    new_test_path = "../data/new_notation_separated_data_en/test.pkl"

    convert_save_to_new_file(train_data, new_train_path)
    convert_save_to_new_file(val_data, new_val_path)
    convert_save_to_new_file(test_data, new_test_path)

if __name__ == "__main__":
    main()




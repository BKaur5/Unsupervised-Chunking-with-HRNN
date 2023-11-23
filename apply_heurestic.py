from library.heurestics import double_word_heurestic, single_word_heurestic
import pickle

def heuristic_output_in_bracketed_form(original_sentences_file, output_file, heuristic):
    data = pickle.load(open(original_sentences_file, 'rb'))
    with(open(output_file, 'w')) as f:
        for row in data:
            sentence = ' '.join(row)
            if heuristic == 'single':
                chunks = single_word_heurestic(sentence)
            elif heuristic == 'double':
                chunks = double_word_heurestic(sentence)
            f.write(str(chunks)+"\n")

if __name__ == "__main__":
    single_word_hu_ouputs = ["data/en_new_notation_single_word_hu/test.txt", "data/en_new_notation_single_word_hu/val.txt", "data/en_new_notation_single_word_hu/train.txt"]
    double_word_hu_ouputs = ["data/en_new_notation_double_word_hu/test.txt", "data/en_new_notation_double_word_hu/val.txt", "data/en_new_notation_double_word_hu/train.txt"]
    original_data_files = ["data/en_sentences/test.pkl", "data/en_sentences/val.pkl", "data/en_sentences/train.pkl"]

    for i in range(3):
        original_sentences_file = original_data_files[i]
        single_output = single_word_hu_ouputs[i]
        double_output = double_word_hu_ouputs[i]
        heuristic_output_in_bracketed_form(original_sentences_file, single_output, 'single')
        heuristic_output_in_bracketed_form(original_sentences_file, double_output, 'double')

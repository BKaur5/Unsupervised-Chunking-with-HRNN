'''
This file purpose is to evaluate a heuristic by calculating f1 score and accuracy by using a function named evaluate_heuristic 
'''
from library.eval_spans import get_chunk_spans, get_f1_score

def get_tags(file_path):
    '''
    This function reads a file with sentences in bracketed notation and gives the BI tags for each sentence in it as a list of lists.
    Each list in the main list corresponds to each sentence and its BI tags
    '''
    all_sentence_tags = []
    with(open(file_path, 'r')) as f:
        sentences = f.readlines()
        for sentence in sentences:
            sentence_tags = []
            split_in_chunks = sentence.split('] [')
            for chunk in split_in_chunks:
                sentence_tags.append('B')
                split_in_words = chunk.split(' ')
                if len(split_in_words) > 1:
                    for i in range(len(split_in_words) - 1):
                        sentence_tags.append('I')
            all_sentence_tags.append(sentence_tags) 
    return all_sentence_tags


def evaluate_heuristic(gt_sentences_file, pred_sentences_file):
    # First we read the input files with sentences in bracketed notations and get the BI tags for them
    ground_truths = get_tags(gt_sentences_file)
    predicted_labels = get_tags(pred_sentences_file)
    # Next we calculate the spans for the ground truths and the predictions
    ground_truth_spans = [get_chunk_spans(sentence) for sentence in ground_truths]
    predicted_spans = [get_chunk_spans(sentence) for sentence in predicted_labels]
    # We use the spans to calculate the f1 score
    f1  = get_f1_score(ground_truth_spans, predicted_spans)
    
    # following is the calculation of accuracy, which is correctly identified tags / total tags 
    correct_tags = 0
    total_tags = 0

    for true_tags, pred_tags in zip(ground_truths, predicted_labels):
        for true_tag, pred_tag in zip(true_tags, pred_tags):
            total_tags += 1
            if true_tag == pred_tag:
                correct_tags += 1

    accuracy = (correct_tags / total_tags) * 100 if total_tags > 0 else 0
    return f1, accuracy
            

if __name__ == "__main__":
    HEURISTIC_PRED_SENTENCES = 'data\\en_new_notation_pseudo_labels\\test.txt'
    GROUND_TRUTH_SENTENCES = 'data\\en_new_notation_ground_truths\\test.txt'

    fscore, acc = evaluate_heuristic(GROUND_TRUTH_SENTENCES, HEURISTIC_PRED_SENTENCES)
    print( " __________________________________")
    print(f"| Test:")
    print(f"|     F1:       {fscore}")
    print(f"|     Accuracy: {acc}")
    print( "|__________________________________")
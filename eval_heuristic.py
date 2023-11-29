'''
This file's purpose is to evaluate a heuristic by calculating f1 score and accuracy by using a function named evaluate_heuristic 
'''
from library.eval_spans import get_chunk_spans, get_f1_score
from library.labelled_entry import LabelledEntry

def get_tags(file_path):
    '''
    This function reads a file with sentences in bracketed notation and gives the BI tags for each sentence in it as a list of lists.
    Each list in the main list corresponds to each sentence and its BI tags
    '''
    # Get chunks as class entries as a list of lists
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

def get_chunk_entries(file_path):
    sentences_chunks = []
    with(open(file_path, 'r')) as f:
        sentences = f.readlines()
        for sentence in sentences:
            chunks = LabelledEntry.load_from_bracket_format(sentence)
            sentences_chunks.append(chunks)
    print(len(sentences_chunks))
    return sentences_chunks

def evaluate_heuristic(gt_entries, pred_entries):
    ground_truth_spans = [get_chunk_spans(sentence) for sentence in gt_entries]
    predicted_spans = [get_chunk_spans(sentence) for sentence in pred_entries]
    # We use the spans to calculate the f1 score
    f1  = get_f1_score(ground_truth_spans, predicted_spans)
    # Calculate the number of correct tags and total tags
    correct_tags = 0
    total_tags = 0
    for i in range(len(ground_truth_spans)):
        gt_span = ground_truth_spans[i]
        pred_span = predicted_spans[i]
        b_gt = set([span[0] for span in gt_span])
        b_pred = set([span[0] for span in pred_span])
        i_gt = set([index for span in gt_span for index in range(span[0] + 1, span[1])])
        i_pred = set([index for span in pred_span for index in range(span[0] + 1, span[1])])
        total_tags += len(b_gt) + len(i_gt)
        correct_tags += len(b_gt.intersection(b_pred)) + len(i_gt.intersection(i_pred))  
    # Calculate accuracy
    if total_tags > 0:
        accuracy = (correct_tags / total_tags) * 100
    else:
        accuracy = 0

    return f1, accuracy
            

if __name__ == "__main__":
    HEURISTIC_PRED_SENTENCES = 'data\\en_new_notation_pseudo_labels\\test.txt'
    GROUND_TRUTH_SENTENCES = 'data\\en_new_notation_ground_truths\\test.txt'
    ground_truth_entries = get_chunk_entries(GROUND_TRUTH_SENTENCES)
    heuristic_entries = get_chunk_entries(HEURISTIC_PRED_SENTENCES)
    fscore, acc = evaluate_heuristic(ground_truth_entries, heuristic_entries)
    print( " __________________________________")
    print(f"| Test:")
    print(f"|     F1:       {fscore}")
    print(f"|     Accuracy: {acc}")
    print( "|__________________________________")
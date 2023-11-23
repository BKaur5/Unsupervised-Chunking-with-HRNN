def get_chunk_spans(sentence_tags):
    spans = []
    start = None

    for i, tag in enumerate(sentence_tags):
        if tag == 'B':
            if start is not None:
                # End the current span on a new 'B'
                spans.append((start, i - 1))
            start = i

    # Handle the case where a span continues to the end of the sequence
    if start is not None:
        spans.append((start, len(sentence_tags) - 1))

    return spans


def get_f1_score(ground_truth_spans, predicted_spans):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for true_spans, pred_spans in zip(ground_truth_spans, predicted_spans):
        true_set = set(true_spans)
        pred_set = set(pred_spans)
        true_positive += len(true_set.intersection(pred_set))
        false_positive += len(pred_set - true_set)
        false_negative += len(true_set - pred_set)
     
    precision =  100 * true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
    recall = 100 * true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return f1

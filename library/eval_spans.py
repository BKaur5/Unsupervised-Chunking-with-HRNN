def get_chunk_spans(entry):
    begins = [0]
    for prev_chunk in entry.chunks[:-1]:
        begins.append(begins[-1]+len(prev_chunk))
    ends = [b for b in begins[1:]]
    ends.append(sum([len(chunk) for chunk in entry.chunks]))
    return list(zip(begins, ends))


def get_f1_score(ground_truth_spans, predicted_spans):
    true_positive = 0
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

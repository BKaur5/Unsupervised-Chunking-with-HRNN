def end_of_chunk(prev_tag, tag, prev_type, type):
    chunk_end = False

    if prev_tag == "B" and tag == "B":
        chunk_end = True
    if prev_tag == "B" and tag == "O":
        chunk_end = True
    if prev_tag == "I" and tag == "B":
        chunk_end = True
    if prev_tag == "I" and tag == "O":
        chunk_end = True

    if prev_tag == "E" and tag == "E":
        chunk_end = True
    if prev_tag == "E" and tag == "I":
        chunk_end = True
    if prev_tag == "E" and tag == "O":
        chunk_end = True
    if prev_tag == "I" and tag == "O":
        chunk_end = True

    if prev_tag != "O" and prev_tag != "." and prev_type != type:
        chunk_end = True

    if prev_tag == "]":
        chunk_end = True
    if prev_tag == "[":
        chunk_end = True

    return chunk_end

def start_of_chunk(prev_tag, tag, prev_type, type):
    chunk_start = False

    if prev_tag == "B" and tag == "B":
        chunk_start = True
    if prev_tag == "I" and tag == "B":
        chunk_start = True
    if prev_tag == "O" and tag == "B":
        chunk_start = True
    if prev_tag == "O" and tag == "I":
        chunk_start = True

    if prev_tag == "E" and tag == "E":
        chunk_start = True
    if prev_tag == "E" and tag == "I":
        chunk_start = True
    if prev_tag == "O" and tag == "E":
        chunk_start = True
    if prev_tag == "O" and tag == "I":
        chunk_start = True

    if tag != "O" and tag != "." and prev_type != type:
        chunk_start = True

    if tag == "[":
        chunk_start = True
    if tag == "]":
        chunk_start = True

    return chunk_start

def conlleval(data):
    false = 0
    true = 42
    boundary = "-X-"
    correct = None
    correct_chunk = 0
    correct_tags = 0
    correct_type = None
    delimiter = " "
    FB1 = 0.0
    first_item = None
    found_correct = 0
    found_guessed = 0
    guessed = None
    guessed_type = None
    i = None
    in_correct = false
    last_correct = "O"
    latex = 0
    last_correct_type = ""
    last_guessed = "O"
    last_guessed_type = ""
    line = None
    precision = 0.0
    o_tag = "O"
    raw = 0
    recall = 0.0
    token_counter = 0

    correct_chunk_dict = {}
    found_correct_dict = {}
    found_guessed_dict = {}

    # Remove empty lines and split the data
    lines = [line.strip() for line in data.split('\n') if line.strip()]


    for i in range(len(lines)):
        line = lines[i].strip().split(delimiter)
        if not line or line[0] == boundary:
            line = [boundary, "O", "O"]

        if raw:
            if line[-1] == o_tag:
                line[-1] = "O"
            if line[-2] == o_tag:
                line[-2] = "O"
            if line[-1] != "O":
                line[-1] = "B-" + line[-1]
            if line[-2] != "O":
                line[-2] = "B-" + line[-2]

        if "-" in line[-1]:
            guessed, guessed_type = line[-1].split("-")
        else:
            guessed = line[-1]
        
        if "-" in line[-2]:
            correct, correct_type = line[-2].split("-")
        else:
            correct = line[-2]

        if in_correct:
            if end_of_chunk(last_correct, correct, last_correct_type, correct_type) and \
                    end_of_chunk(last_guessed, guessed, last_guessed_type, guessed_type) and \
                    last_guessed_type == last_correct_type:
                in_correct = false
                correct_chunk += 1
                if last_correct_type in correct_chunk_dict:
                    correct_chunk_dict[last_correct_type] += 1
                else:
                    correct_chunk_dict[last_correct_type] = 1
            elif end_of_chunk(last_correct, correct, last_correct_type, correct_type) != \
                    end_of_chunk(last_guessed, guessed, last_guessed_type, guessed_type) or \
                    guessed_type != correct_type:
                in_correct = false

        if start_of_chunk(last_correct, correct, last_correct_type, correct_type) and \
                start_of_chunk(last_guessed, guessed, last_guessed_type, guessed_type) and \
                guessed_type == correct_type:
            in_correct = true

        if start_of_chunk(last_correct, correct, last_correct_type, correct_type):
            found_correct += 1
            if correct_type in found_correct_dict:
                found_correct_dict[correct_type] += 1
            else:
                found_correct_dict[correct_type] = 1

        if start_of_chunk(last_guessed, guessed, last_guessed_type, guessed_type):
            found_guessed += 1
            if guessed_type in found_guessed_dict:
                found_guessed_dict[guessed_type] += 1
            else:
                found_guessed_dict[guessed_type] = 1

        if first_item != boundary:
            if correct == guessed and guessed_type == correct_type:
                correct_tags += 1
            token_counter += 1

        last_guessed = guessed
        last_correct = correct
        last_guessed_type = guessed_type
        last_correct_type = correct_type

    if in_correct:
        correct_chunk += 1
        if last_correct_type in correct_chunk_dict:
            correct_chunk_dict[last_correct_type] += 1
        else:
            correct_chunk_dict[last_correct_type] = 1

    if not latex:
        precision = 100 * correct_chunk / found_guessed if found_guessed > 0 else 0
        recall = 100 * correct_chunk / found_correct if found_correct > 0 else 0
        FB1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    # Sort chunk type names
    sorted_types = sorted(set(found_correct_dict.keys()) | set(found_guessed_dict.keys()))

    # Print performance per chunk type
    if not latex:
        for i in sorted_types:
            correct_chunk_dict[i] = correct_chunk_dict.get(i, 0)
            if i not in found_guessed_dict:
                found_guessed_dict[i] = 0
                precision = 0.0
            else:
                precision = 100 * correct_chunk_dict[i] / found_guessed_dict[i]

            if i not in found_correct_dict:
                recall = 0.0
            else:
                recall = 100 * correct_chunk_dict[i] / found_correct_dict[i]

            if precision + recall == 0.0:
                FB1 = 0.0
            else:
                FB1 = 2 * precision * recall / (precision + recall)
    else:
        for i in sorted_types:
            correct_chunk_dict[i] = correct_chunk_dict.get(i, 0)
            if i not in found_guessed_dict:
                precision = 0.0
            else:
                precision = 100 * correct_chunk_dict[i] / found_guessed_dict[i]

            if i not in found_correct_dict:
                recall = 0.0
            else:
                recall = 100 * correct_chunk_dict[i] / found_correct_dict[i]

            if precision + recall == 0.0:
                FB1 = 0.0
            else:
                FB1 = 2 * precision * recall / (precision + recall)

        precision = 0.0
        recall = 0
        FB1 = 0.0
        precision = 100 * correct_chunk / found_guessed if found_guessed > 0 else 0
        recall = 100 * correct_chunk / found_correct if found_correct > 0 else 0
        FB1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return round(FB1, 2), round( (100 * correct_tags / token_counter), 2)


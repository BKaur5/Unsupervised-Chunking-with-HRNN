'''
This file contains definitions for heurestics that can be used in the process
'''

import random
from library.labelled_entry import LabelledEntry

# change to the square notation instead of B I
def single_word_heurestic(sentence):
    return LabelledEntry(list(map(lambda x: [x], sentence.split())))

def double_word_heurestic(sentence):
    words = sentence.split()
    chunks = [words[i:i+2] for i in range(0, len(words), 2)]
    return LabelledEntry(chunks)

def bernoulli_random(sentence, p=0.5):
    words = sentence.split()
    chunks = [[words[0]]]
    for word in words[1:]:
        if random.rand()>p:
            chunks.append([])
        chunks[-1].append(word)
    return LabelledEntry(chunks)
            
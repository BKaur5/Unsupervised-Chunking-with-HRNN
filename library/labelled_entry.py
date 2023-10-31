class LabelledEntry:
    def __init__(self, bracketed_sent):
        self.chunks = list(map(str.split, bracketed_sent[1:-1].split('] [')))

    def load_from_IB_format(words, tags):
        # list(zip(map(lambda x: x[0], test.pkl), test_tag.pkl))
        bracketed_sent = ''
        for word, tag in zip(words, tags):
            if tag=='B':
                bracketed_sent += '] ['
            bracketed_sent += word+' '
        bracketed_sent = bracketed_sent[2:-1]
        return LabelledEntry(bracketed_sent)
    
    def load_from_12_format(words, tags):
        # test.pkl
        bracketed_sent = '['
        tags[-1] = '-1'
        for word, tag in zip(words, tags):
            bracketed_sent += word
            if tag=='2':
                bracketed_sent += '] ['
        return LabelledEntry(bracketed_sent)
        
    def __str__(self):
        return '['+'] ['.join(list(map(' '.join, self.chunks)))+']'
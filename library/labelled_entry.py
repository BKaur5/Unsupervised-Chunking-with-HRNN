class LabelledEntry:
    def __init__(self, chunks):
        self.chunks = chunks
    
    def load_from_bracket_format(bracketed_sent):
        chunks = list(map(str.split, bracketed_sent[1:-1].split('] [')))
        return LabelledEntry(chunks)

    def load_from_IB_format(words, tags):
        # list(zip(map(lambda x: x[0], test.pkl), test_tag.pkl))
        bracketed_sent = ''
        for word, tag in zip(words, tags):
            if tag=='B':
                bracketed_sent += '] ['
            bracketed_sent += word+' '
        bracketed_sent = bracketed_sent[2:]
        return LabelledEntry.load_from_bracket_format(bracketed_sent)
    
    def load_from_12_format(words, tags):
        # test.pkl
        bracketed_sent = '['
        tags[-1] = '-1'
        
        for word, tag in zip(words, tags):
            bracketed_sent += word
            if tag=='2':
                bracketed_sent += '] ['
            elif tag == '1':
                bracketed_sent += ' '

        bracketed_sent += ']'
            
            # bracketed_sent += ' '
        return LabelledEntry.load_from_bracket_format(bracketed_sent)
    
    @property
    def sentence(self):
        return ' '.join(map(' '.join, self.chunks))

    def __str__(self):
        return '['+'] ['.join(list(map(' '.join, self.chunks)))+']'
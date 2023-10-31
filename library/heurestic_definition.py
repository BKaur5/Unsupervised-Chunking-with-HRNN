'''
This file contains definitions for heurestics that can be used in the process
'''


import pickle

def read_files(test_data_path,test_tags_path):
    test_data = pickle.load(open(test_data_path, 'rb'))
    test_tags = pickle.load(open(test_tags_path, 'rb'))
    return test_data,test_tags

def single_word_heurestic(test_data_path,test_tags_path,**args):
    test_data, test_tags = read_files(test_data_path,test_tags_path)
    output = []
    for words,tags in test_data:
        markers = ['B'] * len(tags)
        output.append(markers)
    
    return output

def double_word_heurestic(test_data_path,test_tags_path,**args):
    test_data, test_tags = read_files(test_data_path,test_tags_path)
    output = []
    for words,tags in test_data:
        markers = ['B','I'] * (len(tags)//2)
        if len(tags) % 2 != 0:
            markers.append('B')
        output.append(markers)

    return output
        
            



# (['It', 'also', 'will', 'hand', 'Unilab', 'new', 'markets'], ['2', '2', '2', '2', '2', '1', '1']) 
# ['B', 'B', 'B', 'I', 'B', 'B', 'I']
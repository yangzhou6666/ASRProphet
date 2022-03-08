'''This file is for generating new test cases. (RQ3)'''

import nltk
nltk.download('punkt')

def get_nouns_list(text):
    '''
    This function returns a list of nouns in the input text.
    Return a list: [(position, word)]
    '''
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    print(tagged)
    # get position of nouns
    nouns = [(i, word) for i, (word, tag) in enumerate(tagged) if tag == 'NN' or tag == 'NNS']
    return nouns

if __name__=='__main__':
    text = "I have a dogs"
    nouns = get_nouns_list(text)
    print(nouns)


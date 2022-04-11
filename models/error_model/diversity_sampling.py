
from g2p_en import G2p
import numpy as np


MASK = '<mask>'
SOS = '<s>'
EOS = '</s>'

# prepare global variables for phoneme related info
get_phoneme_seq = G2p()
phone_vocab = get_phoneme_seq.phonemes[0:4]
phone_vocab += np.unique([item if len(item)!=3 else item[:-1] for item in get_phoneme_seq.phonemes[4:]]).tolist()
phone_vocab += [MASK, ' ']

class DiversityDrivenSampler():
    '''
    The class for two diversity-drive test case selection methods:
    (1) A Method for the Extraction of Phonetically-Rich Triphone Sentences
    (2) A submodular optimization approach to sentence set selection
    '''

    def __init__(self):
        pass

    def get_phonemes(self, text):
        '''Given a text, return the list of phonemes'''
        phone_list = get_phoneme_seq(text)
        phone_list = [item if len(item)!=3 else item[:-1] for item in phone_list]
        phone_list = [item for item in phone_list if item not in {"'"}]
        return phone_list


if __name__ == '__main__':
    text = 'I have a doy.'
    phones = get_phoneme_seq(text)
    print(phones)
    sampler = DiversityDrivenSampler()
    print(sampler.get_phonemes(text))

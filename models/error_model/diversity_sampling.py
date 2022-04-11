
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

    def get_sentence_wise_phone_freqs(self):
        '''Compute the phone freq for each sentence stored in the class'''
        sent_wise_phone_freqs = []
        for text in self.sentences:
            freq = np.zeros(len(self.phone_vocab))
            phones = self.get_phonemes(text)
            for phone in phones:
                freq[self.phone_to_id[phone]] +=1
            self.phone_sentences.append(phones)
            sent_wise_phone_freqs.append(freq)
        return sent_wise_phone_freqs


if __name__ == '__main__':
    text = 'I have a doy.'
    phones = get_phoneme_seq(text)
    print(phones)
    sampler = DiversityDrivenSampler()
    print(sampler.get_phonemes(text))

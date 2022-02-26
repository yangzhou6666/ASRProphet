import sys
sys.path.append("..")

from quartznet_asr.metrics import __levenshtein, word_error_rate
from power import Levenshtein, ExpandedAlignment
from power.aligner import PowerAligner
from transformers import AutoTokenizer
from transformers import BertTokenizer, BertForTokenClassification
import torch
import re
def get_label(path: str):
    '''
    Given the path to the reference and transcriptions
    return the model inputs and correpsonding labels
    '''
    inputs = []
    labels = []
    with open(path_to_result, 'r') as f:
        for chunk in f.read().strip().split('\n\n'):
            data = chunk.split('\n')
            WER = float(data[1][5:])
            CER = float(data[2][5:])
            ref = data[3][5:]
            hyp = data[4][5:]

            # convert string to list of words
            re_expr = "\'| "
            ref = [x for x in re.split(re_expr, ref.strip()) if x]
            hyp = [x for x in re.split(re_expr, hyp.strip()) if x]

            lev = Levenshtein.align(ref, hyp) 
            lev.editops()
            alignment = lev.expandAlign()

            '''
            The data structure of alignment:
            REF:  anatomy         hi  hello  democracy  
            HYP:  and that to me      hello  de mo gracy
            Eval: S               D   C      S    
            '''

            '''
            Explaination for symbols:
            S: Single word subsitutions
            D: Deletion
            I: Insertion, a hyoothesis word with no aligned reference word
            C: Correct.
            '''

            label = []
            for x in alignment.align:
                if x == 'I':
                    continue
                elif x == 'C':
                    label.append(0)
                else:
                    label.append(1)
            assert len(label) == len(alignment.ref())
            labels.append(label)
            inputs.append(alignment.ref())
    
    return inputs, labels

if __name__ == "__main__":
    path_to_result = '/workspace/data/l2arctic/processed/ASI/manifests/train/quartznet/error_model_tts/500/seed_1/test_out_ori.txt'


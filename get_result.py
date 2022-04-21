from operator import index
import numpy as np
import pandas as pd
import os

import helper
import json

import os 

from models.error_model.error_model_sampling import ErrorModelSampler

DATA = "data/l2arctic/processed"

asrs = ["quartznet", "hubert", "wav2vec-base"]
datasets = ["YBAA", "ZHAA", "ASI", "TNI", "NCC", "TXHC", "EBVS", "ERMS", "YDCK", "YKWK", "THV", "TLV"]
    

def save_performance_of_original_model_on_test_set():
    """Measure the WER and CER using the original model"""

    tools = ["random", "error_model_triphone_rich","error_model_pure_diversity", "error_model_without_diversity_enhancing", "error_model", "asrevolve_error_model_real", "word_error_predictor_real/no_word_enhance", "word_error_predictor_real/word_enhance"]

    os.makedirs("result/RQ1", exist_ok=True)
    
    raw_results = {}
    for asr in asrs :
        raw_results[asr] = {}
        for dataset in datasets :
            raw_results[asr][dataset] = {}
            for tool in tools :
                df = helper.gather_result_RQ1(asr, dataset, tool)                
                fpath = f"result/RQ1/{asr}_{dataset}_{tool.replace('/','_')}.csv"
                raw_results[asr][dataset][tool] = fpath
                df.to_csv(fpath, index=False)
    
    with open('result/RQ1.json', 'w') as fp:
        json.dump(raw_results, fp)    


def save_performance_of_finetuned_model_on_test_set():
    """Measure the WER and CER using the fine-tuned model"""
    
    tools = ["random", "triphone_rich", "pure_diversity", "icassp_without_diversity_enhancing_real_mix", "icassp_real_mix", "asrevolve_error_model_real", "word_error_real_mix/no_word_enhance", "word_error_real_mix/word_enhance"]
    
    os.makedirs("result/RQ2", exist_ok=True)
    
    raw_results = {}
    for asr in asrs :
        raw_results[asr] = {}
        for dataset in datasets :
            raw_results[asr][dataset] = {}
            for tool in tools :                
                df = helper.gather_result_RQ2(asr, dataset, tool)
                fpath = f"result/RQ2/{asr}_{dataset}_{tool.replace('/','_')}.csv"
                df.to_csv(fpath, index=False)
                raw_results[asr][dataset][tool] = fpath

    with open('result/RQ2.json', 'w') as fp:
        json.dump(raw_results, fp)    
    

def save_performance_of_original_model_on_the_dataset():
    """Measure the WER and CER using the original model"""
    
    raw_results = {}
    for asr in asrs :
        raw_results[asr] = {}
        for dataset in datasets :
            
            raw_results[asr][dataset] = {}
            
            for t in ["all", "seed", "dev", "seed_plus_dev", "test"] :
                wer, cer = helper.get_original_performance(asr, dataset, type=t)
                raw_results[asr][dataset][t] = {"wer":wer, "cer":cer}
    
    
    with open('result/original.json', 'w') as fp:
        json.dump(raw_results, fp)  
    

def save_the_distance_between_the_triphone_rich_selected_samples_and_ideal_distributions():
    """Measure the distance between the triphone-rich selected samples and ideal distributions"""

    tools = ["random", "error_model_triphone_rich","error_model_pure_diversity", "error_model_without_diversity_enhancing", "error_model", "asrevolve_error_model_real", "word_error_predictor_real/no_word_enhance", "word_error_predictor_real/word_enhance"]
    
    raw_results = {}
    for asr in asrs :
        raw_results[asr] = {}
        for dataset in datasets :
            raw_results[asr][dataset] = {}
            for tool in tools :
                raw_results[asr][dataset][tool] = {}
                for seed in [1, 2, 3] :
                    raw_results[asr][dataset][tool][seed] = []        
                    for size in [100, 200, 300, 400] :
                        if tool == "random":
                            fpath = f"{DATA}/{dataset}/manifests/train/{tool}/{size}/seed_{seed}/train.json"
                        elif tool in ["word_error_predictor_real/no_word_enhance", "word_error_predictor_real/word_enhance"] :
                            fpath = f"{DATA}/{dataset}/manifests/train/{asr}/word_error_predictor_real/{size}/{tool.split('/')[-1]}/seed_{seed}/train.json"
                        else :
                            fpath = f"{DATA}/{dataset}/manifests/train/{asr}/{tool}/{size}/seed_{seed}/train.json"
                        
                        if os.path.exists(fpath):
                            sampler = ErrorModelSampler(fpath, "triphone_rich", error_model_weights=None, verbose=False)
                            assert len(sampler.phone_freqs) == len(sampler.ideal_triphone_dists)
                            triphone_dists = sampler.phone_freqs / sampler.phone_freqs.sum()
                            assert 0.95 < triphone_dists.sum() and triphone_dists.sum() < 1.05
                            assert 0.95 < sampler.ideal_triphone_dists.sum() and sampler.ideal_triphone_dists.sum() < 1.05
                            distance = sampler.compute_euclidean_distance(triphone_dists, sampler.ideal_triphone_dists)
                        else :
                            print(f"Missing: {fpath}")
                            distance = -999
                            
                        raw_results[asr][dataset][tool][seed].append(distance)
                    
    with open('result/triphone_rich.json', 'w') as fp:
        json.dump(raw_results, fp)
        
def save_the_value_of_phoneme_submodular_function_from_selected_samples():
    """Measure the value of phoneme submodular function from selected samples"""

    tools = ["random", "error_model_triphone_rich","error_model_pure_diversity", "error_model_without_diversity_enhancing", "error_model", "asrevolve_error_model_real", "word_error_predictor_real/no_word_enhance", "word_error_predictor_real/word_enhance"]
    
    raw_results = {}
    for asr in asrs :
        raw_results[asr] = {}
        for dataset in datasets :
            raw_results[asr][dataset] = {}
            for tool in tools :
                raw_results[asr][dataset][tool] = {}
                for seed in [1, 2, 3] :
                    raw_results[asr][dataset][tool][seed] = []        
                    for size in [100, 200, 300, 400] :
                        if tool == "random":
                            fpath = f"{DATA}/{dataset}/manifests/train/{tool}/{size}/seed_{seed}/train.json"
                        elif tool in ["word_error_predictor_real/no_word_enhance", "word_error_predictor_real/word_enhance"] :
                            fpath = f"{DATA}/{dataset}/manifests/train/{asr}/word_error_predictor_real/{size}/{tool.split('/')[-1]}/seed_{seed}/train.json"
                        else :
                            fpath = f"{DATA}/{dataset}/manifests/train/{asr}/{tool}/{size}/seed_{seed}/train.json"
                        if os.path.exists(fpath):
                            sampler = ErrorModelSampler(fpath, "pure_diversity", error_model_weights=None, verbose=False)
                            log_frequency = sampler.get_f2(sampler.phone_freqs)
                            assert len(sampler.ideal_phone_dists) == len(log_frequency) 
                            phone_submodular_value = np.dot(sampler.ideal_phone_dists, log_frequency )
                            assert phone_submodular_value > 0
                            assert isinstance(phone_submodular_value, float)
                        else :
                            print(f"Missing: {fpath}")
                            phone_submodular_value = -999
                        
                        raw_results[asr][dataset][tool][seed].append(phone_submodular_value)
                        
    with open('result/phoneme_submodular_function.json', 'w') as fp:
        json.dump(raw_results, fp)


def save_the_number_of_test_cases_from_selected_samples():
    """Measure the number of test cases from selected samples"""

    tools = ["random", "error_model_triphone_rich","error_model_pure_diversity", "error_model_without_diversity_enhancing", "error_model", "asrevolve_error_model_real", "word_error_predictor_real/no_word_enhance", "word_error_predictor_real/word_enhance"]
    
    raw_results = {}
    for asr in asrs :
        raw_results[asr] = {}
        for dataset in datasets :
            raw_results[asr][dataset] = {}
            for tool in tools :
                raw_results[asr][dataset][tool] = {}
                for seed in [1, 2, 3] :
                    raw_results[asr][dataset][tool][seed] = []        
                    for size in [100, 200, 300, 400] :
                        if tool == "random":
                            fpath = f"{DATA}/{dataset}/manifests/train/{tool}/{size}/seed_{seed}/train.json"
                        elif tool in ["word_error_predictor_real/no_word_enhance", "word_error_predictor_real/word_enhance"] :
                            fpath = f"{DATA}/{dataset}/manifests/train/{asr}/word_error_predictor_real/{size}/{tool.split('/')[-1]}/seed_{seed}/train.json"
                        else :
                            fpath = f"{DATA}/{dataset}/manifests/train/{asr}/{tool}/{size}/seed_{seed}/train.json"
                        
                        
                        if os.path.exists(fpath):
                            test_cases = open(fpath).readlines()
                            assert len(test_cases) > 0 
                            number_of_test_cases = len(test_cases)
                        else :
                            print(f"Missing: {fpath}")
                            number_of_test_cases = -999
                        
                        raw_results[asr][dataset][tool][seed].append(number_of_test_cases)
                        
    with open('result/number_of_test_cases.json', 'w') as fp:
        json.dump(raw_results, fp)

if __name__ == "__main__":
    
    # save_performance_of_original_model_on_test_set()
    # save_performance_of_finetuned_model_on_test_set()
    # save_performance_of_original_model_on_the_dataset()
    # save_the_distance_between_the_triphone_rich_selected_samples_and_ideal_distributions()
    save_the_number_of_test_cases_from_selected_samples()
    
    
    
    
      
    
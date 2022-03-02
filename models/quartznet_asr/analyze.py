import prettytable as pt
import os


def analyze_result(log_path):
    if os.path.exists(log_path): 
        with open(log_path) as f:
            contents = f.readlines()
            WER = contents[-6].strip()
            WER = WER.split(" ")[-1]

            CER = contents[-2].strip()
            CER = CER.split(" ")[-1]
    else:
        print("\033[0;37;41m\t!!Data Missing!!\033[0m")
        WER = -1
        CER = -1
    assert len(WER) < 6
    return WER, CER

tb = pt.PrettyTable()
tb.field_names = ['Dataset', 'Seed', 'Size', 'Ran-WER', 'Ran-CER']


def compare_original():
    # Compare the original performance
    for dataset in ['ASI', 'RRBI']:
        data_path = f'../../data/l2arctic/processed/{dataset}/manifests/train/quartznet/word_error_predictor_tts/'
        for seed in ['seed_1', 'seed_2', 'seed_3']:
            
            tb = pt.PrettyTable()
            tb.field_names = ['Dataset', 'Seed', 'Size', 'WER', 'CER']  
            for size in [50, 75, 100, 150, 200, 300, 400, 500]:
                size = str(size)
                path_to_log = os.path.join(data_path, size, 'sum', seed, 'test_out_ori_log.txt')
                try:
                    WER, CER = analyze_result(path_to_log)
                except:
                    print(path_to_log)
                    WER = -1
                    CER = -1
                tb.add_row([dataset, seed, size, WER, CER])

            print(tb)


def compare_retrained():
    # Compare the re-trained performance
    for dataset in ['ASI', 'RRBI']:
        data_path = f'../pretrained_checkpoints/quartznet/finetuned/{dataset}/'
        model = "word_error_predictor_tts/sum"
        for seed in ['seed_1', 'seed_2', 'seed_3']:
            
            tb = pt.PrettyTable()
            tb.field_names = ['Dataset', 'Seed', 'Size', 'WER', 'CER']  
            for size in [50, 75, 100, 150, 200, 300, 400, 500]:
                size = str(size)
                path_to_log = os.path.join(data_path, size, seed, model, 'test_infer_log.txt')
                try:
                    WER, CER = analyze_result(path_to_log)
                except:
                    print(path_to_log)
                    WER = -1
                    CER = -1
                tb.add_row([dataset, seed, size, WER, CER])

            print(tb)

if __name__ == "__main__":
    compare_retrained()
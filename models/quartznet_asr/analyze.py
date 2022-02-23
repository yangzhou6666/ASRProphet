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

data_path = '../pretrained_checkpoints/quartznet/finetuned'

for dataset in ['ASI', 'RRBI']:
    for seed in ['seed_1', 'seed_2', 'seed_3']:
        
        tb = pt.PrettyTable()
        tb.field_names = ['Dataset', 'Seed', 'Size', 'Ran-WER', 'Ran-CER']  
        for size in [50, 75, 100, 150, 200, 300, 400, 500]:
            size = str(size)
            path_to_log = os.path.join(data_path, dataset, size, seed, 'error_model_tts', 'test_infer_log.txt')
            try:
                WER, CER = analyze_result(path_to_log)
            except:
                print(path_to_log)
                WER = -1
                CER = -1
            tb.add_row([dataset, seed, size, WER, CER])

        print(tb)
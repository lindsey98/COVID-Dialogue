from pytorch_pretrained_bert import BertTokenizer
import torch
import os
import codecs
import json

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

MAX_ENCODER_SIZE = 400
MAX_DECODER_SIZE = 100

def clean_dataset(dataset_file, json_file):
    '''
    对 dataset_file 文件进行清洗处理并储存为 json 文件
    '''
    f_in = open(dataset_file, "r")
    f_json = open(json_file, "w", encoding='utf-8')
    
    total = 0
    last_part = ""
    last_turn = 0
    last_dialog = {}
    last_user = ""
    
    Dialog_list = []
    
#     check_list = []
    
    while True:
        line = f_in.readline()
        if not line:
            break
            
        # Get description
        if line.strip() == "Description":
            last_part = "description"
            last_turn = 0
            last_dialog = {}
            last_list = []
            last_user = ""
            last_utterance = ""
            while True:
                line = f_in.readline()
                if (not line) or (line in ["\n", "\n\r"]):
                    break
                last_user = "Patient:" 
                sen = line.rstrip()
                if sen == "":
                    continue
                if sen[-1] not in ',?.!)~':
                    sen += '.'
#                 if sen in check_list: # remove check duplicate according to paper?
#                     last_utterance = "" 
#                 else:
                last_utterance = last_user + sen
#                     check_list.append(sen)
                break
            
        # Get dialogue
        elif line.strip() == "Dialogue":
            if last_part == "description" and len(last_utterance) > 0:
                last_part = "dialogue"
                last_user = "Patient:"
                last_turn = 1
                while True:
                    line = f_in.readline()
                    if (not line) or (line in ["\n", "\n\r"]):
                        last_user = ""
                        last_list.append(last_utterance)
                        
                        if  int(last_turn / 2) > 0: # must have at leat 1 question-response pair
                            temp = int(last_turn / 2)
                            last_dialog["Turn"] = temp
                            total += 1
                            last_dialog["Id"] = total
                            last_dialog["Dialogue"] = last_list[: temp * 2]
                            Dialog_list.append(last_dialog)
#                         else:
#                             print(last_dialog)
                            
                        break
                        
                    if line.strip() == "Patient:" or line.strip() == "Doctor:":
                        user = line.strip()
#                         print(user)
                        line = f_in.readline()
                        sen = line.rstrip()
                        if sen == "":
                            continue
                            
                        if sen[-1] not in ',?.!)~':
                              sen += '.'
                            
                        if user == last_user:
                            last_utterance = last_utterance + sen
                        else:
                            last_user = user
                            last_list.append(last_utterance)
                            last_turn += 1
                            last_utterance = user + sen
                            
    
    print ("Total Cases: ", total)
    json.dump(Dialog_list, f_json, ensure_ascii = False, indent = 4)
    f_in.close()
    f_json.close()    
    
def seq2token_ids(source_seqs, target_seq):
    # 可以尝试对source_seq进行切分
    encoder_input = []
    for source_seq in source_seqs:
        # 去掉 xx：
        encoder_input += tokenizer.tokenize(source_seq[3:]) + ["[SEP]"]

    decoder_input = ["[CLS]"] + tokenizer.tokenize(target_seq[3:])  # 去掉 xx：

    # 设置不得超过 MAX_ENCODER_SIZE 大小
    if len(encoder_input) > MAX_ENCODER_SIZE - 1:
        if "[SEP]" in encoder_input[-MAX_ENCODER_SIZE:-1]:
            idx = encoder_input[:-1].index("[SEP]", -(MAX_ENCODER_SIZE - 1))
            encoder_input = encoder_input[idx + 1:]

    encoder_input = ["[CLS]"] + encoder_input[-(MAX_ENCODER_SIZE - 1):]
    decoder_input = decoder_input[:MAX_DECODER_SIZE - 1] + ["[SEP]"]
    enc_len = len(encoder_input)
    dec_len = len(decoder_input)
    
    # conver to ids
    encoder_input = tokenizer.convert_tokens_to_ids(encoder_input)
    decoder_input = tokenizer.convert_tokens_to_ids(decoder_input)

    # mask
    mask_encoder_input = [1] * len(encoder_input)
    mask_decoder_input = [1] * len(decoder_input)

    # padding
    encoder_input += [0] * (MAX_ENCODER_SIZE - len(encoder_input))
    decoder_input += [0] * (MAX_DECODER_SIZE - len(decoder_input))
    mask_encoder_input += [0] * (MAX_ENCODER_SIZE - len(mask_encoder_input))
    mask_decoder_input += [0] * (MAX_DECODER_SIZE - len(mask_decoder_input))

    # turn into tensor
    encoder_input = torch.LongTensor(encoder_input)
    decoder_input = torch.LongTensor(decoder_input)
    
    mask_encoder_input = torch.LongTensor(mask_encoder_input)
    mask_decoder_input = torch.LongTensor(mask_decoder_input)

    return encoder_input, decoder_input, mask_encoder_input, mask_decoder_input



def get_splited_data_by_file(dataset_file):
    datasets = [[], [], []]

    with open(dataset_file, "r", encoding='utf-8') as f:
        json_data = f.read()
        data = json.loads(json_data)

    total_id_num = len(data)
    validate_idx = int(float(total_id_num) * 8 / 10)
    test_idx = int(float(total_id_num) * 9 / 10)

    datasets[0] = [d['Dialogue'] for d in data[:validate_idx]]
    datasets[1] = [d['Dialogue'] for d in data[validate_idx:test_idx]]
    datasets[2] = [d['Dialogue'] for d in data[test_idx:]]
    return datasets


if __name__ == "__main__":

    dataset_file = os.path.join(os.path.abspath('..'), 'COVID-Dialogue-Dataset-English.txt')
    json_file = os.path.join(os.path.abspath('..'), 'COVID-Dialogue-Dataset-English.json')

    clean_dataset(dataset_file, json_file)
    
    
    data = get_splited_data_by_file(json_file)

    print(f'Process the train dataset')
    make_dataset(data[0], 'train_data.pth')

    print(f'Process the validate dataset')
    make_dataset(data[1], 'validate_data.pth')

    print(f'Process the test dataset')
    make_dataset(data[2], 'test_data.pth')
    
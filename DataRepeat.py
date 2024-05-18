# We repeat the data when if it contains more than one slots
# for train and test
import os
def data_repeat(in_path, out_path_i, mode='train', type='slu'):
    """
    type: slu, parser
    
    """

    out_path = os.path.join(out_path_i, type, mode)

    if mode=='test':
       out_path = os.path.join(out_path_i, type, "train") 

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    in_file = os.path.join(in_path, mode, 'seq.in')
    out_file = os.path.join(in_path, mode, 'seq.out_' + type)

    input_set = set()

    with open(in_file, "r") as f_in, open(out_file, "r") as f_out:
        for line_in, line_out in zip(f_in, f_out):
            

            line_in = line_in.strip()  
            line_out = line_out.strip()  
            line_in = line_in.split()
            l2_list = line_out.split()
            if 'B-none' in l2_list:
                continue

            if ' '.join(line_in) in input_set:
                continue
            else:
                input_set.add(' '.join(line_in))
            
            num_B = 0
            repeated_out = []
            length = len(l2_list)

            for idx, label_i in enumerate(l2_list):

                if "!" in label_i:
                    label_i = label_i[:-1]
                if 'B' in label_i:
                    repeated_out.append(['O']*length)
                    repeated_out[num_B][idx]=label_i
                    num_B+=1
                if "I" in label_i:
                    repeated_out[num_B-1][idx]=label_i
            if num_B==0:
                repeated_out = [['O']*len(line_in)]

            for line in repeated_out:
                with open(os.path.join(out_path, 'seq.in'), "a") as in_write:
                    in_write.write(' '.join(line_in) + '\n')
                with open(os.path.join(out_path, 'seq.out'), "a") as out_write:
                    out_write.write(' '.join(line) + '\n')

    return input_set        
                    
for dataset in ['carslu', 'camrest', 'woz-hotel','woz-attr', 'atis']: 
    in_path = '/storage_fast/yxwu/my_slot_discovery/joint-induction/data_pre/' + dataset
    out_path = './data/'+ dataset
    for mode in ["train", "valid", "test"]:
        for type in ["slu", "parser"]:
            input_set = data_repeat(in_path, out_path, mode=mode, type=type)

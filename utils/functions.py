import os
import torch
import numpy as np
import pandas as pd
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME, CONFIG_NAME
import json
import copy
def normalize_num(n):
    known_nos = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    dct = {x: str(y) for y, x in enumerate(known_nos)}
    if n.lower() in dct:
        return dct[n.lower()]
    return n


def is_number(n):
    known_nos = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    n = n.strip().lower()
    return n.isdigit() or n in known_nos

def save_npy(npy_file, path, file_name):
    npy_path = os.path.join(path, file_name)
    np.save(npy_path, npy_file)

def load_npy(path, file_name):
    npy_path = os.path.join(path, file_name)
    npy_file = np.load(npy_path)
    return npy_file

def save_model(model, model_dir):

    save_model = model.module if hasattr(model, 'module') else model  
    model_file = os.path.join(model_dir, WEIGHTS_NAME)
    model_config_file = os.path.join(model_dir, CONFIG_NAME)
    torch.save(save_model.state_dict(), model_file)
    with open(model_config_file, "w") as f:
        f.write(save_model.config.to_json_string())

def restore_model(model, model_dir):
    output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
    model.load_state_dict(torch.load(output_model_file))
    return model

def save_results(args, test_results):
    
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    var = [args.dataset, args.method, args.backbone, args.known_cls_ratio, args.labeled_ratio, args.cluster_num_factor, args.seed]
    names = ['dataset', 'method', 'backbone', 'known_cls_ratio', 'labeled_ratio', 'cluster_num_factor', 'seed']
    vars_dict = {k:v for k,v in zip(names, var) }
    results = dict(test_results,**vars_dict)
    keys = list(results.keys())
    values = list(results.values())
    
    results_path = os.path.join(args.result_dir, args.results_file_name)
    
    if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
        ori = []
        ori.append(values)
        df1 = pd.DataFrame(ori,columns = keys)
        df1.to_csv(results_path,index=False)
    else:
        df1 = pd.read_csv(results_path)
        new = pd.DataFrame(results,index=[1])
        df1 = df1.append(new,ignore_index=True)
        df1.to_csv(results_path,index=False)
    data_diagram = pd.read_csv(results_path)
    
    print('test_results', data_diagram)

def eval_and_save(args, test_results, slot_value_hyp, slot_value_gt, data_type, slot_map=None):

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    var = [args.dataset, args.method, args.backbone, args.known_cls_ratio, args.labeled_ratio, args.cluster_num_factor, args.seed]
    names = ['dataset', 'method', 'backbone', 'known_cls_ratio', 'labeled_ratio', 'cluster_num_factor', 'seed']

    results_folder_name = ""
    for k, v in zip(names, var):
        results_folder_name += str(k) + '-' + str(v) + '-'

    results_path = os.path.join(args.result_dir, results_folder_name, data_type)
    print('results_path: ', results_path)
    if not os.path.exists(results_path):
        os.makedirs(results_path)    
    # save
    
    pred_labels_path = os.path.join(results_path, 'y_pred.npy')
    feats_path = os.path.join(results_path, 'feats.npy')
    try:
        np.save(pred_labels_path, test_results['y_pred'])
        del test_results['y_pred']
        np.save(feats_path, test_results['feats'])
        del test_results['feats']
    except:
        pass
    
    with open (os.path.join(results_path, 'samp_results.json'), 'w') as f:
        f.write(json.dumps(test_results, indent=4))

    with open (os.path.join(results_path, 'slot_value_hyp.json'), 'w') as f:
        f.write(json.dumps(slot_value_hyp, indent=4))
    
    with open (os.path.join(results_path, 'slot_value_gt.json'), 'w') as f:
        f.write(json.dumps(slot_value_gt, indent=4))

    slot_value_hyp_set = {slot:list(set(value)) for slot, value in slot_value_hyp.items()}
    slot_value_gt_set = {slot:list(set(value)) for slot, value in slot_value_gt.items()}
    with open (os.path.join(results_path, 'slot_value_hyp_set.json'), 'w') as f:
        f.write(json.dumps(slot_value_hyp_set, indent=4))
    with open (os.path.join(results_path, 'slot_value_gt_set.json'), 'w') as f:
        f.write(json.dumps(slot_value_gt_set, indent=4))
    
    # slot_mappig
    if slot_map is None:
        if args.dataset=="woz-attr":
            slot_value_gt_copy = copy.deepcopy(slot_value_gt)
            del slot_value_gt_copy["slot"]
            slot_map = slot_mapping(slot_value_gt_copy, slot_value_hyp)
        else:
            slot_map = slot_mapping(slot_value_gt, slot_value_hyp)
        value_map = value_mapping(slot_value_gt, slot_value_hyp)
    #print('slot_map: ', slot_map)
    with open (os.path.join(results_path, 'slot_map.json'), 'w') as f:
        f.write(json.dumps(slot_map, indent=4))
    # evaluator
    if args.dataset == 'camrest':
        slot_names = ['food', 'pricerange', 'area', 'slot']
    elif args.dataset == 'movies':
        slot_names = ['spatial_relation', 'timerange', 'object_location_type', 'movie_type', 'location_name', 'object_type', 'movie_name']
    elif args.dataset == 'woz-hotel':
        slot_names = ['req-stars', 'choice', 'req-phone', 'req-address', 'req-area', 'req-price', 'day', 'type', 'area', 'price', 'parking', 'internet', 'people', 'stars', 'stay']
        slot_names = ['slot', 'day', 'type', 'area', 'price', 'people', 'stars', 'stay']
    elif args.dataset == 'woz-attr':
        replace_request = True
        slot_names = ['slot', 'area', 'type']
    elif args.dataset == 'carslu':
        slot_names = ['food', 'pricerange', 'area', 'phone', 'type', 'address']
        slot_names = ['food', 'pricerange', 'area', 'slot', 'type']
    elif args.dataset == 'atis':
        slot_names = ['toloc.city_name','fromloc.city_name','depart_date.day_name','airline_name','depart_time.period_mod','flight_mod','depart_time.time_relative','arrive_date.month_name','arrive_date.day_number','meal','fromloc.state_code','connect','flight_days','toloc.airport_name','fromloc.state_name','airport_name','economy','aircraft_code','mod','airport_code','depart_time.start_time','depart_time.end_time','depart_date.year','restriction_code','arrive_time.start_time','toloc.airport_code','arrive_time.end_time','fromloc.airport_code','arrive_date.date_relative','return_date.date_relative','state_code','meal_code','day_name','period_of_day','stoploc.state_code','return_date.month_name','return_date.day_number','arrive_time.period_mod','toloc.country_name','days_code','return_time.period_of_day','time','today_relative','state_name','arrive_date.today_relative','return_time.period_mod','month_name','day_number','stoploc.airport_name','time_relative','return_date.today_relative','return_date.day_name']
    else:
        slot_names = []
    
    print('slot_names: ', slot_names)

    model_evaluator = GenericEvaluator('NN MODEL', slot_map, slot_names)
    
    test_results_select = {}  # {slot: value}
    test_results_before_select = {}  # {slot: value}
    recall_value_all = []
    for uttr, result in test_results.items():
        test_results_select[uttr] = {}
        test_results_before_select[uttr] = {}
        #turn_slu = {v:k for k, v in result['gt'].items()}
        #print('uttr: ',uttr)
        #print('result: ', result)
        turn_slu = {}
        if len(result['gt'].values())==0:
            print('uttr: ', uttr)
            continue
        for value, slot in result['gt'].items():
            if slot!='noslot':
                if not slot in turn_slu.keys():
                    turn_slu[slot] = []
                turn_slu[slot].append(value.lower())        
        # Maybe there are more than one values belonging to the same slot in slu_hyp
        #print('turn_slu: ', turn_slu)
        
        
        # use slot map
        '''
        slu_hyp = {}
        for value, slot_prob in result['hyp'].items():
            slot = slot_prob[0]
            new_slot = slot_map[slot]

            #if value=='number' or value=='numbers':
            #    value = 'phone'
            if new_slot!='none':
                if not new_slot in slu_hyp.keys():
                    slu_hyp[new_slot] = []
                value = value.lower()
                slu_hyp[new_slot].append(value)
        #print('slu_hyp: ', slu_hyp)
        for slot, value in slu_hyp.items():
            if len(value[0])>1:
                value_i = value[0]
                prob = np.array([eval(i) for i in value[1]])
                value_i = value_i[np.argsort(prob)[-1]]
                slu_hyp[new_slot]=[value_i]
        '''
        # use slot map
        
        slot_refine_attr = set(["address", "phone", "entrance", "area", "tell"])
        # if value in slot_refine_attr, "slot":"?"
        slot_refine_hotel = set(["address", "phone", "area", "tell", "number","code"])
        slot_refine_camrest = set(["address", "phone", "telephone","phones","name"])

        slu_hyp = {}  # slot: [value, prob, indices]
        for value, slot_prob in result['hyp'].items():
            slot = slot_prob[0]
            new_slot = slot_map[slot] # slot_map: id:slot_name
            value = value.lower()
            value = value.replace("range","price").replace("number", "phone")
            # result refine!!!
            '''
            if args.dataset=="camrest":
                if value in slot_refine_camrest:
                    new_slot = "slot"
                    #value = "?"
            if args.dataset=="carslu":
                if value in slot_refine_camrest:
                    new_slot = "slot"
            '''


            if args.dataset=="woz-attr":
                if value in slot_refine_attr:
                    new_slot = "slot"
                    value = "?"
            
            if args.dataset=="woz-hotel":
                # slot:?
                if value in slot_refine_hotel:
                    new_slot = "slot"
                    value = "?"
                # number for "people, stars, stay(nights)"
                if is_number(value):
                    number = normalize_num(value)
                    # if next token in ["people","peoples", "nights","night","star", "stars"]
                    ind = int(slot_prob[1][-1])
                    #print('ind: ', ind)
                    #print(' uttr.split(): ', len( uttr.split()))
                    try:
                        next_token = uttr.split()[ind+1]
                        if next_token in ["people","peoples"]:
                            new_slot = "people"
                        elif next_token in ["nights","night"]:
                            new_slot = "stay" 
                        elif next_token in ["star", "stars"]:
                            new_slot = "stars" 
                    except:
                        pass                        

            if new_slot!='noslot':
                if not new_slot in slu_hyp.keys():
                    slu_hyp[new_slot] = [[],[]]

                #if value=='number' or value=='numbers':
                #    value = 'phone'

                slu_hyp[new_slot][0].append(value)
                slu_hyp[new_slot][1].append(eval(slot_prob[1]))

        if args.dataset=="atis":
            uttr_list = uttr.split()
            #print(uttr_list)
            if "from" in uttr_list:
                slu_hyp['fromloc.city_name'] = [[uttr_list[int(uttr_list.index("from")+1)]], [1] ]
            if "to" in uttr_list:
                try:
                    slu_hyp['toloc.city_name'] = [[uttr_list[int(uttr_list.index("to")+1)]], [1] ]
                except:
                    pass

        test_results_before_select[uttr] ['hyp'] = copy.deepcopy(slu_hyp)
        test_results_before_select[uttr] ['gt'] = copy.deepcopy(turn_slu)


        slu_hyp_select = {}
        for slot, value_prob in slu_hyp.items():
            if len(value_prob[0])>1:
                value_i = value_prob[0]
                prob = np.array([value_prob[1]])
                value_select = value_i[np.argsort(prob)[0][-1]]
                
                slu_hyp_select[slot] = [value_select]
            else:
               
                slu_hyp_select[slot] = value_prob[0]

        '''
        # use value map
        slu_hyp = {}
        for value, slot in result['hyp'].items():
            new_slot = value_map[value]
            if not new_slot in slu_hyp.keys():
                slu_hyp[new_slot] = []
            slu_hyp[new_slot].append(value.lower())
        with open (os.path.join(results_path, 'value_map.json'), 'w') as f:
            f.write(json.dumps(value_map, indent=4))
        '''

        slu_hyp_copy = copy.deepcopy(slu_hyp_select)
        turn_slu_copy = copy.deepcopy(turn_slu)
        test_results_select[uttr] ['hyp'] = slu_hyp_copy
        test_results_select[uttr] ['gt'] = turn_slu_copy

        #m_tps,m_fps,m_fns = model_evaluator.add_turn(turn_slu, slu_hyp)
        m_tps,m_fps,recall_value = model_evaluator.add_turn(turn_slu, slu_hyp_select)
        #test_results[uttr]['hyp'] = {k: slot_map[v] for k, v in result['hyp'].items()}
        recall_value_all.append(recall_value)
    print('recall_value: ', np.mean(np.array(recall_value_all)))
    
    with open (os.path.join(results_path, 'samp_results_map_before_select.json'), 'w') as f:
        f.write(json.dumps(test_results_before_select, indent=4))   

    with open (os.path.join(results_path, 'samp_results_map_select.json'), 'w') as f:
        f.write(json.dumps(test_results_select, indent=4))   

    with open(os.path.join(results_path, 'result_metrics.json'), 'wt') as of:
        weighted_result = model_evaluator.eval(of)
    return weighted_result


class SlotEvaluator:
    def __init__(self, name='dummy'):
        self.tp = 0.000001
        self.fp = 0.000001
        self.tn = 0
        self.fn = 0.000001

    @property
    def precision(self):
        return round(self.tp) / (self.tp + self.fp)

    @property
    def recall(self):
        return round(self.tp) / (self.tp + self.fn)

    @property
    def f1(self):
        return 2 * self.precision * self.recall / (self.precision + self.recall + .000000000000001)

    @property
    def instances(self):
        return self.tp + self.fn


class GenericEvaluator:

    def __init__(self, name, eval_mapping, slot_names):
        self.name = name
        self.eval_mapping = eval_mapping
        self.slot_evaluators = {x: SlotEvaluator() for x in slot_names}

    def add_turn(self, turn_slu, slu_hyp):
        tps = 0
        fps = 0
        tns = 0
        fns = 0
        

        gold_values = [v for slot, value in turn_slu.items() for v in value]
        hyp_values = [v for slot, value in slu_hyp.items() for v in value]

        recall_value = len(set(gold_values) & set(hyp_values))/len(set(gold_values))

        for gold_slot, gold_value in list(turn_slu.items()):
                
            if gold_slot not in self.slot_evaluators:
                continue
            
            if gold_slot not in slu_hyp:
                self.slot_evaluators[gold_slot].fn += 1
                fns += 1
                continue
            #if slu_hyp[gold_slot].lower() in gold_value.lower() or gold_value.lower() == 'slot':
            if len(set(slu_hyp[gold_slot]) & set(gold_value))>0: #or gold_value.lower() == 'slot':
                self.slot_evaluators[gold_slot].tp += 1
                tps += 1
                del slu_hyp[gold_slot]
                continue
            else:
                self.slot_evaluators[gold_slot].fp += 1
                fps += 1
                del slu_hyp[gold_slot]
        for predicted_slot, predicted_value in slu_hyp.items():
            if predicted_slot not in self.slot_evaluators:
                continue
            self.slot_evaluators[predicted_slot].fp += 1
            fps += 1
        #return tps, fps, fns
        return tps, fps, recall_value 

    def eval(self, result):
        print(self.name, file=result)
        mean_precision = mean_recall = mean_f1 = 0
        w_mean_precision = w_mean_recall = w_mean_f1 = 0
        for name, evaluator in self.slot_evaluators.items():
            print(name, evaluator.precision, evaluator.recall, evaluator.f1, file=result)
            mean_precision += evaluator.precision
            mean_recall += evaluator.recall
            mean_f1 += evaluator.f1
            total_instances = sum([evaltr.instances for evaltr in self.slot_evaluators.values()])
            w_mean_precision += evaluator.precision * (evaluator.instances) / total_instances
            w_mean_recall += evaluator.recall * (evaluator.instances) / total_instances
            w_mean_f1 += evaluator.f1 * (evaluator.instances) / total_instances
        
        print('mean', mean_precision / len(self.slot_evaluators), mean_recall / len(self.slot_evaluators), mean_f1 / len(self.slot_evaluators), file=result)
        print('weighted-mean', w_mean_precision, w_mean_recall, w_mean_f1, file=result)
        print('mean: ', [mean_precision / len(self.slot_evaluators), mean_recall / len(self.slot_evaluators), mean_f1 / len(self.slot_evaluators)])
        print('weighted-mean: ', [w_mean_precision, w_mean_recall, w_mean_f1])
                
        print('-' * 80, file=result)
        w_result = {'weighted_P': w_mean_precision,
                'weighted_R': w_mean_recall,
                'weighted_F1': w_mean_f1
                }
        return w_result


def compute_ap(frames, sorted_list):
    spoted = 0
    i = 0
    precision_sum = 0
    cut_off = 20
    for fr_name in sorted_list:
        i += 1
        spoted_any = False
        for fr in fr_name.split('-'):
            if fr in frames:
                spoted += 1
                print(fr, 'adding {}/{}'.format(spoted, i))
                precision_sum += spoted / i
                i += 1
        if spoted == len(frames):
            break
    # return precision_sum
    return precision_sum / len(frames)


from collections import Counter

def slot_mapping(slot_value_gt, slot_value_hyp):
    
    del slot_value_gt['noslot']
    sort_gt = sorted(slot_value_gt.items(), key=lambda x: len(x[1]), reverse=True)
    value2slot_hyp = {}
    #print('hyp...........')
    slot_in_hyp = []
    for slot, value in slot_value_hyp.items():
        slot_in_hyp.append(slot)
        #print('slot: ', slot)
        #print('value: ', Counter(value))
        for v in value:
            if v not in value2slot_hyp.keys():
                value2slot_hyp[v] = []
            value2slot_hyp[v].append(slot)

    #print('gt...........')
    slot_map = {}
    candidate_id = [str(i) for i in range(len(slot_value_gt))]
    #print( 'candidate_id: ', candidate_id  )
    for slot, value in sort_gt:
        #print('slot: ', slot)
        #print('value: ', Counter(value))
        #print('len_value: ', len(value))

        slot_id_in_hyp = []
        for v in value:
            try:
                slot_id_in_hyp.extend(value2slot_hyp[v])
            except:
                pass
        counter = Counter(slot_id_in_hyp)
        
        #print('len(slot_id_in_hyp): ', len(slot_id_in_hyp))
        #print('Counter: ', counter)
        #print ([k for k,v in counter.most_common()])
        for k, v in counter.most_common():
            if k in candidate_id:
                slot_map[k] = slot
                #print('slot_map: ', slot_map)
                candidate_id.remove(k)
                break
            else:
                if len(candidate_id)==1:
                    slot_map[candidate_id[0]] = slot
                #print('len(candidate_id)',len(candidate_id))
                #print('candidate_id',candidate_id)
                #if len(candidate_id)==2:
                #    slot_map[candidate_id[0]] = slot
                #    candidate_id.remove(candidate_id[0])
                #print('slot_map: ', slot_map)    
    for slot in slot_in_hyp:
        if not slot in slot_map.keys():
            slot_map[slot]="noslot"
            
    return slot_map


def value_mapping(slot_value_gt, slot_value_hyp):
    value2slot_gt = {}
    for slot, value in slot_value_gt.items():
        for v in value:
            value2slot_gt[v]=slot
    
    value_map = {}
    for slot, value in slot_value_hyp.items():
        for v in value:
            if v in value2slot_gt.keys():
                value_map[v] = value2slot_gt[v]
            else:
                value_map[v] = 'no'
    return value_map
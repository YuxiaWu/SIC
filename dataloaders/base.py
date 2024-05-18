import numpy as np
import os   
import random
import torch
import logging
import tensorflow as tf
from collections import Counter

from .__init__ import max_seq_lengths, backbone_loader_map, benchmark_labels

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)

class DataManager:
    
    def __init__(self, args, logger_name = 'Discovery'):

        self.logger = logging.getLogger(logger_name)

        
        args.max_seq_length = 256
        self.data_dir = os.path.join(args.data_dir, args.dataset)

        self.all_label_list_ori = self.get_labels(args.dataset)
        cc=Counter(self.all_label_list_ori)
        self.all_label_set = sorted(set(self.all_label_list_ori), key = self.all_label_list_ori.index)
        self.logger.info('origin all_label_set is  %s', len(self.all_label_set))
        self.all_label_list = [label for label in self.all_label_set if cc[label]>15]
        self.logger.info('all_label_list after select is   %s', len(self.all_label_list))

        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)
        set_seed(args.seed)
        self.known_label_list = list(np.random.choice(np.array(self.all_label_list), self.n_known_cls, replace=False))

        self.all_label_list.append('noslot')
        self.known_label_list.append('noslot')
        self.n_known_cls +=1
        self.all_label_list = self.known_label_list+list(set(self.all_label_list)-set(self.known_label_list))

        self.logger.info('The number of known slot is %s', self.n_known_cls)
        self.logger.info('The number of all slot is %s', len(self.all_label_list))
        self.logger.info('Lists of known labels are: %s', str(self.known_label_list))
        self.logger.info('Lists of all labels are: %s', str(self.all_label_list))

        args.num_labels = self.num_labels = int(len(self.all_label_list) * args.cluster_num_factor)
        self.logger.info('num_labels int(len(self.all_label_list) * args.cluster_num_factor): %s', args.num_labels)
        self.dataloader = self.get_loader(args, self.get_attrs())

       #len(set(self._get_labels_slot(mode='test')) - set(self.all_label_list))         
    def get_labels(self, dataset):
        labels = self._get_labels_slot(mode='train')

        return labels

    def _get_labels_slot(self, mode):
        labels = []
        with open(os.path.join(self.data_dir, 'slu', mode, 'seq.out'), "r") as f:
            for i, line_out in enumerate(f):
                line_out = line_out.strip()  
                l2_list = line_out.split()
                
                for l in l2_list:
                    if "B" in l:
                        slot_name = l.split("-")[1]
                        labels.append(slot_name)
        return labels

    def get_loader(self, args, attrs):
        
        dataloader = backbone_loader_map[args.backbone](args, attrs)

        return dataloader
    
    def get_attrs(self):

        attrs = {}
        for name, value in vars(self).items():
            attrs[name] = value

        return attrs




import torch
import torch.nn.functional as F
import numpy as np
import copy
import logging
import os

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, f1_score, accuracy_score
from tqdm import trange, tqdm
from scipy.optimize import linear_sum_assignment
from losses import loss_map
from utils.functions import save_model, restore_model, eval_and_save
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)

from utils.metrics import clustering_score
from .pretrain import PretrainModelManager
from pytorch_pretrained_bert.tokenization import BertTokenizer
import numpy as np
from .semi_kmeans import Semi_KMeans
import time
from sklearn import metrics
import json

class ModelManager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):

        self.logger = logging.getLogger(logger_name)
        self.model = model.model
        self.optimizer = model.optimizer
        self.device = model.device
        self.args = args
        self.Data = data
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            data.dataloader.train_loader, data.dataloader.eval_loader, data.dataloader.test_loader
        
        self.train_input_ids, self.train_input_mask, self.train_segment_ids, self.train_bin_label_ids, self.train_slot_label_ids = \
            data.dataloader.train_input_ids, data.dataloader.train_input_mask, data.dataloader.train_segment_ids, data.dataloader.train_bin_label_ids, data.dataloader.train_slot_label_ids

        self.all_unlabeled_loader_parser = data.dataloader.all_unlabeled_loader_parser
        self.all_unlabeled_loader_slu = data.dataloader.all_unlabeled_loader_slu

        self.train_labeled_Feature = data.dataloader.train_labeled_Feature 
        self.all_unlabeled_Feature = data.dataloader.all_unlabeled_Feature_parser

        self.train_labeled_loader = data.dataloader.train_labeled_loader

        self.loss_fct = loss_map[args.loss_fct]
        
        self.known_label_map = {}
        for i, label in enumerate(data.known_label_list):
            self.known_label_map[label] = i
        self.classifier_num_label = len(self.known_label_map)
        pretrain_manager = PretrainModelManager(args, data, model)  
        
        if args.train:

            if args.pre_train:
                self.logger.info('Pre-training start...')
                pretrain_manager.train(args, data)
                self.logger.info('Pre-training finished...')

                self.pretrained_model = pretrain_manager.model
            else:
                self.logger.info('restore_model pre-training model...')
                self.pretrained_model = restore_model(pretrain_manager.model, os.path.join(args.method_output_dir, 'pretrain'))
            self.num_labels = len(self.known_label_map)
            self.logger.info('num_labels when training: %s', self.num_labels)
            self.load_pretrained_model(self.pretrained_model)
            self.model = pretrain_manager.model


        else:

            self.pretrained_model = restore_model(pretrain_manager.model, os.path.join(args.method_output_dir, 'pretrain'))
            self.num_labels = len(self.known_label_map)

            self.model = restore_model(self.model, args.model_output_dir)

        self.logger.info('init_centroids...')
        self.centroids = self.get_init_centroids()
    
    
    def get_init_centroids(self):
        total_features, total_labels = self.get_features_labels(self.train_labeled_loader)
        centroids = []
        for label in range(len(self.known_label_map)):
            feats_label_i = total_features[total_labels==label]
            cent_i = torch.mean(feats_label_i,0)
            centroids.append(cent_i)
        return torch.vstack(centroids).to(self.device)

    def predict_k(self, args, data, feats):

        self.logger.info('Predict number of clusters start...')

        #feats = self.get_outputs(args, mode = 'train', model = self.pretrained_model, get_feats = True)
        #feats = feats.cpu().numpy()

        km = KMeans(n_clusters = data.num_labels).fit(feats)
        y_pred = km.labels_

        pred_label_list = np.unique(y_pred)
        drop_out = len(feats) / data.num_labels

        cnt = 0
        for label in pred_label_list:
            num = len(y_pred[y_pred == label]) 
            if num < drop_out:
                cnt += 1

        K = len(pred_label_list) - cnt
        
        self.logger.info('Predict number of clusters finish...')
        outputs = {'K': K, 'mean_cluster_size': drop_out}
        for key in outputs.keys():
            self.logger.info("  %s = %s", key, str(outputs[key]))

        return K



    def feature2loader(self, all_unlabeled_Feature):
        datatensor = TensorDataset(
            all_unlabeled_Feature[0], 
            all_unlabeled_Feature[1], 
            all_unlabeled_Feature[2],
            all_unlabeled_Feature[3],
            all_unlabeled_Feature[4],
            all_unlabeled_Feature[5])
        sampler = SequentialSampler(datatensor)
        dataloader = DataLoader(datatensor, sampler=sampler, batch_size = self.args.train_batch_size) 
        return dataloader

    def sample_select(self, iter_i):
        '''
        select certain samples from unlabeled dataset, based on the predict prob
        after selecting:

            add the samps into the labeled data, change the ori labels to be the predicted labels(label the unlabeled data)
            remove the ori_samps from the unlabeled data
        output:
            ori is used for remove unlabeled data
            psu_label is used for update the labeled data
        ''' 
        train_unlabeled_dataloader = self.feature2loader(self.all_unlabeled_Feature)
        total_features, _ = self.get_features_labels(train_unlabeled_dataloader)
        self.logger.info('n_iter sample_select : %s', iter_i)
        
        if iter_i>0:
            n_center = self.centroids.size(0)
            cos = torch.nn.CosineSimilarity(dim=2)
            total_logits = cos(total_features.unsqueeze(0).repeat(n_center,1,1), self.centroids.unsqueeze(1))
            total_logits = 0.5+0.5*total_logits.transpose(0,1)
        else:
            total_logits = torch.empty((0, self.centroids.size(0))).to(self.device)
            for batch in tqdm(train_unlabeled_dataloader, desc="OURS-OURS-Iteration"):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, bin_label_ids, slot_label_ids, label_ids = batch

                with torch.set_grad_enabled(False):
                    pooled_output, logits = self.model(input_ids, segment_ids, input_mask, bin_label_ids = bin_label_ids)
                    total_logits = torch.cat((total_logits, logits))
        
            total_logits = F.softmax(total_logits,dim=1)
        sorted_logits, indices = torch.sort(total_logits, dim=1, descending=True)
        max_logits = sorted_logits[:,0]
 
        if len(train_unlabeled_dataloader.dataset)<500:
            self.logger.info('len_unlabel<500, all in!!')
            selected_idx = torch.where(max_logits>0)[0]
            none_selected_idx = torch.where(max_logits<0)[0]
        else:

            selected_idx = torch.where(max_logits>self.args.thr)[0]
            none_selected_idx = torch.where(max_logits<self.args.thr)[0]

            self.logger.info('self.args.thr: {}'.format(self.args.thr))
            self.logger.info('min max logits: {}'.format(max_logits.min()))
            self.logger.info('max max logits: {}'.format(max_logits.max()))
            self.logger.info('len selected_idx: {}'.format(len(selected_idx)))
            del max_logits

        label_pre_selected = indices[:,0][selected_idx]
        selected_feature = []
        for i in range(6):
            if i==5:
                label_ids = torch.tensor(label_pre_selected, dtype=torch.long).cpu()
                selected_feature.append(label_ids)
            else:
                selected_feature.append(self.all_unlabeled_Feature[i][selected_idx])
        self.train_labeled_Feature = [torch.cat([x,y]) for x, y in zip(self.train_labeled_Feature, selected_feature)] 
        none_selected_feature = [self.all_unlabeled_Feature[i][none_selected_idx] for i in range(6)]
        self.all_unlabeled_Feature = none_selected_feature

        self.train_labeled_dataloader = self.feature2loader(self.train_labeled_Feature)
        self.all_unlabeled_dataloader = self.feature2loader(self.all_unlabeled_Feature)
        
        self.train_semi_dataloader = self.feature2loader([torch.cat([x,y]) for x, y in zip(self.train_labeled_Feature, self.all_unlabeled_Feature)] ) 
    
    def get_features_labels(self, dataloader):
        self.model.eval()
        total_features = torch.empty((0,self.args.feat_dim)).to(self.device)
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        #total_logits = torch.empty((0, self.classifier_num_label),dtype=torch.long).to(self.device)
        for batch in tqdm(dataloader, desc="OURS-Extracting representation"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, bin_label_ids, slot_label_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                pooled_output, logits = self.model(input_ids, segment_ids, input_mask, bin_label_ids = bin_label_ids)
                
                total_labels = torch.cat((total_labels,label_ids))
                total_features = torch.cat((total_features, pooled_output))
                #total_logits = torch.cat((total_logits, logits))
            
        return total_features, total_labels ##, total_logits
    
    def update_pseudo_labels(self, pseudo_labels, Feature):
        new_labels = pseudo_labels[:Feature[0].size(0)]
        #assert new_labels.size()
        train_data = TensorDataset(
            Feature[0], 
            Feature[1], 
            Feature[2],
            Feature[3],
            Feature[4],
            Feature[5],
            new_labels)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = self.args.train_batch_size)
        return train_dataloader


    def train(self, args, data):
        n_iter = 100
        best_score = 0
        best_model = None
        wait = 0
        for iter_i in range(n_iter):
            self.logger.info('iter: {}'.format(iter_i))
            if iter_i>0:
                self.args.thr -= 0.05
    # 1. select data
            self.logger.info('--------------1. select data')

            self.known_id2label = dict(zip(self.known_label_map.values(), self.known_label_map.keys()))
            self.logger.info('known_id2label.size: {}'.format(len(self.known_id2label)))
            
    # 2. update data, label the unlabeled data, using self.known_label_map
            self.logger.info('--------------2. update data, label the unlabeled data,')
            self.sample_select(iter_i)
            self.num_labeled_examples = len(self.train_labeled_dataloader.dataset)
            num_unlabeled_examples = len(self.all_unlabeled_dataloader.dataset) 
            self.logger.info('num_labeled_examples: {}'.format(self.num_labeled_examples))
            self.logger.info('num_unlabeled_examples: {}'.format(num_unlabeled_examples))
            
    # 3. train model with only labeled data
            self.logger.info('--------------3. train model with only labeled data')
            self.logger.info('training with more epochs.......')

            best_loss = 10000
            best_model_cls = None
            self.logger.info('self.centroids.size: {}'.format(self.centroids.size()))
            self.model.train()
            for epoch_cls in range(10):
                self.logger.info('-----epoch_cls: {}'.format(epoch_cls))
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                train_time = time.time()
                for batch in tqdm(self.train_labeled_dataloader, desc="OURS-OURS-Pseudo-Training"):
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, input_mask, segment_ids, bin_label_ids, slot_label_ids, instance_label_ids = batch
                    feats, _ = self.model(input_ids, segment_ids, input_mask, bin_label_ids = bin_label_ids)

                    n_center = self.centroids.size(0)
                    cos = torch.nn.CosineSimilarity(dim=2)
                    total_logits = cos(feats.unsqueeze(0).repeat(n_center,1,1), self.centroids.unsqueeze(1))
                    
                    total_logits = 0.5+0.5*total_logits.transpose(0,1)
                    loss_slotname = self.loss_fct(total_logits, instance_label_ids)
                    loss = loss_slotname

                    loss.backward()

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                tr_loss = tr_loss / nb_tr_steps
                self.logger.info('train_loss_epoch: {}'.format(tr_loss))
                self.logger.info('training time one epoch: {}'.format(time.time() - train_time))

                if tr_loss < best_loss:
                    best_model_cls = copy.deepcopy(self.model)
                    wait = 0
                    best_loss = tr_loss
                    self.logger.info('aha, find better model')
                else:
                    wait += 1
                    self.logger.info('wait: {}'.format(wait))

                    if wait >= 3: #self.args.wait_patient:
                        self.model = best_model_cls
                        break 
            self.logger.info('finish training...')

    # 4. semi-kmeans on all data
            self.logger.info('--------------4. semi-kmeans on all data')
            if num_unlabeled_examples==0 or num_unlabeled_examples<2:
                self.logger.info('--------------------break iter')
                break
            # set the cluster number, add new label
            if self.args.predict_k:
                feats, labels = self.get_features_labels(self.train_semi_dataloader)
                feats = feats.cpu().numpy()
                K_new = self.predict_k(args, data, feats)
                self.logger.info('K_old: %s ', str(K_old))
                self.logger.info('K_new: %s ', str(K_new))
                if K_new>K_old:
                    add_num = int(K_new-K_old)
                    self.logger.info('add_num: %s ', str(add_num))
                    K_old = copy.deepcopy(K_new)
                else:
                    add_num=0
            else:
                add_num = 1
                
            new_label_name = ['_'.join(['new',str(iter_i),str(i)]) for i in list(range(add_num))]
            new_label_id = [int(len(self.known_label_map)+i) for i in list(range(add_num))]
            new_map = dict(zip(new_label_name, new_label_id))
            self.known_label_map.update(new_map)
            self.classifier_num_label = len(self.known_label_map)
            self.logger.info('for kmeans: new known_label size: {}'.format(self.classifier_num_label))
            
            feats, labels = self.get_features_labels(self.train_semi_dataloader)
            feats = feats.cpu().numpy()
            labels = labels.squeeze().cpu().numpy()
            y_labeled = labels[:self.num_labeled_examples]

            self.logger.info('SemiKMeans innner')
            feats_labeled = feats[:self.num_labeled_examples,:]
            feats_unlabeled = feats[self.num_labeled_examples:,:]
            cluster_time = time.time()
            km = Semi_KMeans(n_clusters = self.classifier_num_label).fit_semi_kmeans(feats_labeled, feats_unlabeled, y_labeled)
            self.logger.info('cluster time: {}'.format(time.time() - cluster_time))
            all_labels =  km.labels_
            label_unique_num = len(np.unique(all_labels))
            self.logger.info('km label set num: {}'.format(label_unique_num))
            self.centroids = torch.Tensor(km.cluster_centers_).to(self.device) # new centroids
            pseudo_labels = torch.tensor(all_labels, dtype=torch.long).to(self.device)
            
            self.logger.info('--------------5. change the label of labeled_data after semi-kmeans')
            self.train_labeled_data = self.update_pseudo_labels(pseudo_labels, self.train_labeled_Feature) 

        self.logger.info('finish iter...')
        if args.save_model:
            save_model(self.model, args.model_output_dir)


    def load_pretrained_model(self, pretrained_model):

        pretrained_dict = pretrained_model.state_dict()
        classifier_params = ['classifier.weight','classifier.bias']
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)

    def evaluator(self, args, data, data_type = "test"):
        """
        type:
        unlabel: all the unlabeled data
        """
        if data_type=="test":
            dataloader =   data.dataloader.test_loader
            dataloader_gt = data.dataloader.test_loader_slu
        else:
            dataloader =   data.dataloader.all_unlabeled_loader_parser
            dataloader_gt = data.dataloader.all_unlabeled_loader_slu            

        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
                
        self.results = {}
        self.slot_value_hyp = {}
        self.slot_value_gt = {}
        self.logger.info('len(self.known_label_map): %s',self.classifier_num_label)
        self.get_results_samp(args, dataloader, mode = 'hyp', data_type='unlabel' )
        self.get_results_samp(args, dataloader_gt, mode = 'gt', data_type='unlabel')
        
        self.get_results_samp(args, self.train_labeled_loader, mode='gt', data_type='label',write_result=False)
        self.logger.info('Results saved in %s', str(args.result_dir))
        weighted_result = eval_and_save(args, self.results, self.slot_value_hyp, self.slot_value_gt, data_type) #, slot_map = id2slot)
        return weighted_result

    def get_results_samp(self, args, dataloader, mode='gt', data_type='unlabel', write_result=True):

        if mode=='hyp':
            feats,_ = self.get_features_labels(dataloader)
            self.logger.info('n_clusters = self.classifier_num_label: %s',self.classifier_num_label)
            
            km = KMeans(n_clusters = self.classifier_num_label).fit(feats.cpu().numpy())
            y_pred = km.labels_
            self.results['y_pred'] = y_pred
            self.results['feats'] = feats.cpu().numpy()
            
            centroids = torch.Tensor(km.cluster_centers_).to(self.device) 
            total_logits = torch.empty((0, centroids.size(0))).cuda()

            for i in feats:
                total_logits = torch.cat((total_logits, F.cosine_similarity(i.unsqueeze(0).repeat(centroids.size(0),1), centroids).unsqueeze(0)))
            total_logits = 0.5+0.5*total_logits
            total_maxprobs, total_preds = total_logits.max(dim = 1)
            
            total_maxprobs = total_maxprobs.cpu().numpy()
            
        samp_n = 0
        for batch in tqdm(dataloader, desc="OURS-OURS-Iteration of prediction"):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, bin_label_ids, slot_label_ids, label_ids = batch
            
            input_ids = input_ids.cpu().numpy()
            bin_label_ids = bin_label_ids.cpu().numpy()
            #slot_label_ids = slot_label_ids.cpu().numpy()
            label_ids = label_ids.cpu().numpy()
            
            for i in range(input_ids.shape[0]):
                
                uttr_list = [self.tokenizer.ids_to_tokens[k] for k in input_ids[i]]
                
                SEP_ind = uttr_list.index('[SEP]')
                uttr = ' '.join(uttr_list[1:SEP_ind])
                
                bin_label = np.array(bin_label_ids[i])
                indices = np.nonzero(bin_label)[0]
                value = ' '.join([uttr_list[int(ind)] for ind in indices])

                if mode=='hyp':
                    slot = y_pred[samp_n]
                    slot = str(slot)
                    prob = total_maxprobs[samp_n]
                else:
                    slot = label_ids[i]
                    if data_type=='unlabel':
                        id2slot = {i:slot_name for i, slot_name in enumerate(self.Data.all_label_list)}
                    else:
                        id2slot = {i:slot_name for i, slot_name in enumerate(self.Data.known_label_list)}
                    slot = id2slot[slot]
 
                if write_result:
                    if not uttr in self.results.keys():
                        self.results[uttr] = {'hyp': {}, 'gt': {}}
                    if mode=='hyp':
                        self.results[uttr][mode][value] = [slot, str(prob), str(indices)]
                    else:
                        self.results[uttr][mode][value] = slot
                
                if mode=='hyp':
                    if not slot in self.slot_value_hyp.keys():
                        self.slot_value_hyp[slot] = [] #set()
                    self.slot_value_hyp[slot].append(value)
                else:
                    if not slot in self.slot_value_gt.keys():
                        self.slot_value_gt[slot] = [] #set()
                    self.slot_value_gt[slot].append(value)  
                samp_n +=1     
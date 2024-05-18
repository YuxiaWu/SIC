from lib2to3.pgen2 import token
from operator import mod
import torch
import torch.nn.functional as F
from torch import nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from torch.nn.parameter import Parameter
from .utils import PairEnum
import copy

from pytorch_pretrained_bert.tokenization import BertTokenizer
activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}

def get_slot_emb(args, bert_model, input_ids, input_mask, segment_ids, bin_label_ids, lstm_for_value=None):
    lengths = torch.sum(input_mask, 1) # length of every example (bsz,)
    
    binary_labels = bin_label_ids 
    feats_concat = []
    bsz = input_ids.size()[0]
    for i in range(bsz):
        valid_length = lengths[i]
        bin_label = binary_labels[i]
        indices = torch.nonzero(bin_label) 
        
        # the input for value only
        input_ids_value = input_ids[i][indices].transpose(0,1)
        segment_ids_value = segment_ids[i][indices].transpose(0,1)
        input_mask_value = input_mask[i][indices].transpose(0,1)
        hidden_values, _ = bert_model(
            input_ids_value, 
            token_type_ids=segment_ids_value, 
            attention_mask=input_mask_value,
            output_all_encoded_layers=False)

        if args.value_enc_type == "lstm":
            value_feats, (_, _) = lstm_for_value(hidden_values)   # (1, subseq_len, hidden_dim)
            value_feats = torch.mean(value_feats, dim=1)  # (1, hidden_dim)
        else:
            value_feats = torch.mean(hidden_values, dim=1) # (1, hidden_dim)

        # replace the indice of value with MASK
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True) 
        
        input_ids_copy = copy.deepcopy(input_ids)
        input_id = input_ids_copy[i].unsqueeze(0) #[1,256]
        
        mask_id = tokenizer.convert_tokens_to_ids(['mask'])

        for ind in indices:
            input_id[0,ind] = torch.LongTensor(mask_id).cuda()
        #print(input_id.size())
        feats_token, _ = bert_model(
            input_id, 
            token_type_ids=segment_ids[i].unsqueeze(0), 
            attention_mask=input_mask[i].unsqueeze(0),
            output_all_encoded_layers=False)
       
        hidden_values_mask = feats_token[0][indices].transpose(0,1) #[1,1,768] #[4,1,768]
        if args.value_enc_type == "lstm":
            context_feats, (_, _) = lstm_for_value(hidden_values_mask)   # (1, subseq_len, hidden_dim)
            context_feats = torch.mean(context_feats, dim=1)  # (1, hidden_dim)
        else:
            context_feats = torch.mean(hidden_values_mask, dim=1) # (1, hidden_dim)

        if args.context:
            value_context = torch.cat((value_feats, context_feats),1)
        else:
            value_context = value_feats
        feats_concat.append(value_context)

    feats_concat_o = torch.vstack(feats_concat)
    return feats_concat_o   

class BERT(BertPreTrainedModel):
    
    def __init__(self,config, args):

        super(BERT, self).__init__(config)
        self.args = args
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.lstm_for_value = nn.LSTM(
            input_size = config.hidden_size, 
            hidden_size = config.hidden_size//2, 
            num_layers=2, 
            bidirectional=True, 
            batch_first=True)
        if args.context:
            self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)      
        self.apply(self.init_bert_weights)
        
    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels = None,
                feature_ext = False, mode = None, loss_fct = None, bin_label_ids = None):

        pooled_output = get_slot_emb(self.args, self.bert,
            input_ids = input_ids, 
            input_mask = attention_mask, 
            segment_ids = token_type_ids, 
            bin_label_ids = bin_label_ids,
            lstm_for_value = self.lstm_for_value)
        
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if feature_ext:
            return pooled_output
        else:
            if mode == 'train':
                
                loss = loss_fct(logits, labels)
                return loss
            else:
                return pooled_output, logits
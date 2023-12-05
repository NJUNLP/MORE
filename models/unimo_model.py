import sys
sys.path.append("..")
import os
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertConfig, CLIPConfig, BertModel
from .modeling_unimo import UnimoModel
from .modeling_clip import CLIPModel
import numpy as np
# from utils import *


class UnimoREModel(nn.Module):
    def __init__(self, num_labels, tokenizer, args):
        super(UnimoREModel, self).__init__()
        self.args = args
        # print(vision_config)
        # print(text_config)
        clip_model = CLIPModel.from_pretrained(args.vit_name, ignore_mismatched_sizes=True)
        clip_vit = clip_model.vision_model
        vision_config = CLIPConfig.from_pretrained(args.vit_name).vision_config
        text_config = BertConfig.from_pretrained(args.bert_name)
        bert = BertModel.from_pretrained(args.bert_name)
        clip_model_dict = clip_vit.state_dict()
        bert_model_dict = bert.state_dict()

        self.vision_config = vision_config
        self.text_config = text_config

        # for re
        vision_config.device = args.device
        self.model = UnimoModel(vision_config, text_config)

        vision_names, text_names = [], []
        model_dict = self.model.state_dict()

        avg_conv_weight = torch.mean(clip_model_dict['embeddings.patch_embedding.weight'], dim=1).unsqueeze(1)
        clip_model_dict['embeddings.depth_embedding.weight'] = torch.tensor(avg_conv_weight.data.clone())
        for name in model_dict:
            if 'vision' in name:
                clip_name = name.replace('vision_', '').replace('model.', '')
                if clip_name in clip_model_dict:
                    vision_names.append(clip_name)
                    model_dict[name] = clip_model_dict[clip_name]
            elif 'text' in name:
                text_name = name.replace('text_', '').replace('model.', '')
                if text_name in bert_model_dict:
                    text_names.append(text_name)
                    model_dict[name] = bert_model_dict[text_name]
        assert len(vision_names) == len(clip_model_dict) and len(text_names) == len(bert_model_dict), \
            (len(vision_names), len(text_names), len(clip_model_dict), len(bert_model_dict))
        self.model.load_state_dict(model_dict)

        self.model.resize_token_embeddings(len(tokenizer))
        self.args = args

        # self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.text_config.hidden_size * 2, num_labels)
        self.mm_linear = nn.Linear(self.text_config.hidden_size * 2, self.text_config.hidden_size)

        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        self.tokenizer = tokenizer

        self.output_attentions = True
        self.output_hidden_states = True

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            org_image=None,
            ent_imgs=None,
            ent_idx=None,
            position=None,
    ):
        output, vision_output = self.model(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           token_type_ids=token_type_ids,
                                           aux_values=ent_imgs,
                                           position=position,
                                           output_attentions=True,
                                           output_hidden_states=True,
                                           return_dict=True, )
        bsz, seq_len, hidden_size = output.shape
        entity_hidden_state = torch.Tensor(bsz, hidden_size * 2)  # batch, 2*hidden
        for i in range(bsz):
            head_idx = input_ids[i].eq(self.head_start).nonzero().item()
            head_hidden = output[i, head_idx, :].squeeze()
            vision_hidden = vision_output[i, ent_idx[i], :]
            if self.args.use_cap:
                tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
                tail_hidden = output[i, tail_idx, :].squeeze()
                mm_tail_hidden = self.mm_linear(torch.cat([vision_hidden, tail_hidden], dim=-1))
                entity_hidden_state[i] = torch.cat([head_hidden, mm_tail_hidden], dim=-1)
            else:
                entity_hidden_state[i] = torch.cat([head_hidden, vision_hidden], dim=-1)
        entity_hidden_state = entity_hidden_state.to(self.args.device)
        logits = self.classifier(entity_hidden_state)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            return loss_fn(logits, labels.view(-1)), logits
        return logits

from typing import Any, Optional, Tuple
import math

import torch
from torch import nn, Tensor, device

from .modeling_bert import *
from .modeling_clip import *


# some function
def get_extended_attention_mask(attention_mask: Tensor, input_shape: Tuple[int], device: device) -> Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (:obj:`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (:obj:`Tuple[int]`):
            The shape of the input to the model.
        device: (:obj:`torch.device`):
            The device of the input to the model.

    Returns:
        :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=torch.long)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


def get_head_mask(
        head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
) -> Tensor:
    """
    Prepare the head mask if needed.

    Args:
        head_mask (:obj:`torch.Tensor` with shape :obj:`[num_heads]` or :obj:`[num_hidden_layers x num_heads]`, `optional`):
            The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
        num_hidden_layers (:obj:`int`):
            The number of hidden layers in the model.
        is_attention_chunked: (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the attentions scores are computed by chunks or not.

    Returns:
        :obj:`torch.Tensor` with shape :obj:`[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or
        list with :obj:`[None]` for each layer.
    """
    head_mask = [None] * num_hidden_layers

    return head_mask


class UnimoEncoder(nn.Module):
    def __init__(self, vision_config, text_config):
        super().__init__()
        self.vision_config = vision_config
        self.text_config = text_config

        self.vision_layers = nn.ModuleList([CLIPEncoderLayer(vision_config) for _ in range(vision_config.num_hidden_layers)])
        self.text_layer = nn.ModuleList([BertLayer(text_config) for _ in range(text_config.num_hidden_layers)])
        self.image_position_embeddings = nn.Linear(5, self.text_config.hidden_size)

        self.layernorm = nn.LayerNorm(vision_config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)


    def forward(
            self,
            vision_embeds=None,
            text_embeds=None,
            attention_mask=None,
            head_mask=None,
            position=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        assert self.vision_config.num_hidden_layers == self.text_config.num_hidden_layers

        all_vision_hidden_states = () if output_hidden_states else None
        all_text_hidden_states = () if output_hidden_states else None
        all_vision_attentions = () if output_attentions else None
        all_text_attentions = () if output_attentions else None

        vision_hidden_states = vision_embeds
        text_hidden_states = text_embeds
        for idx in range(self.vision_config.num_hidden_layers):
            if output_hidden_states:
                all_vision_hidden_states = all_vision_hidden_states + (vision_hidden_states,)
                all_text_hidden_states = all_text_hidden_states + (text_hidden_states,)

            # vision
            # TODO: 9-12 layers past text as pkv to vision
            past_key_values = text_layer_output[-1] if idx >= 8 else None
            vision_layer_module = self.vision_layers[idx]
            vision_layer_output = vision_layer_module(
                vision_hidden_states,
                output_attentions=output_attentions,
                past_key_values=past_key_values,
                current_layer=idx,
            )
            vision_hidden_states = vision_layer_output[0]

            if idx == 7:
                obj_num = 10
                patch_num = int(vision_hidden_states.shape[1] / obj_num)
                vision_embedding_output = []
                for i in range(obj_num):
                    start = i * patch_num
                    mid = start + (patch_num >> 1)
                    end = start + patch_num
                    aux_embeddings = vision_hidden_states[:, start:end, :, ]
                    embeddings = aux_embeddings.transpose(1, 2)  # batch_size, 768, patch_num
                    embeddings = torch.avg_pool1d(embeddings, kernel_size=embeddings.shape[-1]).squeeze(2)
                    vision_embedding_output.append(embeddings)  # batch_size, 768
                vision_hidden_states = torch.stack(vision_embedding_output, dim=1)  # batch_size, img_num, 768
                # Position-Fusion
                if position is not None:
                    position_embedding = self.image_position_embeddings(position)
                    vision_hidden_states = self.layernorm(vision_hidden_states + position_embedding)
                    vision_hidden_states = self.dropout(vision_hidden_states)

            # text
            # TODO: 9-12 layers past vison qks to text
            last_hidden_state = vision_hidden_states if idx >= 8 else None
            output_qks = True if idx >= 7 else None
            layer_head_mask = head_mask[idx] if head_mask is not None else None
            text_layer_module = self.text_layer[idx]
            text_layer_output = text_layer_module(
                text_hidden_states,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                visual_hidden_state=last_hidden_state,
                output_attentions=output_attentions,
                output_qks=output_qks,
                current_layer=idx,
            )
            text_hidden_states = text_layer_output[0]
            if output_attentions:
                all_vision_attentions = all_vision_attentions + (vision_layer_output[1],)
                all_text_attentions = all_text_attentions + (text_layer_output[1],)

            if output_hidden_states:
                all_vision_hidden_states = all_vision_hidden_states + (vision_hidden_states,)
                all_text_hidden_states = all_text_hidden_states + (text_hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [
                    text_hidden_states,
                    all_text_hidden_states,
                    all_text_attentions,
                ] if v is not None)

        return BaseModelOutput(
            last_hidden_state=text_hidden_states,
            hidden_states=all_text_hidden_states,
            attentions=all_text_attentions
        ), BaseModelOutput(
            last_hidden_state=vision_hidden_states,
            hidden_states=all_vision_hidden_states,
            attentions=all_vision_attentions
        )


class UnimoModel(nn.Module):
    def __init__(self, vision_config, text_config, add_pooling_layer=True):
        super(UnimoModel, self).__init__()
        # vision model
        self.vision_config = vision_config
        self.vision_embeddings = CLIPVisionEmbeddings(vision_config)
        self.vision_pre_layrnorm = nn.LayerNorm(vision_config.hidden_size)
        self.vision_post_layernorm = nn.LayerNorm(vision_config.hidden_size)
        # self.location_layernorm = nn.LayerNorm(vision_config.hidden_size)

        # text model
        self.text_config = text_config
        self.text_embeddings = BertEmbeddings(text_config)
        self.text_pooler = BertPooler(text_config) if add_pooling_layer else None

        # all
        self.encoder = UnimoEncoder(vision_config, text_config)

        self.device = vision_config.device

        # self.mm_linear = nn.Linear(self.text_config.hidden_size * 2, self.text_config.hidden_size)

        # self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(64 * 28 * 28, self.text_config.hidden_size)
        # self.fc2 = nn.Linear(512, 10)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            pixel_values=None,
            aux_values=None,
            position=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        # pre vision
        vision_embedding_output = []
        obj_num = aux_values.shape[1]
        bsz = aux_values.shape[0]
        for i in range(obj_num):
            aux_value = aux_values[:, i, :, :, :].squeeze(1)
            if aux_values.shape[-3] == 4:
                patch_embeddings, depth_embeddings = self.vision_embeddings(aux_value)
                # print(patch_embeddings.shape, depth_embeddings.shape)
                patch_embedding = patch_embeddings.squeeze(1)
                depth_embedding = depth_embeddings.squeeze(1)
                vision_embedding_output.append(patch_embedding)  # batch_size, 768
                vision_embedding_output.append(depth_embedding)  # batch_size, 768
            else:
                patch_embeddings = self.vision_embeddings(aux_value)  # batch_size, 1 + patch_num, 768
                # patch_embedding = patch_embeddings.squeeze(1)
                vision_embedding_output.append(patch_embeddings)  # batch_size, 768
        vision_embedding_output = torch.stack(vision_embedding_output, dim=1)  # batch_size, img_num, 768
        vision_embedding_output = vision_embedding_output.reshape(bsz, -1, self.vision_config.hidden_size)
        vision_embedding_output = self.vision_pre_layrnorm(vision_embedding_output)

        # pre text
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        if token_type_ids is None:
            raise ValueError("token_type_ids is None!")

        extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, input_shape, device)
        head_mask = get_head_mask(head_mask, self.text_config.num_hidden_layers)  # [None]*12

        text_embedding_output = self.text_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        # all encoder
        text_encoder_outputs, vision_encoder_outputs = self.encoder(
            vision_embeds=vision_embedding_output,
            text_embeds=text_embedding_output,
            attention_mask=extended_attention_mask,
            position=position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        t_last_hidden_state = text_encoder_outputs[0]
        v_last_hidden_state = vision_encoder_outputs[0]

        if not return_dict:
            return (t_last_hidden_state, v_last_hidden_state)

        return t_last_hidden_state, v_last_hidden_state

    def _init_text_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.text_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.text_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        return self.text_embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.text_embeddings.word_embeddings = value

    def resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

    def _get_resized_embeddings(
            self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int] = None
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (:obj:`torch.nn.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (:obj:`int`, `optional`):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or :obj:`None`, just returns a pointer to the input tokens
                :obj:`torch.nn.Embedding`` module of the model without doing anything.

        Return:
            :obj:`torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
            :obj:`new_num_tokens` is :obj:`None`
        """
        if new_num_tokens is None:
            return old_embeddings
        else:
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens:
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}."
                f"You should either use a different resize function or make sure that `old_embeddings` are an instance of {nn.Embedding}."
            )

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim).to(
            self.device, dtype=old_embeddings.weight.dtype
        )

        # initialize all new embeddings (in particular added tokens)
        self._init_text_weights(new_embeddings)

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        return new_embeddings

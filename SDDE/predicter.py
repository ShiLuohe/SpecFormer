import tqdm

from typing import Optional, Tuple, Callable, List

import torch
import torch.nn as nn
from transformers.models.llama import (LlamaConfig, LlamaForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import Cache, StaticCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.utils.import_utils import is_flash_attn_greater_or_equal_2_10

from SDDE.cache import SpeculativeDynamicCache

def check_outpus(outputs: CausalLMOutputWithPast):
    if outputs.hidden_states is None:
        raise ValueError("Please set 'output_hidden_states=True' as a kwarg for the base model forward.")

def LHS_EMB_Hook(outputs: CausalLMOutputWithPast):
    check_outpus(outputs)
    return outputs.hidden_states[-1], outputs.hidden_states[0]

def PHS_EMB_Hook(outputs: CausalLMOutputWithPast):
    check_outpus(outputs)
    return outputs.hidden_states[-2], outputs.hidden_states[0]

def VAR_Hook(outputs: CausalLMOutputWithPast, layer_list: list[int]):
    check_outpus(outputs)
    return [outputs.hidden_states[idx] for idx in layer_list]

from flash_attn.layers.rotary import apply_rotary_emb as FlashAttn_apply_rotary_emb
def Wrapped_FlashAttn_apply_rotary_pos_emb(q, k, cos, sin, half_head_dim) \
    -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        FlashAttn_apply_rotary_emb(q, cos[0][..., :half_head_dim], sin[0][..., :half_head_dim]), 
        FlashAttn_apply_rotary_emb(k, cos[0][..., :half_head_dim], sin[0][..., :half_head_dim])
    )

from flash_attn.ops.rms_norm import RMSNorm

class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""
    """Copied and modified from the Transformers Library."""

    def __init__(self, 
                 config: LlamaConfig, 
                 device_dtype_dict: dict,
                 is_causal: bool,
                 layer_idx: int,
                 ):
        super().__init__()
        self.config = config
        self.layer_idx = config.num_hidden_layers + layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = is_causal

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, 
                                bias=config.attention_bias, **device_dtype_dict)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, 
                                bias=config.attention_bias, **device_dtype_dict)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, 
                                bias=config.attention_bias, **device_dtype_dict)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, 
                                bias=config.attention_bias, **device_dtype_dict)

        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[SpeculativeDynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states: torch.Tensor = self.q_proj(hidden_states)
        key_states  : torch.Tensor = self.k_proj(hidden_states)
        value_states: torch.Tensor = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        if self.is_causal:
            cos, sin = position_embeddings
            query_states, key_states = Wrapped_FlashAttn_apply_rotary_pos_emb(
                query_states, key_states, cos, sin, self.head_dim // 2
            )

            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(
                    key_states, 
                    value_states, 
                    self.layer_idx,
                    cache_kwargs, 
                )

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        
        attn_output: torch.Tensor = _flash_attention_forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            query_length=q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

class SwiGLU(nn.Module):
    def __init__(self, 
                 hidden_size: int, 
                 intermediate_size: int, 
                 device_dtype_dict: dict,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.act_func  = nn.SiLU()
        self.up_proj   = nn.Linear(self.hidden_size, self.intermediate_size, False, **device_dtype_dict)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, False, **device_dtype_dict)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, False, **device_dtype_dict)
    
    def forward(self, x: torch.Tensor):
        return self.down_proj(self.gate_proj(x) * self.act_func(self.up_proj(x)))

class CloneWithBias(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8, device_dtype_dict: dict={}):
        super().__init__()
        self.max_len = max_len
        self.bias = nn.Parameter(torch.zeros(max_len, d_model, **device_dtype_dict))

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(-2).repeat(1, 1, self.max_len, 1) + self.bias

class MultiTokenPredictor(nn.Module):

    config: LlamaConfig
    max_ahead: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(
        self, 
        outputs: CausalLMOutputWithPast, 
        attention_mask: torch.Tensor,
        past_key_values: Optional[Cache]=None,
        return_lhs: bool=False
    ) -> tuple[torch.Tensor, Cache, torch.Tensor]:
        '''Core forward function. '''
        raise NotImplementedError()

class DE_MultiTokenPredictor(MultiTokenPredictor):
    def __init__(self,
                 base_model: LlamaForCausalLM,
                 max_ahead: int,
                 hook_list: list[int],
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.base_model = base_model
        self.config: LlamaConfig
        self.config = base_model.config
        self.hidden_size = self.config.hidden_size
        self.config.attention_bias = False
        self.intermediate_size = self.config.intermediate_size

        self.device_dtype_kwargs = {"device": base_model.device, "dtype": base_model.dtype}
        # self.rms_norm_kwargs = {"normalized_shape": self.config.hidden_size, 
        #                         "eps": self.config.rms_norm_eps}
        self.rms_norm_kwargs = {"hidden_size": self.config.hidden_size, 
                                "eps": self.config.rms_norm_eps}

        self.max_ahead = max_ahead
        self.hook_list = hook_list

        self.hook_func = VAR_Hook
        self.norms = nn.ModuleList([RMSNorm(
            **self.rms_norm_kwargs, 
            **self.device_dtype_kwargs
        ) for _ in self.hook_list])
        self.down_samp = nn.Linear(
            len(self.hook_list) * self.hidden_size, 
            self.hidden_size, 
            False, 
            **self.device_dtype_kwargs
        )

        self.rotary_emb = base_model.model.rotary_emb
        self.position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None

        self.decoder_input_norm = RMSNorm(**self.rms_norm_kwargs, **self.device_dtype_kwargs)
        self.decoder_att = Attention(self.config, self.device_dtype_kwargs, True, 0)
        self.positional_input_norm = RMSNorm(**self.rms_norm_kwargs, **self.device_dtype_kwargs)
        self.positional = nn.Linear(
            self.hidden_size, 
            self.max_ahead * self.hidden_size, 
            True, #Bias here is fffectively trainable static positional embedding.
            **self.device_dtype_kwargs
        )
        # self.positional = CloneWithBias(self.hidden_size, self.max_ahead, self.device_dtype_kwargs)

        self.encoder_input_norm = RMSNorm(**self.rms_norm_kwargs, **self.device_dtype_kwargs)
        self.encoder_att = Attention(self.config, self.device_dtype_kwargs, True, 5)
        self.encoder_ffn_input_norm = RMSNorm(**self.rms_norm_kwargs, **self.device_dtype_kwargs)
        self.encoder_ffn = SwiGLU(self.hidden_size, self.intermediate_size, self.device_dtype_kwargs)

        self.vocab_size = base_model.vocab_size
        # self.final_norm = base_model.model.norm
        # self.lm_head = base_model.lm_head
        # self.final_norm.requires_grad_(False)
        # self.lm_head.requires_grad_(False)
    
    def forward(
        self, 
        outputs: CausalLMOutputWithPast, 
        attention_mask: torch.Tensor,
        max_predict: Optional[int]=None,
        past_key_values: Optional[SpeculativeDynamicCache]=None,
        return_lhs: bool=False
    ):

        if max_predict is None: max_predict = self.max_ahead

        # Hook out the desired hidden states, do norm independently and down-sampling them into 
        # the shape of inter-layer hidden states.
        hs_list = self.hook_func(outputs, self.hook_list)
        lhs_r = hs_list[-1] if return_lhs else None
        norm_list = [norm(shs) for norm, shs in zip(self.norms, hs_list) ]
        hs: torch.Tensor = self.down_samp(torch.cat(norm_list, dim=-1))

        # Preparing for decoder attention.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + hs.shape[1], device=hs.device
        )
        position_ids = cache_position.unsqueeze(0)
        self.position_embeddings = self.rotary_emb(hs, position_ids)

        # Decoder attention and residual connection. We suspect here that the residual
        # connection around an attention block is basically mandatory.
        next_hs = self.decoder_input_norm(hs)
        next_hs, _, past_key_values = self.decoder_att(
            next_hs, 
            attention_mask,
            past_key_value=past_key_values,
            position_embeddings=self.position_embeddings
        )
        hs = next_hs + hs

        # Positional FFN (also the decoder FFN). No residual connection here.
        bs, length, _ = hs.shape
        hs = self.positional_input_norm(hs)
        mths: torch.Tensor = self.positional(hs)
        mths = mths.view(bs * length, self.max_ahead, self.hidden_size)

        # Encoder attention and residual connection.
        next_mths = self.encoder_input_norm(mths)
        next_mths, _, _ = self.encoder_att(
            next_mths, 
            position_embeddings=self.position_embeddings
        )
        mths = next_mths + mths

        # Drop the unwanted positions.
        if max_predict != self.max_ahead:
            mths = mths[:, :max_predict, :]

        # Encoder FFN (SwiGLU here) and residual connection.
        next_mths = self.encoder_ffn_input_norm(mths)
        next_mths = self.encoder_ffn(next_mths)
        mths = next_mths + mths

        mths = mths.view(bs, length, max_predict, self.hidden_size)
        return mths, past_key_values, lhs_r

    # def forward(self, *args, **kwargs):
    #     mths, past_key_values, lhs = self.core_fwd(*args, **kwargs)

    #     # mths = base_model.model.norm(mths)
    #     # logits = base_model.lm_head(mths)

    #     return mths, past_key_values, lhs

    def fwd_bwd( 
        self, 
        base_model: LlamaForCausalLM,
        outputs: CausalLMOutputWithPast, 
        attention_mask: torch.Tensor,
        do_distill: bool,
        compute_lhs_loss: bool,
        loss_func_main: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        loss_func_lhs : Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]=None,
        labels: Optional[torch.Tensor]=None,
        logits: Optional[torch.Tensor]=None,
        lhs_loss_lambd: Optional[float]=None,
        past_key_values: Optional[Cache]=None
    ):
        ''' 
        Abandoned.
        The gradient accumulation trick from 
        "Better & Faster Large Language Models via Multi-token Prediction" 
        to save memory occupied by activation states. 
        Accelerate training not supported.
        '''

        if compute_lhs_loss and (loss_func_lhs is None or lhs_loss_lambd is None):
            raise RuntimeError("Please specific the loss function and lambda for last hidden states loss!")
        if do_distill and logits is None:
            raise RuntimeError("Please specific the logits for distillation!")
        if not do_distill and labels is None:
            raise RuntimeError("Please specific the labels for kNTP training!")
        ofs = 1 if do_distill else 2

        mths, _, lhs = self.core_fwd(outputs, attention_mask, past_key_values, True)

        detached_mths = mths.detach()
        detached_mths.requires_grad_(False)
        grads: list[torch.Tensor] | torch.Tensor = []
        tot_loss_main = []
        tot_loss_lhs  = []
        for next_i in range(self.max_ahead):
            next_hs = detached_mths[:, :, next_i, :]
            next_hs.requires_grad_(True)
            cut_next_hs = torch.where(
                (attention_mask[:, (next_i + ofs):] == 1).unsqueeze(dim=-1),
                next_hs[:, :-(next_i + ofs), :],
                lhs[:, (next_i + ofs):, :]
            ).contiguous().view(-1, self.hidden_size)
            scaler = attention_mask.numel() / attention_mask.sum(dim=-1).sum(dim=-1)

            cut_next_hs = base_model.model.norm(cut_next_hs)
            cut_next_logits = base_model.lm_head(cut_next_hs)
            cut_next_logits: torch.Tensor

            if do_distill:
                cut_logits = logits[:, (next_i + ofs):].contiguous().view(-1, self.vocab_size)
                cur_loss = loss_func_main(
                    cut_next_logits.log_softmax(dim=-1), 
                    cut_logits.softmax(dim=-1)
                ) * (scaler / self.max_ahead)
                tot_loss_main.append(cur_loss.detach())
            else:
                cut_labels = labels[:, (next_i + ofs):].contiguous().view(-1)
                cur_loss = loss_func_main(cut_next_logits, cut_labels) / self.max_ahead
                tot_loss_main.append(cur_loss.detach())
            
            if compute_lhs_loss:
                cut_lhs = lhs[:, (next_i + ofs):, :].contiguous().view(-1, self.hidden_size)
                add_loss = loss_func_lhs(cut_next_hs, cut_lhs) * (scaler / self.max_ahead)
                cur_loss += add_loss * lhs_loss_lambd
                tot_loss_lhs.append(add_loss.detach())
                
            cur_loss.backward()

            grads.append(next_hs.grad.unsqueeze(-2))
        
        grads = torch.cat(grads, dim=-2)
        mths.backward(gradient=grads)
        
        return tot_loss_main, tot_loss_lhs


class DED_MultiTokenPredictor(nn.Module):
    pass


PNAME_TO_PCLASS = {
    "DE": DE_MultiTokenPredictor,
    "DED": DED_MultiTokenPredictor
}
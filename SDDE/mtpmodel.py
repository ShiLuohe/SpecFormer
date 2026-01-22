
import time

from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
# from accelerate import Accelerator

from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from SDDE.predicter import MultiTokenPredictor, PNAME_TO_PCLASS
from SDDE.cache import SpeculativeDynamicCache

@dataclass
class CausalLMOutputWithMTPLogits(CausalLMOutputWithPast):
    """
    Data class for causal language model (or autoregressive) outputs with multi-token prediction logits.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mtp_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class MTPModel(nn.Module):

    def __init__(
        self, 
        base_model: LlamaForCausalLM,
        max_ahead: int, 
        hook_list: list[int],
        predictor_class: str="DE",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.base_model = base_model
        self.max_ahead = max_ahead
        self.max_predict = self.max_ahead

        self.mtp: MultiTokenPredictor = PNAME_TO_PCLASS[predictor_class](self.base_model, self.max_ahead, hook_list)
        # self.mtp = torch.compile(self.mtp)
        self.base_model.requires_grad_(False)
    
    def forward(self, *args, **kwargs):

        kwargs["output_hidden_states"] = True
        use_mtp = kwargs.pop("use_mtp", True)
        with torch.no_grad():
            outputs: CausalLMOutputWithPast = self.base_model(*args, **kwargs)
        
        if not use_mtp:
            return CausalLMOutputWithMTPLogits(
                loss=outputs.loss,
                logits=outputs.logits,
                past_key_values=kwargs.get("past_key_values", None),
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                mtp_logits=None
            )

        mths, past_key_values, _ = self.mtp(
            outputs, 
            kwargs["attention_mask"], 
            self.max_predict,
            kwargs.get("past_key_values", None),
        )

        mths = self.base_model.model.norm(mths)
        mtp_logits = self.base_model.lm_head(mths)

        return CausalLMOutputWithMTPLogits(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            mtp_logits=mtp_logits
        )

    # Nope we won't do that anymore.
    # @torch.compile # Yes we are doing that
    def fit_batch(self, *args, **kwargs):
        kwargs["output_hidden_states"] = True
        labels: torch.Tensor = kwargs.pop("labels")
        loss_func: nn.Module = kwargs.pop("loss_func")
        grad_acc: int = kwargs.pop("grad_acc_steps", None)
        accelerator = kwargs.pop("accelerator", None)
        distilling: bool = kwargs.pop("do_distill", False)
        if grad_acc is None: grad_acc = 1

        with torch.no_grad():
            outputs: CausalLMOutputWithPast = self.base_model(*args, **kwargs)
        
        # mths (bs, length, self.max_ahead, self.hidden_size)
        mths, _, _ = self.mtp.forward(outputs, kwargs["attention_mask"])
        mths_det = mths.detach()
        mths_det.requires_grad_(True)

        loss_all = []
        for i in range(self.max_ahead):
            ofs = i + (1 if distilling else 2)

            norm_lhs = self.base_model.model.norm(mths_det[:, :-ofs, i, :])
            bs, seq_len, _ = norm_lhs.shape
            logits = self.base_model.lm_head(norm_lhs)

            loss: torch.Tensor = loss_func(
                logits.view(bs * seq_len, -1), 
                labels[:, ofs:, ...].view(bs * seq_len)
            ) / (grad_acc * self.max_ahead)
            loss_all.append(loss.detach().cpu().item() * (grad_acc * self.max_ahead))

            if accelerator is None: loss.backward()
            # https://github.com/huggingface/accelerate/issues/2951 mentioned that the accelerate 
            # backward actually do too many thing for deepspeed, which is not desirable here since
            # all we need is to calculate the gradient of mths.
            elif accelerator.distributed_type == "DEEPSPEED": 
                accelerator.deepspeed_engine_wrapped.engine.backward(loss)
            else:
                accelerator.backward(loss)

        if accelerator is None: mths.backward(gradient=mths_det.grad)
        else: accelerator.backward(mths, gradient=mths_det.grad)

        return loss_all

    def get_mtp_module(self):
        return self.mtp

    def set_max_predict(self, new_max_predict):
        if new_max_predict > self.max_ahead or new_max_predict < 0:
            raise ValueError(f"The available max prediction token count is the integers in [0, {self.max_ahead}].")
        self.max_predict = new_max_predict

    @torch.no_grad()
    def generate(
        self, 
        inputs, 
        max_new_length=64, 
        eos_token=-1,
        return_statics=False,
    ):
        
        # Initializing
        kv_cache = SpeculativeDynamicCache(
            [-2] * self.mtp.config.num_hidden_layers + [-3]
        )
        rounds = 0
        t_hits = 0

        # Prefilling
        outputs: CausalLMOutputWithMTPLogits = self(**inputs, past_key_values=kv_cache)
        new_tokens = outputs.logits[:, -1:, :].argmax(dim=-1) # bs, 1
        spec_tokens = outputs.mtp_logits[:, -1, :, :].argmax(dim=-1) # bs, max_pred
        all_new_tokens = new_tokens.cpu()
        new_input_tokens = torch.cat([new_tokens, spec_tokens], dim=-1)

        tik = time.time()
        while(all_new_tokens.shape[-1] < max_new_length):
            
            rounds += 1

            outputs = self(
                new_input_tokens, 
                attention_mask=torch.ones(
                    [new_input_tokens.shape[0], kv_cache.get_seq_length() + self.max_predict + 1], 
                    device=new_input_tokens.device
                ), 
                past_key_values=kv_cache,
            )

            new_tokens = outputs.logits.argmax(dim=-1)

            verify_token = new_tokens[:, :-1]
            val, pos = (verify_token[0] == spec_tokens[0]).min(dim=-1)
            hit = pos.item() if val.item() == 0 else self.max_predict
            t_hits += hit + 1

            reserved_tokens = new_tokens[:, :hit + 1]
            spec_tokens = outputs.mtp_logits[:, hit, :, :].argmax(dim=-1)
            new_input_tokens = torch.cat([reserved_tokens[:, -1:], spec_tokens], dim=-1)
            kv_cache.reverse(self.max_predict - hit)

            all_new_tokens = torch.cat([all_new_tokens, reserved_tokens.cpu()], dim=-1)

            if (reserved_tokens.flatten() == eos_token).any(): break
        tok = time.time()

        if return_statics:
            return all_new_tokens, t_hits, rounds, tok - tik
        else:
            return all_new_tokens


    @torch.no_grad()
    def generate_no_mtp(
        self, 
        inputs, 
        max_new_length=64, 
        eos_token=-1,
        return_statics=False,
    ):
        
        # Initializing
        kv_cache = SpeculativeDynamicCache(
            [-2] * self.mtp.config.num_hidden_layers + [-3]
        )
        rounds = 0

        # Prefilling
        outputs: CausalLMOutputWithMTPLogits = self(**inputs, past_key_values=kv_cache, use_mtp=False)
        new_tokens = outputs.logits[:, -1:, :].argmax(dim=-1) # bs, 1
        all_new_tokens = new_tokens.cpu()

        tik = time.time()
        while(all_new_tokens.shape[-1] < max_new_length):
            
            rounds += 1

            outputs = self(
                new_tokens, 
                attention_mask=torch.ones(
                    [new_tokens.shape[0], kv_cache.get_seq_length() + 1], 
                    device=new_tokens.device
                ), 
                past_key_values=kv_cache,
                use_mtp=False
            )

            new_tokens = outputs.logits.argmax(dim=-1)
            all_new_tokens = torch.cat([all_new_tokens, new_tokens .cpu()], dim=-1)

            if (new_tokens.flatten() == eos_token).any(): break
        tok = time.time()

        if return_statics:
            return all_new_tokens, rounds, rounds, tok - tik
        else:
            return all_new_tokens
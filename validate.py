
from tqdm import tqdm
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.optim.optimizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2Tokenizer

from traindata import * 
from SDDE.mtpmodel import MTPModel, CausalLMOutputWithMTPLogits

def get_assistant_mask(inputs, bos_token_id):
    input_ids: torch.LongTensor = inputs.input_ids
    im_end_cnt = torch.where(input_ids == bos_token_id, 1, 0).cumsum(dim=-1)
    im_end_max = im_end_cnt.max(dim=-1)
    return im_end_cnt == im_end_max[0]

@torch.no_grad()
def compare_mtp_seq(mpreds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, max_ahead: int):

    mpreds = mpreds.argmax(dim=-1).cpu()
    tot_len = labels.shape[0]

    tot_rnd = 0
    tot_hit = 0

    hit_len_dict = {
        prev + 1: 0 for prev in range(mpreds.shape[-1] + 1)
    }

    i: int
    for i in range(2, tot_len): 
        if mask[i - 2].item(): break
    while i < (tot_len - max_ahead - 2):
        true_cut = labels[i + 2: i + max_ahead + 2]
        pred_cut = mpreds[i]

        val, pos = (true_cut == pred_cut).min(dim=-1)
        min_hit_id = pos.item() if val.item() == 0 else max_ahead
        step = 1 + min_hit_id

        # step = 5 if step > 5 else step

        tot_hit += step
        tot_rnd += 1
        i += step
        hit_len_dict[step] += 1
    
    return tot_rnd, tot_hit, hit_len_dict

@torch.no_grad()
def validate(model: MTPModel, validloader: DataLoader, tokenizer: Qwen2Tokenizer):

    tot_rnd = 0
    hit_len = 0. 
    
    hit_len_dict = {
        prev + 1: 0 for prev in range(model.mtp.max_ahead + 1)
    }

    for idx, inputs in enumerate(tqdm(validloader)):

        assistant_mask = get_assistant_mask(inputs, tokenizer.bos_token_id)
        inputs = inputs.to(model.base_model.device)
        outputs: CausalLMOutputWithMTPLogits = model(**inputs)
        
        for mpreds, labels, mask in zip(outputs.mtp_logits, inputs.input_ids.cpu(), assistant_mask):
            cur_rnd, cur_hit, len_dict = compare_mtp_seq(mpreds, labels, mask, model.max_ahead)
            tot_rnd += cur_rnd
            hit_len += cur_hit
            
            for k, v in len_dict.items():
                hit_len_dict[k] += v
    
        if (idx + 1) % 1000 == 0:
            print(hit_len / tot_rnd)
            print(hit_len_dict)

    return hit_len / tot_rnd

@torch.no_grad()
def main():

    model_path = "/data/shilh/model/Qwen3-4B/"
    device = torch.device("cuda")

    model = AutoModelForCausalLM.from_pretrained(model_path).half().to(device)
    model: Qwen2ForCausalLM
    tokenizer =  AutoTokenizer.from_pretrained(model_path)
    tokenizer: Qwen2Tokenizer
    tokenizer.bos_token_id = 151644

    mtp_model = MTPModel(model, 8, [0, 17, -2, -1])

    mtp_raw: nn.Module = torch.load("checkpoints/may07th.pt", map_location=device, weights_only=False)
    mtp_weight_dict = mtp_raw.state_dict()
    del mtp_raw
    
    # for i in range(4):
    #     mtp_weight_dict[f"norms.{i}.weight"] = torch.ones([2048])

    # for k, v in mtp_weight_dict.items():
    #     print(k, v.shape)
    
    mtp_model.get_mtp_module().load_state_dict(mtp_weight_dict)
    # torch.save(mtp_model.get_mtp_module(), "./checkpoints/mar24th.pt")

    _, testloader = get_uc_200k_model(tokenizer, 1, 1, "Qwen3-4B", 1)

    result = validate(mtp_model, testloader, tokenizer)

    print(result)
    
if __name__ == "__main__":
    main()

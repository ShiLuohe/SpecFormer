
import os
import yaml
import argparse
from tqdm import tqdm
from time import time
import math

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2Tokenizer
from transformers.models.qwen3 import Qwen3ForCausalLM

from traindata import * 
from SDDE.mtpmodel import MTPModel

def get_scheduler_func(
    learning_rate: float,
    learning_rate_min: float,
    total_steps: int,
    warm_up_ratio: Optional[float]=None,
    warm_up_step: Optional[int]=None,
):

    if warm_up_step is None:
        if warm_up_ratio is None:
            raise RuntimeError("Please specify warm_up_ratio or warm_up_step!")
        warm_up_step = total_steps * warm_up_ratio

    def cosing_annealing_with_warmip_schedule_func(cur_step: int) -> float:
        return cur_step / warm_up_step if cur_step < warm_up_step else \
            (learning_rate_min + \
                0.5 * (learning_rate - learning_rate_min) * \
                (1.0 + math.cos((cur_step - warm_up_step) / (total_steps - warm_up_step) * math.pi))\
             ) / learning_rate

    return cosing_annealing_with_warmip_schedule_func

def get_assistant_mask(inputs, bos_token_id):
    input_ids: torch.LongTensor = inputs.input_ids
    im_end_cnt = torch.where(input_ids == bos_token_id, 1, 0).cumsum(dim=-1)
    im_end_max = im_end_cnt.max(dim=-1, keepdim=True)
    return im_end_cnt == im_end_max[0]

class KLDivWithMasking():
    def __init__(self):
        pass

    def set_mask(self, mask: torch.BoolTensor):
        self.mask = mask
    
    def forward(
        self, 
        student_logits: torch.Tensor, 
        teacher_logits: torch.Tensor, 
        padding_mask: Optional[torch.BoolTensor]=None
    ):
        if padding_mask is None:
            padding_mask = self.mask

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)

        kl = F.kl_div(student_log_probs, teacher_probs, reduction='none')
        kl = kl.sum(dim=-1)
        kl = kl * padding_mask
        total_non_pad = padding_mask.sum()
        kl_mean = kl.sum() / total_non_pad

        return kl_mean

def main():

    def load_config(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ================== assign variables ==================

    version_name = cfg["version_name"]

    rnd_seed = cfg["rnd_seed"]
    torch.manual_seed(rnd_seed)

    device_id = cfg["device_id"]
    device = torch.device(device_id)

    model_name = cfg["model_name"]
    bos_token_id = cfg["bos_token_id"]

    do_distillation = cfg["do_distillation"]

    model_path = f"/data/shilh/model/{model_name}/"
    log_path = f"./logs/{version_name}"
    ckpt_path = f"./checkpoints/{version_name}.pt"
    temp_path = ckpt_path[:-3]
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    max_ahead = cfg["max_ahead"]
    hook_list = cfg["hook_list"]

    train_bs = cfg["train_bs"]
    valid_bs = cfg["valid_bs"]
    test_size = cfg["test_size"]
    num_epochs = cfg["num_epochs"]

    lr_max = cfg["lr_max"]
    lr_min = cfg["lr_min"]
    adam_eps = cfg["adam_eps"]

    warm_up_ratio = cfg["warm_up_ratio"]
    grad_acc_steps = cfg["grad_acc_steps"]
    log_steps = cfg["log_steps"]
    save_steps = cfg["save_steps"]
    saved = cfg["saved"]
    # ===========================================================

    loss_func = nn.CrossEntropyLoss()

    # Model, tokenizer, and trainloader loading.
    model = AutoModelForCausalLM.from_pretrained(model_path).half().to(device)
    model: Qwen3ForCausalLM
    tokenizer =  AutoTokenizer.from_pretrained(model_path)
    tokenizer: Qwen2Tokenizer
    tokenizer.bos_token_id = bos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    trainloader, _ = get_uc_200k_model(tokenizer, train_bs, valid_bs, model_name, test_size)

    mtp_model = MTPModel(model, max_ahead, hook_list)

    # mtp_raw: nn.Module = torch.load("checkpoints/may07th.pt", map_location=device, weights_only=False)
    # mtp_weight_dict = mtp_raw.state_dict()
    # del mtp_raw
    # mtp_model.get_mtp_module().load_state_dict(mtp_weight_dict)

    optimizer = torch.optim.AdamW(
        mtp_model.get_mtp_module().parameters(), 
        lr=lr_max,
        eps=adam_eps
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_scheduler_func(
        lr_max, lr_min, 
        (len(trainloader) * num_epochs) // grad_acc_steps, 
        warm_up_ratio 
    ))

    log_writer = SummaryWriter(log_path)
    cur_step = 1
    tot_loss = [0.] * max_ahead

    for epoch_idx in range(num_epochs):
        mtp_model.train()
        for inputs in tqdm(
            trainloader,
            total=len(trainloader),
            desc=f"Epoch {epoch_idx + 1}/{num_epochs}..."
        ):

            # if cur_step <= 3656: 
            #     cur_step += 1
            #     continue
            
            # print(inputs)
            # print(tokenizer.decode(inputs.input_ids[0]))
            # input()
            inputs = inputs.to(device)

            attention_mask: torch.Tensor = inputs.attention_mask.bool()
            assistant_mask = get_assistant_mask(inputs, tokenizer.bos_token_id)
            labels = torch.where(attention_mask & assistant_mask, inputs.input_ids, -100).to(device)

            cur_loss = mtp_model.fit_batch(
                **inputs, 
                labels=labels, 
                loss_func=loss_func,
                accelerator=None,
                distilling=do_distillation,
            )

            # if cur_step > 3648: 
            #     for loss in cur_loss:
            #         print(loss, end=" ")
            #     input(f"\n Step: {cur_step} \n")

            for i in range(max_ahead):
                tot_loss[i] += cur_loss[i]
                # nan_flag |= (cur_loss[i] == float("nan"))

            if cur_step % grad_acc_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                nan_flag = False

            if cur_step % log_steps == 0:
                try:
                    log_writer.add_scalars(
                        "loss_ntp", 
                        {f"loss#{idx + 2}": loss / log_steps for idx, loss in enumerate(tot_loss)}, 
                        cur_step
                    )
                    log_writer.add_scalar("lr", scheduler.get_last_lr()[0], cur_step)
                except OSError:
                    pass

                tot_loss = [0.] * max_ahead

            if cur_step % save_steps == 0:
                saved += 1
                torch.save(mtp_model.get_mtp_module(), temp_path + f"/{saved}.pt")

            cur_step += 1

            # if cur_step >= 65:
            #     break

        torch.save(mtp_model.get_mtp_module(), ckpt_path)

    log_writer.close()


if __name__ == "__main__":
    main()

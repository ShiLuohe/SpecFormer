
from tqdm.auto import tqdm
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.optim.optimizer
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2Tokenizer

from traindata import * 
from SDDE.mtpmodel import MTPModel

from accelerate import Accelerator
from accelerate.utils import LoggerType, ProjectConfiguration

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
        return cur_step / warm_up_step if  cur_step < warm_up_step else \
            (learning_rate_min + \
                0.5 * (learning_rate - learning_rate_min) * \
                (1.0 + math.cos((cur_step - warm_up_step) / (total_steps - warm_up_step) * math.pi))\
             ) / learning_rate

    return cosing_annealing_with_warmip_schedule_func

def get_assistant_mask(inputs, bos_token_id):
    input_ids: torch.LongTensor = inputs.input_ids
    im_end_cnt = torch.where(input_ids == bos_token_id, 1, 0).cumsum(dim=-1)
    im_end_max = im_end_cnt.max(dim=-1)
    return im_end_cnt == im_end_max[0]

def main():

    version_name = "mar19th"

    rnd_seed = 14336
    torch.manual_seed(rnd_seed)
    # device_id = "cuda"

    model_path = "/data1/shilh/model/Qwen2.5-3B-Instruct/"
    log_path = f"./logs/{version_name}"
    ckpt_path = f"./checkpoints/{version_name}.pt"

    max_ahead = 8

    train_bs = 4
    valid_bs = 1
    test_size = 0.01
    num_epochs = 1

    lr_max = 5e-4
    lr_min = 1e-5
    adam_eps = 5e-3

    warm_up_ratio = 0.05
    grad_acc_steps = 16

    proj_config = ProjectConfiguration(project_dir=".", logging_dir=f"logs/{version_name}/")
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_acc_steps,
        log_with=LoggerType.TENSORBOARD,
        project_config=proj_config
    )

    accelerator.init_trackers(f"MTP: {version_name}")
    device = accelerator.device

    loss_func = nn.CrossEntropyLoss()

    # Model, tokenizer, and trainloader loading.
    model = AutoModelForCausalLM.from_pretrained(model_path).half().to(device)
    model: Qwen2ForCausalLM
    tokenizer =  AutoTokenizer.from_pretrained(model_path)
    tokenizer: Qwen2Tokenizer  
    tokenizer.bos_token_id = 151644
    trainloader, _ = get_uc_200k(tokenizer, train_bs, valid_bs, test_size)

    mtp_model = MTPModel(model, max_ahead)

    optimizer = torch.optim.AdamW(
        mtp_model.get_mtp_module().parameters(), 
        lr = lr_max,
        eps=adam_eps
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_scheduler_func(
        lr_max, lr_min, 
        (len(trainloader) * num_epochs) // grad_acc_steps, 
        warm_up_ratio 
    ))

    mtp_model, trainloader, optimizer, scheduler = \
        accelerator.prepare(mtp_model, trainloader, optimizer, scheduler)

    cur_step = 1

    for epoch_idx in range(num_epochs):
        mtp_model.train()
        for inputs in tqdm(
            trainloader,
            total=len(trainloader),
            desc=f"Epoch {epoch_idx + 1}/{num_epochs}..."
        ):
            with accelerator.accumulate(mtp_model):

                attention_mask: torch.Tensor = inputs.attention_mask.bool()
                assistant_mask = get_assistant_mask(inputs, tokenizer.bos_token_id)
                labels = torch.where(attention_mask & assistant_mask, inputs.input_ids, -100)

                inputs.to(device)

                cur_loss = mtp_model.fit_batch(
                    **inputs, 
                    labels=labels, 
                    loss_func=loss_func,
                    accelerator=accelerator
                )

                optimizer.step()
                scheduler.step()

                accelerator.log({f"loss#{idx}": loss for idx, loss in enumerate(cur_loss)}, step=cur_step)

                cur_step = cur_step + 1

                if cur_step == 65:
                    break

    accelerator.end_training()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        mtp_model_save: MTPModel = accelerator.unwrap_model(mtp_model)
        mtprdictor = mtp_model_save.get_mtp_module()
        mtprdictor._save_to_state_dict(f"{version_name}/")

if __name__ == "__main__":
    main()

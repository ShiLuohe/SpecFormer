
from tqdm import tqdm
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


def main():

    model_path = "/data1/shilh/model/Qwen2.5-3B-Instruct/"
    device = torch.device("cuda")

    model = AutoModelForCausalLM.from_pretrained(model_path).half().to(device)
    model: Qwen2ForCausalLM
    tokenizer =  AutoTokenizer.from_pretrained(model_path)
    tokenizer: Qwen2Tokenizer  

    mtp_model = MTPModel(model, 8)

    _, testloader = get_uc_200k(tokenizer, 1, 1, 1)

    prompts = [
        "Hi! My name is Qwen!",
        "CUDA CUDA CUDA!!! More cuda cores!"
    ] * 3

    inputs = tokenizer(prompts, return_tensors="pt").to(device)

    outputs = mtp_model(**inputs)

    print(outputs.logits.shape, outputs.mtp_logits.shape)
    

if __name__ == "__main__":
    main()

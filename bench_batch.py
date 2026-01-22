
import math
import time
import argparse
import logging
from tqdm import tqdm
from typing import Optional

import torch
import torch.nn as nn
import torch.optim.optimizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2Tokenizer

from traindata import * 
from SDDE.mtpmodel import MTPModel, CausalLMOutputWithMTPLogits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test model with arguments")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--bs", type=int, default=56, help="Batch size")
    parser.add_argument("--entries", type=int, default=32, help="Number of entries")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to MTP checkpoint")
    return parser.parse_args()

def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def main() -> None:
    setup_logger()
    args = parse_args()
    logging.info(f"Starting model test with parameters:")
    logging.info(f"  model_path: {args.model_path}")
    logging.info(f"  device: {args.device}")
    logging.info(f"  batch size (bs): {args.bs}")
    logging.info(f"  entries: {args.entries}")
    logging.info(f"  checkpoint_path: {args.checkpoint_path}")

    model_path: str = args.model_path
    device: torch.device = torch.device(args.device)
    bs: int = args.bs
    entries: int = args.entries

    model: Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path).half().to(device)  # type: ignore
    tokenizer: Qwen2Tokenizer = AutoTokenizer.from_pretrained(model_path)  # type: ignore
    tokenizer.bos_token_id = 151644

    mtp_model: MTPModel = MTPModel(model, 8, [0, 17, -2, -1])
    mtp_model.set_max_predict(4)
    mtp_model.eval()

    mtp_raw: nn.Module = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    mtp_weight_dict = mtp_raw.state_dict()
    del mtp_raw
    mtp_model.get_mtp_module().load_state_dict(mtp_weight_dict, assign=True)

    _, testloader = get_uc_200k(tokenizer, 1, 1, 1)

    time_tot_mtp = 0.
    time_tot_ntp = 0.

    num_tot_gen = 0
    num_tot_rnd = 0

    for idx, inputs in enumerate(tqdm(testloader, total=entries)):
        
        if idx >= entries: break

        inputs.input_ids = inputs.input_ids.repeat(bs, 1)
        inputs.attention_mask = inputs.attention_mask.repeat(bs, 1)

        _, num_gen, num_rnd, dec_time = mtp_model.generate(
            inputs.to(device), 
            max_new_length=1024,
            eos_token=tokenizer.eos_token_id,
            return_statics=True
        )
        time_tot_mtp += dec_time
        num_tot_gen += num_gen
        num_tot_rnd += num_rnd
        
        _, _, _, dec_time = mtp_model.generate_no_mtp(
            inputs.to(device), 
            max_new_length=1024,
            eos_token=tokenizer.eos_token_id,
            return_statics=True
        )
        time_tot_ntp += dec_time

    print(bs)
    print(time_tot_mtp, time_tot_ntp, time_tot_ntp / time_tot_mtp)
    print(num_tot_rnd, num_tot_gen, num_tot_gen / num_tot_rnd)

if __name__ == "__main__":
    main()

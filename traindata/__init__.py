import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from transformers.models.qwen2 import Qwen2Tokenizer
from datasets import load_dataset

def dict_convert(share_gpts: list[dict]):
    ret_dicts = []
    for entry in share_gpts:
        ret_dicts.append(
            {
                "role": "assistant" if entry["from"] == "gpt" else "user",
                "content": entry["value"]
            }
        )
    return ret_dicts

def get_ntp_col_func(tokenizer: Qwen2Tokenizer):
    
    def collate_func(inputs: list[dict]):

        jsons = [dict_convert(entry["conversations"]) for entry in inputs]
        try: # There are some bad data in ShareGPT that would break Qwen2 tokenizer.
            texts = tokenizer.apply_chat_template(jsons, tokenize=False)
        except: # Returns None when it happened.
            return None, None
        
        tokenized = tokenizer(texts, padding="longest", truncation="longest_first", return_tensors="pt")
        labels = torch.where(tokenized.attention_mask == 1, tokenized.input_ids, -100)

        return tokenized, labels
        
    return collate_func

def get_sft_col_func(tokenizer: Qwen2Tokenizer):
    
    def collate_func(inputs: list[dict]):

        jsons = [entry["messages"] for entry in inputs]

        inputs = tokenizer.apply_chat_template(
            jsons, 
            tokenize=True,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            return_dict=True,
            # return_assistant_tokens_mask=True
        )
    
        return inputs
        
    return collate_func

def get_sharegpt_zh(
    tokenizer: PreTrainedTokenizer, 
    train_batch_size: int,
    valid_batch_size: int,
    test_size: int | float
) -> DataLoader:
    
    ds = load_dataset("json", data_files=f"./traindata/ShareGPT_V3_unfiltered_cleaned_split.json")["train"]
    ds = ds.train_test_split(test_size=test_size)

    collate_func = get_ntp_col_func(tokenizer)

    return DataLoader(
        ds["train"], 
        train_batch_size, 
        shuffle=True, 
        collate_fn=collate_func 
    ), DataLoader(
        ds["test"], 
        valid_batch_size, 
        shuffle=False, 
        collate_fn=collate_func 
    )

def get_sharegpt(
    tokenizer: PreTrainedTokenizer, 
    train_batch_size: int,
    valid_batch_size: int,
    test_size: int | float
) -> DataLoader:
    
    ds = load_dataset("json", data_files=f"./traindata/ShareGPT_V3_unfiltered_cleaned_split.json")["train"]
    ds = ds.train_test_split(test_size=test_size)

    print(ds)
    collate_func = get_ntp_col_func(tokenizer)

    return DataLoader(
        ds["train"], 
        train_batch_size, 
        shuffle=True,
        collate_fn=collate_func
    ), DataLoader(
        ds["test"], 
        valid_batch_size, 
        shuffle=False,
        collate_fn=collate_func
    )

def get_uc_200k(
    tokenizer: PreTrainedTokenizer, 
    train_batch_size: int,
    valid_batch_size: int,
    test_size: int | float
) -> DataLoader:
    
    train_ds = load_dataset(
        "parquet", 
        data_files="traindata/uc_Qwen2.5-3B-Instruct/train_Qwen2.5-3B-Instruct.parquet"
    )["train"]

    test_ds = load_dataset(
        "parquet", 
        data_files="traindata/uc_Qwen2.5-3B-Instruct/test_Qwen2.5-3B-Instruct.parquet"
    )["train"]

    collate_func = get_sft_col_func(tokenizer)

    return DataLoader(
        train_ds, 
        train_batch_size, 
        shuffle=True,
        collate_fn=collate_func
    ), DataLoader(
        test_ds, 
        valid_batch_size, 
        shuffle=False,
        collate_fn=collate_func
    )

def get_uc_200k_model(
    tokenizer: PreTrainedTokenizer, 
    train_batch_size: int,
    valid_batch_size: int,
    model_name: str,
    test_size: int | float
) -> DataLoader:
    
    train_ds = load_dataset(
        "parquet", 
        data_files=f"traindata/uc_{model_name}/train_{model_name}.parquet"
    )["train"]

    test_ds = load_dataset(
        "parquet", 
        data_files=f"traindata/uc_{model_name}/test_{model_name}.parquet"
    )["train"]

    collate_func = get_sft_col_func(tokenizer)

    return DataLoader(
        train_ds, 
        train_batch_size, 
        shuffle=True,
        collate_fn=collate_func
    ), DataLoader(
        test_ds, 
        valid_batch_size, 
        shuffle=False,
        collate_fn=collate_func
    )
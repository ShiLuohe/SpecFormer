
import os

from datasets import load_dataset, DatasetDict

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from transformers.models.qwen2 import Qwen2Tokenizer

BATCH_SIZE = 128

def get_process_batch(llm: LLM, sampling_params: SamplingParams, tokenizer: Qwen2Tokenizer):

    def process_batch(batch: dict):
        conversations = [conversation[:1] for conversation in batch["messages"]]
        conversations: list[list[dict]]

        texts = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
            # enable_thinking=False,
        )

        outputs = llm.generate(
            texts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        for conversation, output in zip(conversations, outputs):
            conversation.append({
                "content": output.outputs[0].text,
                "role": "assistant",
            }) 

        batch["messages"] = conversations
        return batch

    return process_batch

def main(model_name: str, model_path: str, train_path: list[str], test_path: list[str], idx: int):

    train_dataset = load_dataset("parquet", data_files=train_path)["train"]

    test_dataset = load_dataset("parquet", data_files=test_path)["train"]

    raw_dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=16384
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=0,
        max_tokens=512,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    rewrite_dataset: DatasetDict = raw_dataset.map(
        get_process_batch(llm, sampling_params, tokenizer), 
        batched=True, 
        batch_size=BATCH_SIZE
    )

    if not os.path.exists(f"traindata/uc_{model_name}/"):
        os.makedirs(f"traindata/uc_{model_name}/")
    
    rewrite_dataset["train"].to_parquet(f"traindata/uc_{model_name}/train_{model_name}_{idx}.parquet")
    rewrite_dataset["test"].to_parquet(f"traindata/uc_{model_name}/test_{model_name}_{idx}.parquet")


if __name__ == "__main__":

    train_path_1 = [
        "data_rewrite/raw-parquet/train_gen-00000-of-00003-a6c9fb894be3e50b.parquet",
        "data_rewrite/raw-parquet/train_gen-00001-of-00003-d6a0402e417f35ca.parquet",
        "data_rewrite/raw-parquet/train_gen-00002-of-00003-c0db75b92a2f48fd.parquet",
    ]

    test_path_1 = [
        "data_rewrite/raw-parquet/test_gen-00000-of-00001-3d4cd8309148a71f.parquet",
    ]

    train_path_2 = [
        "data_rewrite/raw-parquet/train_sft-00000-of-00003-a3ecf92756993583.parquet",
        "data_rewrite/raw-parquet/train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet",
        "data_rewrite/raw-parquet/train_sft-00002-of-00003-ee46ed25cfae92c6.parquet",
    ]

    test_path_2 = [
        "data_rewrite/raw-parquet/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
    ]

    main("Llama-3.2-3B", "/data/shilh/model/Llama-3.2-3B-Instruct/", train_path_2, test_path_2, 2)
    main("Llama-3.2-3B", "/data/shilh/model/Llama-3.2-3B-Instruct/", train_path_1, test_path_1, 1)

from datasets import load_dataset, concatenate_datasets

def main():

    name = "Llama-3.2-3B"

    train_ds_1 = load_dataset(
        "parquet", 
        data_files=f"traindata/uc_{name}/train_{name}_1.parquet"
    )["train"]

    test_ds_1 = load_dataset(
        "parquet", 
        data_files=f"traindata/uc_{name}/test_{name}_1.parquet"
    )["train"]

    train_ds_2 = load_dataset(
        "parquet", 
        data_files=f"traindata/uc_{name}/train_{name}_2.parquet"
    )["train"]

    test_ds_2 = load_dataset(
        "parquet", 
        data_files=f"traindata/uc_{name}/test_{name}_2.parquet"
    )["train"]

    train_ds = concatenate_datasets([train_ds_1, train_ds_2])
    test_ds = concatenate_datasets([test_ds_1, test_ds_2])

    train_ds.to_parquet(f"traindata/uc_{name}/train_{name}.parquet")
    test_ds.to_parquet(f"traindata/uc_{name}/test_{name}.parquet")

if __name__ == "__main__":
    main()
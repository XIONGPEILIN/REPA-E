import re

with open("train_repae.py", "r") as f:
    content = f.read()

replacement = """
    # Setup data
    if args.use_hf_imagenet:
        train_dataset = HFImageNetDataset()
    elif args.parquet_data_path:
        train_dataset = ParquetDataset(args.parquet_data_path)
    else:
        train_dataset = CustomINH5Dataset(args.data_dir)
"""

content = re.sub(r'    # Setup data\n    train_dataset = CustomINH5Dataset\(args\.data_dir\)', replacement.lstrip("\n"), content)

with open("train_repae.py", "w") as f:
    f.write(content)

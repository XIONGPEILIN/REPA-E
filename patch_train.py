import re
with open("train_repae.py", "r") as f:
    content = f.read()

content = content.replace("from dataset import CustomINH5Dataset", "from dataset import CustomINH5Dataset, ParquetDataset, HFImageNetDataset")

dataset_logic_old = """    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")"""

dataset_logic_new = """    if args.use_hf_imagenet:
        if accelerator.is_main_process:
            logger.info(f"Loading HF dataset ILSVRC/imagenet-1k with cache_dir={args.hf_cache_dir}")
        train_dataset = HFImageNetDataset(cache_dir=args.hf_cache_dir, resolution=args.resolution)
    elif args.parquet_data_path is not None:
        if accelerator.is_main_process:
            logger.info(f"Loading Parquet dataset from {args.parquet_data_path}")
        train_dataset = ParquetDataset(args.parquet_data_path, resolution=args.resolution)
    else:
        train_dataset = CustomINH5Dataset(args.data_dir)

    if accelerator.is_main_process:
        if args.use_hf_imagenet:
            dataset_desc = f"hf_cache={args.hf_cache_dir}"
        elif args.parquet_data_path is not None:
            dataset_desc = args.parquet_data_path
        else:
            dataset_desc = args.data_dir
        logger.info(f"Dataset contains {len(train_dataset):,} images ({dataset_desc})")"""

content = re.sub(r'    train_dataset = CustomINH5Dataset\(args.data_dir\)\s*if accelerator.is_main_process:\s*logger.info\(f"Dataset contains \{len\(train_dataset\):,\} images \(\{args.data_dir\}\)"\)', dataset_logic_new, content)

resume_logic_old = """        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt_path = f'{args.cont_dir}/checkpoints/{ckpt_name}'
        assert os.path.isfile(ckpt_path), f'Missing resume checkpoint: {ckpt_path}'"""

resume_logic_new = """        if args.continue_train_exp_dir is None:
            raise ValueError("--continue-train-exp-dir is required when --resume-step > 0")
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt_path = f'{args.continue_train_exp_dir}/checkpoints/{ckpt_name}'
        assert os.path.isfile(ckpt_path), f'Missing resume checkpoint: {ckpt_path}'"""

content = content.replace(resume_logic_old, resume_logic_new)

arg_old = """    parser.add_argument("--data-dir", type=str, default="data")"""
arg_new = """    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--parquet-data-path", type=str, default=None, help="Path to the Parquet dataset")
    parser.add_argument("--use-hf-imagenet", action="store_true", help="Use datasets.load_dataset('ILSVRC/imagenet-1k') for training data")
    parser.add_argument("--hf-cache-dir", type=str, default=None, help="Cache directory for Hugging Face datasets")"""

content = content.replace(arg_old, arg_new)

with open("train_repae.py", "w") as f:
    f.write(content)


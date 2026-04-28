import torch
from dataset import ParquetDataset
import os

def test():
    data_dir = "/host/home/yanai-lab/Sotsuken24/xiong-p/SiT-GM-moe/SiT/ImageNet/data/"
    print(f"Testing ParquetDataset with dir: {data_dir}")
    
    try:
        dataset = ParquetDataset(data_dir, resolution=256)
        print(f"Dataset loaded. Total items: {len(dataset)}")
        
        img, label = dataset[0]
        print(f"Successfully loaded item 0.")
        print(f"Image shape: {img.shape}")
        print(f"Image dtype: {img.dtype}")
        print(f"Label: {label} (type: {label.dtype})")
        
        assert img.shape == (3, 256, 256), f"Wrong image shape: {img.shape}"
        assert isinstance(label, torch.Tensor), "Label should be a torch.Tensor"
        
        print("Data loading test PASSED!")
    except Exception as e:
        print(f"Data loading test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()

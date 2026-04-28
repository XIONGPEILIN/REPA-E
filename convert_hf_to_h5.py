import os
import io
import json
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
from utils import center_crop_arr

# Paths
PARQUET_DIR = "/host/home/yanai-lab/Sotsuken24/xiong-p/SiT-GM-moe/SiT/ImageNet/data/"
OUTPUT_DIR = "/host/ssd2/xiong-p/repa/data"
H5_PATH = os.path.join(OUTPUT_DIR, "images.h5")
JSON_PATH = os.path.join(OUTPUT_DIR, "images_h5.json")

RESOLUTION = 256
NUM_WORKERS = 28

def process_row(row_tuple):
    idx, row = row_tuple
    img_data = row['image']['bytes']
    label = int(row['label'])
    
    try:
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        img_np = np.array(img)
        img_cropped = center_crop_arr(img_np, RESOLUTION)
        
        # Save as PNG uncompressed in memory
        out_img = Image.fromarray(img_cropped)
        out_bytes = io.BytesIO()
        out_img.save(out_bytes, format="PNG", compress_level=0, optimize=False)
        
        idx_str = f"{idx:08d}"
        arcname = f"{idx_str[:5]}/img{idx_str}.png"
        
        return arcname, out_bytes.getvalue(), label
    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    parquet_files = sorted([f for f in os.listdir(PARQUET_DIR) if f.startswith("train-") and f.endswith(".parquet")])
    
    all_filenames = []
    all_labels = []
    
    global_idx = 0
    
    print(f"Found {len(parquet_files)} parquet files.")
    
    with h5py.File(H5_PATH, 'w') as h5f:
        pool = mp.Pool(NUM_WORKERS)
        
        for p_file in tqdm(parquet_files, desc="Processing Parquet files"):
            df = pd.read_parquet(os.path.join(PARQUET_DIR, p_file))
            
            # Prepare rows for pool
            rows = []
            for _, row in df.iterrows():
                rows.append((global_idx, row))
                global_idx += 1
            
            results = pool.map(process_row, rows)
            
            for res in results:
                if res:
                    arcname, img_bytes, label = res
                    h5f.create_dataset(arcname, data=np.frombuffer(img_bytes, dtype='uint8'))
                    all_filenames.append(arcname)
                    all_labels.append([arcname, label])
        
        pool.close()
        pool.join()

        # Store dataset.json inside H5
        print("Storing dataset.json inside H5...")
        dataset_json = {"labels": all_labels}
        dataset_json_bytes = json.dumps(dataset_json).encode('utf-8')
        h5f.create_dataset("dataset.json", data=np.frombuffer(dataset_json_bytes, dtype='uint8'))

    # Store images_h5.json on disk
    print(f"Storing images_h5.json at {JSON_PATH}...")
    with open(JSON_PATH, 'w') as f:
        json.dump(all_filenames, f)

    print("Done!")

if __name__ == "__main__":
    main()

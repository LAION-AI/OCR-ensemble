from img2dataset import download
import os 
import webdataset as wds 
import numpy as np

def identity(x):
    return x 

def touint8(x):
    return (255*x).astype(np.uint8)


def load_dataset(path, parquetpath, image_size=512, number_sample_per_shard=1000, use_uint8=True):
    if not os.path.exists(path):
        print("downloading dataset...")
        download(
            processes_count=6,
            thread_count=12,
            url_list=parquetpath,
            image_size=image_size,
            output_folder=path,
            output_format="webdataset",
            input_format="parquet",
            url_col="URL",
            caption_col="TEXT",
            enable_wandb=False,
            number_sample_per_shard=number_sample_per_shard,
            distributor="multiprocessing",
        )
    dataset = wds.WebDataset(path + "/{00000..0010}.tar").decode("rgb").to_tuple("jpg", "txt", "__key__")
    if use_uint8:
        dataset.map_tuple(touint8, identity, identity)
    return dataset


def load_dataset_1K():
    path = "../data/laion2b-en-1K"
    parquet_path = "../data/laion2b-en-1K.parquet"
    return load_dataset(path, parquet_path, number_sample_per_shard=100)


def load_dataset_10K():
    path = "../data/laion2b-en-10K"
    parquet_path = "../data/laion2b-en-10K.parquet"
    return load_dataset(path, parquet_path)


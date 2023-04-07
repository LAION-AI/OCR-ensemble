from img2dataset import download
import os 
import webdataset as wds 

def load_dataset_1K():
    path = "../data/laion2b-en-1K"
    if not os.path.exists(path):
        print("downloading dataset...")
        download(
            processes_count=6,
            thread_count=6,
            url_list="../data/laion2b-en-1K.parquet",
            image_size=256,
            output_folder=path,
            output_format="webdataset",
            input_format="parquet",
            url_col="URL",
            caption_col="TEXT",
            enable_wandb=False,
            number_sample_per_shard=100,
            distributor="multiprocessing",
        )
    dataset = wds.WebDataset(path + "/{00000..00010}.tar").decode("rgb").to_tuple("jpg", "txt")
    return dataset

        
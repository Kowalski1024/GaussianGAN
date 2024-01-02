import os
import gdown
import shutil
import tempfile
import subprocess


if __name__ == '__main__':
    download_name = 'cars_train.zip'
    url = 'https://drive.google.com/uc?id=1bThUNtIHx4xEQyffVBSf82ABDDh2HlFn'
    output_dataset_name = '/net/pr2/projects/plgrid/plggtriplane/twojnar/cars_128.zip'

    dir_path = os.path.dirname(os.path.realpath(__file__))

    print("Downloading data...")
    gdown.download(url, output_dataset_name, quiet=False)
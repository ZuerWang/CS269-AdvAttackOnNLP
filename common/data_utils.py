import tarfile

import datasets
from datasets import Dataset

def get_dataset(name="imdb", split_rate=0.8):
  if 0 < split_rate < 1:
    train_percentage = int(split_rate * 100)
    val_percentage = 100 - train_percentage
    split_list = [
      f"train[:{train_percentage}%]",
      f"train[-{val_percentage}%:]",
      "test"
    ]
    return datasets.load_dataset(name, split=split_list)
  return datasets.load_dataset(name, split=['train', 'test'])

def download_model(cwd):
  dir_name = cwd/"data"/"imdb"/"saved_model"
  model_name = "imdb_bert_base_uncased_finetuned_training"
  file_name = f"{model_name}.tar.gz"
  file_path = dir_name/file_name
  shared_id = "1yrRE23s3u5vDiXdHhGfkNTVgSD_uTW4m"
  url = f"https://drive.google.com/uc?id={shared_id}"

  dir_name.mkdir(parents=True, exist_ok=True)
  if not (dir_name/model_name).exists():
    gdown.download(url, str(file_path), quiet=False)
    with tarfile.open(file_path) as tar:
      tar.extractall(dir_name)
    file_path.unlink(missing_ok=True)
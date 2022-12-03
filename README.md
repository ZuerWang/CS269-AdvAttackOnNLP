# CS269-AdvAttackOnNLP

This is the repo for the class project of CS 269 Fall 2022

## Setup (without GPU)

```bash
conda create -y --name advAttack python=3.8.5
conda activate advAttack
cd /path/to/project/root/
pip install -r requirements.txt
pip install -e .
cd TextAttack
pip install .[dev]
python -m spacy download en_core_web_lg
```

## Setup (with GPU)

```bash
conda create -y --name advAttack python=3.8.5
conda activate advAttack

cd /path/to/project/root/
# install gpu version with cuda 11.0
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install -e .
cd TextAttack
pip install .[dev]
python -m spacy download en_core_web_lg
```

## Activate Environment

```bash
conda activate advAttack
```


## Our method
To run our method, you can run the following command:

```bash
python main.py --dataset <imdb or yelp_polarity> [--phrase_off]
```
You can specify the dataset to run on, by default it will use yelp_polarity if no dataset option is specified. You can use the ``--phrase_off`` flag to turn off the phrase tokenization. Without this flag, by default it will use phrase tokenization.

You can also use ``python main.py -h`` to get the argument specific usage info. 



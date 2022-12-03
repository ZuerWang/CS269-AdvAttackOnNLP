from pathlib import Path
from pprint import pprint
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import torch
mixed_precision = True
try:
  from apex import amp
except ImportError:
  mixed_precision = False
import datasets
from datasets import concatenate_datasets
from tqdm import tqdm
from transformers import (
  AutoTokenizer,
  AutoModelForSequenceClassification,
  AutoModelForMaskedLM,
)
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from common.data_utils import get_dataset, download_model
from model.tokenizer import PhraseTokenizer
from model.attacker import Attacker
from model.evaluate import evaluate

from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.goal_functions import UntargetedClassification
from textattack.datasets import HuggingFaceDataset

from textattack import Attack
from textattack import AttackArgs
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.constraints.semantics.sentence_encoders import BERT
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapMaskedLM

import time
import json


if __name__ == "__main__":
  start_time = time.time()

  # 0. init setup
  tf.get_logger().setLevel("ERROR")

  # limit tf gpu memory to runtime allocation
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)

  print('Load word/sentence similarity embedding')
  # retrieve the USE encoder and counter fitting vector embeddings
  url = "https://tfhub.dev/google/universal-sentence-encoder/4"

  with tf.device("/cpu:0"):
    encoder_use = hub.load(url)
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if device == "cuda":
    torch.cuda.empty_cache()

  print(f"Running on {device}")

  # Import the dataset
  print('Load dataset')

  ds_name = "imdb"
  dataset = HuggingFaceDataset("ag_news", None, "test")

  #embeddings_cf = np.load('./data/sim_mat/embeddings_cf.npy')
  #word_ids = np.load('./data/sim_mat/word_id.npy',allow_pickle='TRUE').item()
    
  #cwd/"saved_model"/"imdb_bert_base_uncased_finetuned_normal"
  if ds_name == "imdb":
    target_model_name = "bert-base-uncased-imdb"
  elif ds_name == "yelp_polarity":
    target_model_name = "bert-base-uncased-yelp-polarity"
  target_model_path = f"textattack/{target_model_name}"

  use_cuda = torch.cuda.is_available()
  if use_cuda:
    t = torch.cuda.get_device_properties(0).total_memory *9.31323e-10 #GiB
    r = torch.cuda.memory_reserved(0) *9.31323e-10 #GiB
    a = torch.cuda.memory_allocated(0) *9.31323e-10  #GiB
    f = (r-a) * 1024 # free inside cache [MiB]
    
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    print(f'Allocated/Reserved/Total Memory [GiB]: {a}/{r}/{t}')
    print(f'Free Memory [MiB]: {f}')
    

  print('Obtain model and tokenizer')
  tokenizer = AutoTokenizer.from_pretrained(target_model_path)
  target_model = AutoModelForSequenceClassification.from_pretrained(target_model_path).to(device)

  target_model_wrapper = HuggingFaceModelWrapper(target_model, tokenizer)

  # Create the goal function using the model
  goal_function = UntargetedClassification(target_model_wrapper)

  phrase_tokenizer = PhraseTokenizer()
  mlm_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").to(device)

  # turn models to eval model since only inference is needed
  target_model.eval()
  mlm_model.eval()

  # tokenize the dataset to include words and phrases
  test_ds = test_ds.map(phrase_tokenizer.tokenize)

  # create the attacker
  params = {'k':15, 'beam_width':8, 'conf_thres':3.0, 'sent_semantic_thres':0.7, 'change_threshold':0.2}
  attacker = Attacker(phrase_tokenizer, tokenizer, target_model, mlm_model, encoder_use,  device, **params) #embeddings_cf,

  # Candidate size K is set to 48 for bert-attack
  transformation = WordSwapMaskedLM(method="bert-attack", max_candidates=16) # original 48
  
  # Don't modify the same word twice or stopwords.
  constraints = [RepeatModification(), StopwordModification()]
  constraints.append(MaxWordsPerturbed(max_percent=0.2))

  '''
  use_constraint = UniversalSentenceEncoder(
      threshold=0.7,
      metric="cosine",
      compare_against_original=True,
      window_size=None,
  )
  constraints.append(use_constraint)
  '''
  sent_encoder = BERT(
      model_name="stsb-distilbert-base", threshold=0.9, metric="cosine"
  )
  constraints.append(sent_encoder)

  
  #
  # Goal is untargeted classification.
  #
  goal_function = UntargetedClassification(model_wrapper)
  #
  # "We first select the words in the sequence which have a high significance
  # influence on the final output logit. Let S = [w0, ··· , wi ··· ] denote
  # the input sentence, and oy(S) denote the logit output by the target model
  # for correct label y, the importance score Iwi is defined as
  # Iwi = oy(S) − oy(S\wi), where S\wi = [w0, ··· , wi−1, [MASK], wi+1, ···]
  # is the sentence after replacing wi with [MASK]. Then we rank all the words
  # according to the ranking score Iwi in descending order to create word list
  # L."
  search_method = GreedyWordSwapWIR(wir_method="unk")

  attack = Attack(goal_function, constraints, transformation, search_method)

  attack_args = AttackArgs(num_examples=10)

  attacker = Attacker(attack, dataset, attack_args)

  attack_results = attacker.attack_dataset()

  output_entries = []
  adv_examples = []
  pred_failures = 0

  dir_path = f'./data/features/{ds_name}'
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

  suffix = f"{params['k']}_{params['beam_width']}_{params['sent_semantic_thres']}"
  output_pth = f'{dir_path}/features_{suffix}.json'
  eval_pth = f'{dir_path}/eval_{suffix}.json'
  adv_set_pth = f'{dir_path}/adv_{suffix}.json'

  # clean output file
  #f = open(output_pth, "w")
  #f.writelines('')
  #f.close()
  
  print('\nstart attack')
  # attack the target model
  progressbar = tqdm(test_ds, desc="substitution", unit="doc")
  with torch.no_grad():
    for i, entry in enumerate(progressbar):
      entry = attacker.attack(entry)
      #print(f"success: {entry['success']}, change -words: {entry['word_changes']}, -phrases: {entry['phrase_changes']}")
      #print('original text: ', entry['text'])
      #print('adv text: ', entry['final_adv'])
      #print('changes: ', entry['changes'])

      new_entry = { k: entry[k] for k in {'text', 'label',  'pred_success', 'success', 'changes', 'final_adv',  'word_changes', 'phrase_changes', 'word_num', 'phrase_num',   'query_num', 'phrase_len' } }

      if not entry['pred_success']:
        pred_failures += 1
      else:
        seq_embeddings = encoder_use([entry['final_adv'], entry['text']])
        semantic_sim =  np.dot(*seq_embeddings)
        new_entry['semantic_sim'] = float(semantic_sim)
        adv_examples.append({k: entry[k] for k in {'label', 'text'}})

      #json.dump(new_entry, open(output_pth, "a"), indent=2)
      output_entries.append(new_entry)

      if (i + 1) % 100 == 0:
        evaluate(output_entries, pred_failures, eval_pth, params)

  json.dump(output_entries, open(output_pth, "w"), indent=2)
  json.dump(adv_examples, open(adv_set_pth, "w"), indent=2)
  print("--- %.2f mins ---" % (int(time.time() - start_time) / 60.0))

  evaluate(output_entries, pred_failures, eval_pth, params)
from pprint import pprint

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from model.tokenizer import get_filtered_k_phrases, filter_unwanted_phrases

def get_phrase_masked_list(text, sorted_phrase_offsets, sorted_n_words_in_phrase):
  """retrieve phrase masked list.
  Args:
    text [str]: original text
    sorted_phrase_offsets List[tuple(start, end), ...]: sorted offsets by importance
    sorted_n_words_in_phrase List[int]: sorted number of words in phrases
  Returns:
    phrase_masked_list: len(phrase_masked_list) == len(sorted_n_words_in_phrase)
      for each phrase in the list, 1 < len(list_of_masked_text) < n_words_in_phrase
  """
  phrase_masked_list = []
  # this triple for loop would be super slow
  # TODO: figure a way to optimize it
  for i, (n, (start, end)) in enumerate(zip(sorted_n_words_in_phrase, sorted_phrase_offsets)):
    phrase_masked_list.append([])
    for n_mask in range(1, n+1):
      # make sure there are spaces around it
      mask_text = f" {' '.join(['[MASK]'] * n_mask)} "
      phrase_masked_list[i].append(text[:start] + mask_text + text[end:])

  return phrase_masked_list


# return units masked with UNK at each position in the sequence
def get_unk_masked(text, phrase_offsets, filtered_indices):
  masked_units = []
  for i in filtered_indices:
    start, end = phrase_offsets[i]
    masked_units.append(text[:start] + '[UNK]' + text[end:])
  # list of masked basic units
  return masked_units


def get_important_scores(
    masked_phrases,
    tokenizer,
    target_model,
    orig_label,
    max_prob,
    orig_probs,
    device,
    batch_size=1,
    max_length=512
):
  """compute importance scores based on the target model
  This function takes in the tokens from the original text, and the target model,
  and compute the difference with the original probs if each token is masked with [UNK].
  Args:
    text: the original text
    phrase_offsets: a list of tuples indicating the start and end of a phrase.
    filtered_indices: a list of indices 
    tokenizer: a BERT tokenizer to be used with the target model.
    target_model: a fine-tuned BERT model for sentiment analysis.
    orig_label: the original label of the text.
    max_prob: the maximum probability from the original probability output.
    orig_probs: the set of original probability outputted from the target model.
    device: the device to move around the tensors and models.
    batch_size: the batch size of the input.
    max_length: the maximum length to keep in the original text.
  Returns:
    import_scores: a torch tensor with dim (len(masked_phrases),)
  """

  encoded = tokenizer(masked_phrases,
                      truncation=True,
                      padding='max_length',
                      max_length=max_length,
                      return_token_type_ids=False,
                      return_tensors="pt")

  inputs = torch.cat([encoded['input_ids'].unsqueeze(0), encoded['attention_mask'].unsqueeze(0)]).to(device)
  inputs = inputs.permute(1, 0, 2).unsqueeze(2)
  leave_1_logits = [target_model(*data).logits for data in inputs]

  # turn into tensor
  leave_1_logits = torch.cat(leave_1_logits, dim=0)
  leave_1_probs = torch.softmax(leave_1_logits, dim=-1) # dim: (len(masked_phrases), num_of_classes)
  leave_1_labels = torch.argmax(leave_1_probs, dim=-1) # dim: len(masked_phrases)

  import_scores = (max_prob
                   - leave_1_probs[:, orig_label] # how the probability of original label decreases
                   +
                   (leave_1_labels != orig_label).float() # new label not equal to original label
                   *
                   (leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_labels))
                   )           # probability of changed label

  return import_scores, leave_1_labels


def get_substitutes(top_k_ids, tokenizer, mlm_model, device):
  """get_substitutes find the set of substitution candidates using perplexity.
  Limitation: due to the lack of GPU memory, we set a threshold
  Args:
    top_k_ids: top k ids from the mlm model, tensor (1, n_masks, k)
    tokenizer: Bert Tokenizer
    mlm_model: mlm model
    device: where to transfer the data
  Returns:
    candidates_list: list of list of candidates ranked by perplexity
  """
  # all substitutes  list of list of token-id (all candidates)
  c_loss = nn.CrossEntropyLoss(reduction='none')

  # here we need to get permutation of top k ids
  # because we have no idea what combination fits the most

  # assuming first dimension is 1
  #top_k_ids = top_k_ids.squeeze()
  #  print(top_k_ids)
  # https://stackoverflow.com/questions/1208118
  meshgrid = [tensor.unsqueeze(0) for tensor in torch.meshgrid(*top_k_ids)]
  ids_comb = torch.cat(meshgrid).T.reshape(-1, len(top_k_ids)).unique(dim=-1) \
             if len(top_k_ids.shape) != 1 else top_k_ids.unsqueeze(0).T
  #  print(ids_comb)
  #  print(top_k_ids)
  #  print(ids_comb)

  # set a threshold
  # TODO: we should select combinations instead of this simple cut
  ids_comb = ids_comb[:24]

  # compute perplexity
  N, L = ids_comb.size()
  logits = mlm_model(ids_comb)[0]
  ppl = c_loss(logits.view(N*L, -1), ids_comb.view(-1))
  ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))

  # sort candidates
  sorted_indices = torch.argsort(ppl)
  sorted_token_ids_list = torch.index_select(ids_comb, 0, sorted_indices).tolist()
  tokens_list = [tokenizer.convert_ids_to_tokens(tokens) for tokens in sorted_token_ids_list]
  # necessary to remove subwords
  candidates_list = [[tokenizer.convert_tokens_to_string([token]) for token in tokens] for tokens in tokens_list]

  return candidates_list

def get_phrase_substitutes(input_ids, attention_mask, mask_token_index, stop_words, tokenizer, mlm_model, device, beam_width=10, K=6):
  # all substitutes  list of list of token-id (all candidates)
  c_loss = nn.CrossEntropyLoss(reduction='none')

  word_positions = len(mask_token_index)
  query_num = 0
    
  masked_logits = mlm_model(input_ids, attention_mask).logits
  query_num += len(input_ids)
  
  masked_logits = torch.index_select(masked_logits, 1, mask_token_index[0])
    
  # top_ids has a beam_width number of word combinations with smallest perplexities
  # the initial candidates are the beam_width number of words with the highest logits
  top_ids = torch.topk(masked_logits, K, dim=-1).indices[0]

  #_, sorted_ids = torch.sort(masked_logits[0,0], dim=-1, descending=True)
  #filtered_ids = get_filtered_k_phrases(sorted_ids, tokenizer, stop_words, K)

  #initialize candidates pool with the top k candidates at the first position
  candidate_ids = top_ids.T.to(device)
    
  for p in range(1, word_positions):
    new_inputs = input_ids.repeat(len(candidate_ids), 1)
    new_inputs[:, mask_token_index[:p]] = candidate_ids
    new_attention_mask = attention_mask.repeat(len(candidate_ids), 1)
    
    masked_logits = mlm_model(new_inputs, new_attention_mask).logits
    masked_logits = torch.index_select(masked_logits, 1, mask_token_index[p])
    query_num += len(new_inputs)
    
    top_ids = torch.topk(masked_logits, beam_width, dim=-1).indices
    
    repeated_cands = candidate_ids.unsqueeze(1).repeat(1, beam_width, 1).reshape(-1,p)
    repeated_new_cands = top_ids.squeeze().reshape(-1, 1)
    
    # cur_options = (beam_width, beam_width)
    cur_options = torch.cat((repeated_cands, repeated_new_cands), 1)
    
    N, L = cur_options.size()
    logits = mlm_model(cur_options)[0]
    query_num += len(cur_options)

    ppl = c_loss(logits.view(N*L, -1), cur_options.view(-1))
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))

    # the smaller the perplexity, the more coherent the sequence is
    sorted_indices = torch.argsort(ppl)[:K]
    candidate_ids = torch.index_select(cur_options, 0, sorted_indices)
    
  sorted_token_ids_list = candidate_ids.tolist()
  tokens_list = [tokenizer.convert_ids_to_tokens(tokens) for tokens in sorted_token_ids_list]
    
  # necessary step to remove subwords
  candidates_list = [[tokenizer.convert_tokens_to_string([token]) for token in tokens] for tokens in tokens_list]
    
    
  return candidates_list, query_num

def get_word_substitues(input_ids, attention_mask, mask_token_index, tokenizer, mlm_model, K=8, threshold=3.0):
  masked_logits = mlm_model(input_ids, attention_mask).logits
  masked_logits = torch.index_select(masked_logits, 1, mask_token_index)
  
  top_k_ids = torch.topk(masked_logits, K, dim=-1).indices[0]
  #print(masked_logits.shape)
  #print(top_k_ids.shape)
  #print(mask_token_index)
  substitute_scores = masked_logits[0,0][top_k_ids][0]
  substitute_ids = top_k_ids[0]
    
  words = []
  for (i, score) in zip(substitute_ids, substitute_scores):
    if threshold != 0 and score < threshold:
      break
    words.append([tokenizer._convert_id_to_token(int(i))])
            
  return words
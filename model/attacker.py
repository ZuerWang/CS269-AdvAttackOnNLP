"""
attacker
"""

from pprint import pprint

import numpy as np
import torch
import datasets

from model.tokenizer import filter_unwanted_phrases, phrase_is_wanted
from model.substitution import (
  get_important_scores,
  get_substitutes,
  get_word_substitues,
  get_phrase_substitutes,
  get_unk_masked,
  get_phrase_masked_list
)


class Attacker:
  def __init__(self,
               pre_tok,
               tokenizer,
               target_model,
               mlm_model,
               sentence_encoder,
               #word_embedding,
               device,
               k=8,
               beam_width=10,
               conf_thres=3.0,
               sent_semantic_thres=0.6,
               change_threshold=0.4):
    self.pre_tok          = pre_tok
    self.stop_words       = self.pre_tok.spacy_processor.Defaults.stop_words
    self.tokenizer        = tokenizer
    self.target_model     = target_model
    self.mlm_model        = mlm_model
    self.sent_encoder     = sentence_encoder
    #self.word_embedding   = word_embedding
    self.device           = device

    # select top k candidates
    self.k                = k
    self.beam_width       = beam_width

    # single-word confidence threshold
    self.conf_thres       = conf_thres

    # USE sentence semantic threshold
    self.sent_semantic_thres = sent_semantic_thres

    # percentage of words can be changed in a sequence
    self.change_threshold = change_threshold


  def attack(self, entry):
    # TODO: a problem with get_important_scores is that
    # it does not care the numeber of tokens.
    # since BERT can only accept 512 tokens at max,
    # if [UNK] is inserted at a place after 512th token,
    # the importance score is essentially invalid.

    # potential solution:
    # 1. modify the tokenize to pass
    #    each entry into BertTokenizer and convert it back
    #    to keep truncated text with at max 512 tokens.
    # 2. (currently used) simply skips if mask_token_index is empty

    entry['phrase_changes'] = 0
    entry['word_changes'] = 0
    entry['changes'] = []
    entry['pred_success'] = True
    entry['success'] = False
    entry['word_num'] = len(entry['words'])

    phrase_lengths = [n for n in entry['n_words_in_phrases'] if n > 1]
    entry['phrase_num'] = len(phrase_lengths)
    entry['phrase_len'] = sum(phrase_lengths)
    entry['query_num'] = 0
    entry['final_adv'] = None


    # 1. retrieve logits and label from the target model
    encoded = self.tokenizer(entry['text'],
                             padding=True,
                             truncation=True,
                             return_token_type_ids=False,
                             return_tensors="pt")
    input_ids = encoded['input_ids'].to(self.device)
    attention_mask = encoded['attention_mask'].to(self.device)
    orig_logits = self.target_model(input_ids, attention_mask).logits.squeeze()
    orig_probs  = torch.softmax(orig_logits, -1)
    orig_label = torch.argmax(orig_probs)
    max_prob = torch.max(orig_probs)

    if orig_label != entry['label']:
      entry['pred_success'] = False
      return entry

    # filter out stop_words, digits & symbol combination
    filtered_indices = filter_unwanted_phrases(self.stop_words, entry['phrases'])

    masked_phrases = get_unk_masked(entry['text'], entry['phrase_offsets'], filtered_indices)
    importance_scores, _ = get_important_scores(masked_phrases,
                                                self.tokenizer,
                                                self.target_model,
                                                orig_label,
                                                max_prob,
                                                orig_probs,
                                                self.device)
    entry['query_num'] += len(masked_phrases)

    # this is the index after the filter and
    # cannot only applied to importance scores and filtered_indices
    sorted_filtered_indices_np = torch.argsort(importance_scores, dim=-1, descending=True).data.cpu().numpy()
    importance_scores_np = importance_scores.data.cpu().numpy()
    # obtain correct indices that can be used to index the entry dict
    sorted_indices_np = np.array(filtered_indices)[sorted_filtered_indices_np]
    sorted_importance = importance_scores_np[sorted_filtered_indices_np]
    sorted_phrases = np.array(entry['phrases'])[sorted_indices_np]
    sorted_phrase_offsets = np.array(entry['phrase_offsets'])[sorted_indices_np]
    sorted_n_words_in_phrase = np.array(entry['n_words_in_phrases'])[sorted_indices_np]

    # up to this point,
    # sorted_phrases is a sorted numPy array containing the filtered phrases ranked by importance
    # sorted_n_words_in_phrase is a sorted numPy array containing the number of words in each filtered phrases ranked by importance
    # sorted_importance is a sorted PyTorch Tensor containing importance scores ranked by importance

    max_change_threshold = len(entry['phrases'])

    # record how many perturbations have been made
    phrase_changes = 0
    word_changes = 0
    changes = []

    text = entry['text']
    phrases = entry['phrases']
    phrase_offsets = entry['phrase_offsets']
    n_words_in_phrases = entry['n_words_in_phrases']

    for changed,i in enumerate(sorted_indices_np):
      # break when attack is successful or changes exceed threshold
      if (changed+1)/max_change_threshold > self.change_threshold:
        break

      phrase_masked_list = get_phrase_masked_list(text,
                                                  [phrase_offsets[i]],
                                                  [n_words_in_phrases[i]])[0]

      attack_results = []
      for j, masked_text in enumerate(phrase_masked_list):
        # 3. get masked token candidates from MLM

        encoded = self.tokenizer(masked_text,
                                 truncation=True,
                                 padding=True,
                                 return_token_type_ids=False,
                                 return_tensors='pt')

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        mask_token_index = torch.where(input_ids == self.tokenizer.mask_token_id)[-1]
        # skip if part or all of masks exceed max_length
        if len(mask_token_index) != j + 1:
          continue

        candidates_list = []
        if len(phrase_masked_list) == 1:
          input_ids[0, mask_token_index[0]] = self.tokenizer.convert_tokens_to_ids(phrases[i])
          #encoded = self.tokenizer(text,
          #                       truncation=True,
          #                       padding=True,
          #                       return_token_type_ids=False,
          #                       return_tensors='pt')
          #input_ids = encoded['input_ids'].to(self.device)
          #attention_mask = encoded['attention_mask'].to(self.device)
          candidates_list = get_word_substitues(input_ids, attention_mask, mask_token_index, self.tokenizer, self.mlm_model, K=self.k, threshold=self.conf_thres)
          entry['query_num'] += len(input_ids)
        elif len(phrase_masked_list) > 1:
          candidates_list, qn = get_phrase_substitutes(input_ids, attention_mask, mask_token_index, self.stop_words, self.tokenizer, self.mlm_model, self.device, beam_width=self.beam_width, K=self.k)
          entry['query_num'] += qn

        mask_text = f" {' '.join([self.tokenizer.mask_token] * (j+1))} "
        for candidates in candidates_list:
          perturbed_text = masked_text
          candidate = ' '.join(candidates)

          if phrases[i] == candidate:
            continue

          if '##' in candidate:
            continue

          if not phrase_is_wanted(self.stop_words, candidate):
            continue

          # replace the mask_text with candidate
          perturbed_text = perturbed_text.replace(mask_text, candidate, 1)

          # semantic check -> if the phrase changes too much
          #if len(candidates) > 1:
          #seq_embeddings = self.sent_encoder([candidate, phrases[i]])

          seq_embeddings = self.sent_encoder.encode([perturbed_text, entry['text']])
          #seq_embeddings = self.sent_encoder([perturbed_text, entry['text']])
          semantic_sim = np.dot(seq_embeddings[0], seq_embeddings[1]) / \
            (np.linalg.norm(seq_embeddings[0]) * np.linalg.norm(seq_embeddings[1]))

          if semantic_sim < self.sent_semantic_thres:
            continue

          importance_score, perturbed_label = get_important_scores([perturbed_text],
                                                                   self.tokenizer,
                                                                   self.target_model,
                                                                   orig_label,
                                                                   max_prob,
                                                                   orig_probs,
                                                                   self.device)
          importance_score = importance_score.squeeze()
          entry['query_num'] += 1

          perturbed_label = perturbed_label.squeeze()

          if perturbed_label != orig_label:
            attack_results = [(perturbed_label == orig_label, j, candidate, perturbed_text, importance_score)]
            entry['success'] = True
            if n_words_in_phrases[i] > 1:
              entry['phrase_changes'] = phrase_changes + 1
            entry['word_changes'] = word_changes + n_words_in_phrases[i]
            changes.append((phrases[i], candidate))
            entry['changes'] = changes 
            entry['final_adv'] = perturbed_text
            return entry

          attack_results.append((perturbed_label == orig_label, j, candidate, perturbed_text, importance_score))

      attack_results = sorted(attack_results, key=lambda x: x[-1], reverse=True)

      if len(attack_results) == 0:
        #print('no candidates for: ', phrases[i])
        continue

      # no matter what, changes plus 1
      if n_words_in_phrases[i] > 1:
        phrase_changes += 1
      word_changes += n_words_in_phrases[i]

      # attack the max confidence one when there's no success
      result = attack_results[0]

      text = result[3]

      n_words_in_phrases[i] = result[1] + 1

      # update perturbed token to phrases and phrase offsets
      length_diff = len(phrases[i]) - len(result[2])
      if length_diff != 0:
        new_offsets = phrase_offsets[:i]
        for change_i in range(i, len(phrases)):
          start = phrase_offsets[change_i][0]
          end = phrase_offsets[change_i][1] - length_diff
          # start not change for index position
          if change_i != i:
            start -= length_diff
          new_offsets.append([start, end])

        phrase_offsets = new_offsets


      changes.append((phrases[i], result[2]))
      phrases[i] = result[2]

      text = result[3]

    entry['success'] = False
    entry['phrase_changes'] = phrase_changes
    entry['word_changes'] = word_changes
    entry['changes'] = changes
    entry['final_adv'] = text

    return entry
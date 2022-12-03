import json

def evaluate(entries, num_pred_failures, eval_f_pth, params):

  attack_success = 0

  total_q = 0
  total_word_changes = 0
  total_words = 0
  total_phrase_changes = 0
  total_phrases = 0
  query_num = 0
  total_phrase_len = 0
  num_total_entry = len(entries)
  sent_sim = 0.0

  for entry in entries:
    if entry['success']:
      attack_success += 1

    total_word_changes += entry['word_changes']
    total_phrase_changes += entry['phrase_changes']

    total_words += entry['word_num']
    total_phrases += entry['phrase_num']

    query_num += entry['query_num']
    total_phrase_len += entry['phrase_len']

    if 'semantic_sim' in entry:
        sent_sim += entry['semantic_sim']

  word_change_rate = total_word_changes / total_words
  word_per_seq = total_word_changes / len(entries)

  if total_phrases == 0:
    phrase_change_rate = 0
    mean_phrase_len = 0
  else:
    phrase_change_rate = total_phrase_changes / total_phrases
    mean_phrase_len = total_phrase_len / total_phrases

  phrase_per_seq = total_phrase_changes / len(entries)

  original_acc = 1.0 - num_pred_failures / num_total_entry
  after_atk_acc = 1.0 - (num_pred_failures + attack_success) / num_total_entry

  success_rate = attack_success / (num_total_entry - num_pred_failures)

  query_per_attack = query_num / len(entries)

  

  sent_sim = sent_sim / (num_total_entry - num_pred_failures)

  print()
  print('acc/aft-atk-acc: {:.6f}/ {:.6f}, query-per-attack: {:.4f}, success-rate: {:.4f}'.format(original_acc, after_atk_acc, query_per_attack, success_rate))
  print('word-changed-per-attack: {:.4f}, phrase-changed-per-attack: {:.4f}'.format(word_per_seq, phrase_per_seq))
  print('word-changed-rate: {:.4f}, phrase-changed-rate: {:.4f}'.format(word_change_rate, phrase_change_rate))
  print('mean-phrase-length: {:.4f}'.format(mean_phrase_len))
  print()

  results= [params, {'original_acc': original_acc, 'after_atk_acc': after_atk_acc, 'query_per_attack': query_per_attack, 'success_rate': success_rate, 'word_per_seq': word_per_seq, 'phrase_per_seq': phrase_per_seq, 'word-changed-rate': word_change_rate, 'phrase-changed-rate': phrase_change_rate, 'attack_success': attack_success, 'total_q': total_q, 'total_word_changes': total_word_changes, 'total_words': total_words, 'total_phrase_changes': total_phrase_changes, 'total_phrases': total_phrases, 'query_num': query_num, 'total_phrase_len': total_phrase_len, 'num_total_entry': num_total_entry, 'sentence_semantic_similarity': sent_sim }]

  json.dump(results, open(eval_f_pth, "w"), indent=2)
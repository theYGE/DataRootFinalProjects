import torch
from pytorch_transformers import *
from pytorch_transformers.tokenization_bert import BertTokenizer
from random import random, randrange, randint, shuffle, choice
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import torch
from pytorch_transformers import *
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import torch
import tokenize_uk

from trained_model import device, trained_tokenizer, trained_model
from basic_model import basic_model, basic_tokenizer

def mask_token_in_sentence(sentence):
  tokenized_sentence = tokenize_uk.tokenize_words(sentence)
  number_of_tokens = len(tokenized_sentence)
  masked_token = False
  while (not masked_token):
    # print(number_of_tokens)
    index = randrange(number_of_tokens)
    # To eliminate tokenization of punctioation like .,;:
    delims = {", ", ".", "!", ":", "?", "'", ";", ''}
    if (not tokenized_sentence[index] in delims):
      tokenized_sentence[index] = "[MASK]"
      masked_token = True
  # print(tokenized_sentence)
  reconstructed_sentence = " ".join(tokenized_sentence)
  return reconstructed_sentence

def get_unique_delimiters(texts):
  delimiters = set()
  for text in texts:
    for word in tokenize_uk.tokenize_words(text):
      if (len(word) == 1 and not word in delimiters and not word.isalpha() and not word.isdigit()):
        delimiters.add(word)
  return delimiters


def clean_texts(texts):
    dashes = {'–', '—', '―', '~'}  # replace with -
    special_symbols = {'№', '_', '<', '>', '|', ']', '*', '[', '^', '&'}  # replace with ""
    apostrophes = {'’', '‘'}  # replace with '
    direct_speech = {'“', '»', '«'}  # replace with '"'
    three_dots = {'…'}  # replace with '.

    counter = 0
    for i in range(len(texts)):
        print("Processing text: ", i)
        text = texts[i]
        words = []
        tokenized_words = tokenize_uk.tokenize_words(text)
        for word in tokenized_words:
            added = False

            for dash in dashes:
                if (dash in word):
                    new_word = word.replace(dash, "-")
                    words.append(new_word)
                    added = True
                    continue

            for special_symbol in special_symbols:
                if (special_symbol in word):
                    new_word = word.replace(special_symbol, "")
                    words.append(new_word)
                    added = True
                    continue

            for apostrophe in apostrophes:
                if (apostrophe in word):
                    new_word = word.replace(apostrophe, "'")
                    words.append(new_word)
                    added = True
                    continue

            for direct in direct_speech:
                if (direct in word):
                    new_word = word.replace(direct, '"')
                    words.append(new_word)
                    added = True
                    continue

            for dots in three_dots:
                if (dots in word):
                    counter += 1
                    new_word = word.replace(dots, '.')
                    words.append(new_word)
                    added = True
                    continue
            if (not added):
                words.append(word)
        reconstructed_text = " ".join(words)
        texts[i] = reconstructed_text
    return texts

def prepare_for_tokenization(texts):
  new_texts = []
  for i in range(len(texts)):
    text = texts[i]
    text = text.replace("?", "?.")
    text = text.replace("!", "!.")
    text = text.replace(":", ":.")
    text = text.replace(". -", ". ")
    new_texts.append(text)
  return new_texts

def tokenize_texts_to_sentences(texts):
  text_sentences = []
  for text in texts:
    text_to_add = []
    text_to_add += tokenize_uk.tokenize_sents(text)
    text_sentences.append(text_to_add)
  return text_sentences

def clean_after_tokenization(text_sentences):
  cleaned = []

  for j in range(len(text_sentences)):
    text_to_work = text_sentences[j]
    for i in range(len(text_to_work)):
      sentence = text_to_work[i]
      sentence = sentence.replace("?.", "?")
      sentence = sentence.replace("!.", "!")
      sentence = sentence.replace(":.", ":")
      text_to_work[i] = sentence
    cleaned.append(text_to_work)
  return cleaned

def mask_tokens(text_sentences):
  masked = []
  for j in range(len(text_sentences)):
    text_to_work = text_sentences[j]
    for i in range(len(text_to_work)):
      sentence = text_to_work[i]
      # print(sentence)
      sentence = mask_token_in_sentence(sentence)
      text_to_work[i] = sentence
    masked.append(text_to_work)
  return masked

def get_segments_and_tokens(text_input, given_tokenizer):
  text = text_input
  tokenized_text = []
  indexed_tokens = []
  segment_ids = []

  print("Number of sentences in text: ", len(text))
  longest_sentence = 0
  for sent in text:
    if (len(sent) > longest_sentence):
      longest_sentence = len(sent)
  print("Longest sentence: ", longest_sentence)

  for i in range(len(text)):
    text[i] = "[CLS] " + text[i] + " [SEP]"

  for i in range(len(text)):
    tokenized_text.append(given_tokenizer.tokenize(text[i]))
    # print(tokenized_text[i])
    indexed_tokens.append(given_tokenizer.convert_tokens_to_ids(tokenized_text[i]))

  indexed_tokens = pad_sequences(indexed_tokens, maxlen=longest_sentence, dtype="long", truncating="post", padding="post")
  print("First indexed tokens length: ", len(indexed_tokens[0]))

  for i in range(len(text)):
    segment = []
    num_tokens = len(indexed_tokens[i])
    segment_ids.append([0] * num_tokens)

  print("Segment ids length: ", len(segment_ids))
  print("First segment ids length: ", len(segment_ids[0]))

  return segment_ids, indexed_tokens

def make_inference(predictions, inference):
  # predictions = masked_lm_logits_scores

  # inference = indexed_tokens
  new_inference = []
  for i in range(len(inference)):
    # print(i)
    tokens = inference[i]
    # print(tokens)
    for j in range(len(tokens)):
      if (tokens[j] == 103):
        # print(103)
        token_to_insert = torch.argmax(predictions[i][j]).item()
        tokens[j] = token_to_insert
    # inference[i] = tokens
    new_inference.append(tokens)
  return new_inference

def convert_inference_to_sentences(inference, given_tokenizer):
  predicted_sentences = []
  for x in range(len(inference)):
    first_sentence = given_tokenizer.convert_ids_to_tokens(inference[x])
    first_sentence = given_tokenizer.convert_tokens_to_string(first_sentence)
    first_sequence = first_sentence.replace("[PAD]", "")
    first_sequence = first_sequence.replace("[CLS] ", "")
    first_sequence = first_sequence.replace(" [SEP]", "")
    predicted_sentences.append(first_sequence)
  return predicted_sentences

def generate_results(text_sentences, given_model, given_tokenizer):
    predicted_texts = []

    for i in range(len(text_sentences)):
        print("Processing text: ", i)
        text = text_sentences[i]
        segment_ids, indexed_tokens = get_segments_and_tokens(text, given_tokenizer)
        print("Recieved segments and indexed_tokens")

        segments_tensor = (torch.tensor(segment_ids))
        tokens_tensor = torch.tensor(indexed_tokens)

        print("Model started")

        input_length = len(segments_tensor)
        print(input_length)

        logits = []
        text = []
        for j in range(0, input_length, 10):

            left = j
            right = j + 10
            if (right > input_length):
                right = input_length

            if (left == input_length - 1):
                right = input_length

            segments_portion = segment_ids[left:right]
            tokens_portion = indexed_tokens[left:right]
            segments_tensor = (torch.tensor(segments_portion))
            tokens_tensor = torch.tensor(tokens_portion)

            tokens_tensor = tokens_tensor.to(device)
            segments_tensor = segments_tensor.to(device)

            print("Processing from {} to {}".format(left, right))
            with torch.no_grad():
                masked_lm_logits_scores, _ = given_model(tokens_tensor, segments_tensor)
                # logits.append(masked_lm_logits_scores)

            # logits = np.stack(logits, axis = 0)
            print("Model finished")

            predictions = masked_lm_logits_scores
            inference = indexed_tokens[left:right]

            new_inference = make_inference(predictions, inference)
            print("Recieved new inference")

            predicted_sentences = convert_inference_to_sentences(new_inference, given_tokenizer)
            print("Recieved predicted_sentences")

            text += predicted_sentences

        predicted_texts.append(text)
        print("Appended new sentences for: ", i)
        print("\n")

    saved_predicted = predicted_texts

    cleaned = []
    for j in range(len(saved_predicted)):
        text_to_work = saved_predicted[j]
        for i in range(len(text_to_work)):
            sentence = text_to_work[i]
            sentence = sentence.replace(" ?", "?")
            sentence = sentence.replace(" !", "!")
            sentence = sentence.replace(" :", ":")
            sentence = sentence.replace("  ", "")
            sentence = sentence.replace(" ,", ",")
            sentence = sentence.replace(" .", ".")
            text_to_work[i] = sentence
        cleaned.append(text_to_work)

    cleaned_texts = []
    for clean in cleaned:
        cleaned_texts.append(" ".join(clean))

    return cleaned_texts

def get_masked_texts(first_text, second_text, third_text):
    texts = [first_text, second_text, third_text]

    unique_delimiters = get_unique_delimiters(texts)
    print(unique_delimiters)
    print(len(unique_delimiters))

    texts = clean_texts(texts)

    new_texts = prepare_for_tokenization(texts)
    print("Number of texts: ", len(new_texts))

    text_sentences = tokenize_texts_to_sentences(new_texts)
    print("Number of tokenized texts: ", len(text_sentences))

    for i in range(len(text_sentences)):
        print(i, " contains ", len(text_sentences[i]), " sentences")

    print(new_texts[0])

    text_sentences = clean_after_tokenization(text_sentences)

    for i in range(len(text_sentences)):
        print(i, " contains ", len(text_sentences[i]), " sentences")

    text_sentences = mask_tokens(text_sentences)

    for i in range(len(text_sentences)):
        print(i, " contains ", len(text_sentences[i]), " sentences")

    print("Number of texts: ", len(text_sentences))

    for sent in text_sentences[0]:
        print(sent)

    cleaned_trained = generate_results(text_sentences, trained_model, trained_tokenizer)

    for j in range(len(text_sentences)):
        current = text_sentences[j]
        for i in range(len(current)):
          sentence = current[i]
          sentence = sentence.replace("[CLS] ", "")
          sentence = sentence.replace("[SEP] ", "")
          current[i] = sentence
        text_sentences[j] = current

    cleaned_basic = generate_results(text_sentences, basic_model, basic_tokenizer)

    results_dict = {
        "basic": cleaned_basic,
        "trained": cleaned_trained
    }

    return results_dict

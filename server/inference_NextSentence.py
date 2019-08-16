# from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification, BertForNextSentencePrediction, BertForMaskedLM, BertForPreTraining
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
import os

from trained_model import trained_tokenizer, trained_model, device


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

def get_story_length(text_sentences):
  story_length = len(text_sentences[0])
  length_of_stories = len(text_sentences[1]) + len(text_sentences[2])
  if (story_length > length_of_stories):
    story_length = length_of_stories
  return story_length

def get_sentences_for_consideration(text_sentences):
  sentences_for_considerartion = text_sentences[0][1:]
  sentences_for_considerartion += text_sentences[1]
  sentences_for_considerartion += text_sentences[2]

  return sentences_for_considerartion


def get_segment_ids_and_prepared_text(sentences_for_considerartion, current_sentence):
    prepared_text = []
    segment_ids = []
    for j in range(len(sentences_for_considerartion)):
        first_sentence = "[CLS] " + current_sentence + " [SEP] "
        second_sentence = sentences_for_considerartion[j] + " [SEP] "
        segment_ids.append([0] * len(first_sentence) + [1] * len(second_sentence))
        prepared_text.append(first_sentence + second_sentence)

    return segment_ids, prepared_text

def get_longest_sequence(prepared_text):
  longest_sentence = 0
  for sent in prepared_text:
      if (len(sent) > longest_sentence):
        longest_sentence = len(sent)
  return longest_sentence


def get_indexed_tokens_and_segment_ids(prepared_text, segment_ids, longest_sentence):
    indexed_tokens = []
    segment_ids = segment_ids
    tokenized_text = []
    for j in range(len(prepared_text)):
        tokenized_text.append(trained_tokenizer.tokenize(prepared_text[j]))
        indexed_tokens.append(trained_tokenizer.convert_tokens_to_ids(tokenized_text[j]))

    indexed_tokens = pad_sequences(indexed_tokens, maxlen=longest_sentence, dtype="long", truncating="post",
                                   padding="post")
    segment_ids = pad_sequences(segment_ids, maxlen=longest_sentence, dtype="long", truncating="post", padding="post")

    return indexed_tokens, segment_ids, tokenized_text


def get_index_for_adding_sentence(logits):
    real_logits = []
    for logit in logits:
        real_logits.append([logit[0], logit[1]])

    real_logits = np.asarray(real_logits)

    indices = []
    logits_list = []
    for i in range(real_logits.shape[0]):
        if (np.argmax(real_logits[i]) == 0):
            indices.append(i)
            logits_list.append(real_logits[i][0])  # first trial showed that it's better of without abs function!

    if (len(logits_list) == 0):
        index = np.argmax(logits, axis=0)[0]
    else:
        index = indices[np.argmax(logits_list)]
    return index

def get_generated_story(cleaned):
    texts = [cleaned[0], cleaned[1], cleaned[2]]

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

    story = [text_sentences[0][0]]
    current_sentence_index = 0

    story_length = get_story_length(text_sentences)
    print("Story length: ", story_length)
    sentences_for_considerartion = get_sentences_for_consideration(text_sentences)
    print("Number of sentences for consideration: ", len(sentences_for_considerartion))

    # story_length = 3 # ARTIFICIAL
    for i in range(current_sentence_index, story_length - 1):

        tokenized_text = []
        indexed_tokens = []
        prepared_text = []
        segment_ids = []

        print("Current sentence index: ", current_sentence_index)
        segment_ids, prepared_text = get_segment_ids_and_prepared_text(sentences_for_considerartion, story[current_sentence_index])

        longest_sentence = get_longest_sequence(prepared_text)
        print("Longest sentence is: ", longest_sentence)

        indexed_tokens, segment_ids, tokenized_text = get_indexed_tokens_and_segment_ids(prepared_text, segment_ids, longest_sentence)

        # print(len(indexed_tokens), len(segment_ids))

        input_length = len(segment_ids)
        logits = []
        step = 10

        # input_length = 15 # ARTIFICIAL

        for j in range(0, input_length, step):

            left = j
            right = j + step

            if (right > input_length):
                right = input_length

            if (left == input_length - 1):
                right = input_length

            segments_portion = segment_ids[left:right]
            tokens_portion = indexed_tokens[left:right]
            segments_tensor = (torch.tensor(segments_portion))
            tokens_tensor = torch.tensor(tokens_portion)

            segments_tensor = segments_tensor.to(device)
            tokens_tensor = tokens_tensor.to(device)

            print("Processing from {} to {}".format(left, right))
            with torch.no_grad():
                _, seq_relationship_logits = trained_model(tokens_tensor, segments_tensor)
                logits += list(seq_relationship_logits.cpu().numpy())

        index = get_index_for_adding_sentence(logits)
        print(index)
        print("Adding sentence: ", sentences_for_considerartion[index])
        story.append(sentences_for_considerartion[index])

        del sentences_for_considerartion[index]

        print("Number of sentences for consideration: ", len(sentences_for_considerartion))

        current_sentence_index += 1

    cleaned = []
    for j in range(len(story)):
        sentence = story[j]
        sentence = sentence.replace(" ?", "?")
        sentence = sentence.replace(" !", "!")
        sentence = sentence.replace(" :", ":")
        sentence = sentence.replace("  ", "")
        sentence = sentence.replace(" ,", ",")
        sentence = sentence.replace(" .", ".")
        cleaned.append(sentence)
    # cleaned.append(text_to_work)
    story = " ".join(cleaned)

    return story

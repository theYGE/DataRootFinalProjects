import torch
from pytorch_transformers import *
from pytorch_transformers.tokenization_bert import BertTokenizer
import wget
import os
import os

# directory = "/home/ubuntu/flaskproject/"
directory = os.getcwd() + "/"

global basic_tokenizer
basic_tokenizer_path = directory + "basic_model"
basic_tokenizer = BertTokenizer.from_pretrained(basic_tokenizer_path, do_lower_case = False)

global basic_model
basic_model_folder_path = directory + "basic_model"
basic_model = BertForPreTraining.from_pretrained(basic_model_folder_path)
basic_model.eval()
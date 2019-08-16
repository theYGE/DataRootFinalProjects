import torch
from pytorch_transformers import *
from pytorch_transformers.tokenization_bert import BertTokenizer
import wget
import os
import os

# directory = "/home/ubuntu/flaskproject/"
directory = os.getcwd() + "/"

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

global trained_tokenizer
tokenizer_path = directory + "trained_model"
trained_tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=False)

global trained_model
model_folder_path = directory + "trained_model"
trained_model = BertForPreTraining.from_pretrained(model_folder_path)
trained_model.eval()



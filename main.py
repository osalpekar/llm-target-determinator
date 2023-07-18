import torch
import torch.nn as nn
import torch.optim as optim

from transformers import BertTokenizer, BertModel, BertConfig


text1 = "Replace me by any text you'd like."
text2 = "Hello me by any text you'd like."

class Indexer:
    def __init__(self):
        config = config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.model = BertModel.from_pretrained("bert-base-uncased", config=config)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tests_index = None

    def encode_batch(self, batched_input):
        tokenized_input = self.tokenizer(batched_input, return_tensors='pt', padding=True)
        # print(tokenized_input)
        # print(self.tokenizer.decode(tokenized_input['input_ids'][0]))
        out = self.model(**tokenized_input)
        return out

indexer = Indexer()
output = indexer.encode_batch([text1, text2])
print(output[2][-1].shape)
print(output[2][-2].shape)
print(output[2][-3].shape)
print(output[1].shape) # (batch_size x 768)

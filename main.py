import torch
import torch.nn as nn
import torch.optim as optim

from transformers import BertTokenizer, BertModel, BertConfig


class Indexer:
    def __init__(self):
        config = config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.model = BertModel.from_pretrained("bert-base-uncased", config=config)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.test_index = None

    def encode_batch(self, batched_input):
        tokenized_input = self.tokenizer(batched_input, return_tensors='pt', padding=True)
        # print(tokenized_input)
        # print(self.tokenizer.decode(tokenized_input['input_ids'][0]))
        out = self.model(**tokenized_input)
        return out
    
    def index(self, all_data):
        embedding_list = []

        for idx, batch in enumerate(all_data):
            embeddings = self.encode_batch(batch)[1]
            embedding_list.append(embeddings)

        self.test_index = torch.cat(embedding_list, dim=0)



## Tests Code

text1 = "Replace me by any text you'd like."
text2 = "Hello me by any text you'd like."
text3 = "any text you'd like."
text4 = "Hello me you'd like."

indexer = Indexer()
output = indexer.encode_batch([text1, text2])

print(output[2][-1].shape)
print(output[1].shape) # (batch_size x 768)

all_data = [[text1, text2], [text3, text4]]
indexer.index(all_data)
print(indexer.test_index.shape)

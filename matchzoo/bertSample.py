import torch
import os
import numpy as np
import pandas as pd
import matchzoo as mz
from sklearn.model_selection import train_test_split
# from transformers import BertConfig, BertModel, BertTokenizer
print('matchzoo version', mz.__version__)

"""Not use cuda to debug"""
# export CUDA_VISIBLE_DEVICES=""

classification_task = mz.tasks.Classification(num_classes=2)
classification_task.metrics = ['acc']

train_pack_raw = mz.datasets.wiki_qa.load_data('train', task=classification_task)
dev_pack_raw = mz.datasets.wiki_qa.load_data('dev', task=classification_task)
test_pack_raw = mz.datasets.wiki_qa.load_data('test', task=classification_task)

""" MODEL & Train """
biobert = '/home/sjy1203/Shane/BIOBERT_DIR/'
preprocessor = mz.models.Bert.get_default_preprocessor(
    mode=biobert
    # truncated_length_left=30,
    # truncated_length_right=30,
    # ngram_size=1
)
train_processed = preprocessor.transform(train_pack_raw)
valid_processed = preprocessor.transform(test_pack_raw)

# print(preprocessor.context)

""" TODO : Initialize embedding with Biobert """
# glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=100)
# term_index = preprocessor.context['vocab_unit'].state['term_index']
# embedding_matrix = glove_embedding.build_matrix(term_index)
# l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
# embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]

# ngram_callback = mz.dataloader.callbacks.Ngram(preprocessor, mode='index')

trainset = mz.dataloader.Dataset(
    data_pack=train_processed,
    mode='point',
    batch_size=4,
    # num_dup=1,
    # num_neg=4,
    # resample=True
    # callbacks=[ngram_callback]
)
validset = mz.dataloader.Dataset(
    data_pack=valid_processed,
    mode='point',
    batch_size=4,
    # resample=False
    # callbacks=[ngram_callback]
)


padding_callback = mz.models.Bert.get_default_padding_callback(
    # fixed_length_left=365,
    # fixed_length_right=30, 
    # fixed_ngram_length=15
)

trainloader = mz.dataloader.DataLoader(
    dataset=trainset,
    stage='train',
    callback=padding_callback
)
validloader = mz.dataloader.DataLoader(
    dataset=validset,
    stage='dev',
    callback=padding_callback
)


model = mz.models.Bert()
model.params['task'] = classification_task
model.params['mode'] = biobert#
#model.params['mode'] = 'bert-base-uncased'

# model.params['embedding'] = embedding_matrix
# model.params['mask_value'] = 0
# model.params['hidden_size'] = 200
# model.params['lstm_layer'] = 1
# model.params['embedding_output_dim'] = 100
# model.params['embedding_input_dim'] = preprocessor.context['embedding_input_dim']
# model.params['char_embedding_input_dim'] = preprocessor.context['ngram_vocab_size']
model.params['dropout_rate'] = 0.2
model.guess_and_fill_missing_params()
model.build()

# print(model)
print('Trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

# optimizer = torch.optim.Adam(model.parameters())
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 5e-5},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

from pytorch_transformers import AdamW, WarmupLinearSchedule

optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, betas=(0.9, 0.98), eps=1e-8)
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=6, t_total=-1)

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    trainloader=trainloader,
    validloader=validloader,
    epochs=10
)

trainer.run()

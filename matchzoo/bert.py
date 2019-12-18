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

classification_task = mz.tasks.Classification(num_classes=3)
classification_task.metrics = ['acc']

"""Annotation file"""
annot = pd.read_csv('../annotations/annotations_merged.csv')
# print(annot.dtypes)
# annot.sort_values('PMCID').head(10)

"""Prompt file"""
prompt = pd.read_csv('../annotations/prompts_merged.csv')
# print(prompt.dtypes)
# prompt.sort_values('PMCID').head(10)

"""Format prompt for input"""
ICO_format = []
for triplet in zip(prompt['Intervention'].values, prompt['Comparator'].values, prompt['Outcome'].values):
    I, C, O = triplet[0], triplet[1], triplet[2]
    format_ico = 'With respect to ' + O + ', characterize the reported difference between patients receiving ' \
        + I + ' and those receiving ' + C
    ICO_format.append(format_ico)
prompt['Prompt'] = ICO_format
# print(prompt['Prompt'][0])

"""Process txt file for Articles input"""
TXT_PATH = '../annotations/txt_files/'
TAR_PATH = '../annotations/processed_txt_files/'
look_up = {}

if not os.path.exists(TAR_PATH):
    os.mkdir(TAR_PATH)
# else: os.removedirs(TAR_PATH)
for file in os.listdir(TXT_PATH):
    fname = file[3:]
    look_up[str(file[3:-4])] = fname
    with open(TXT_PATH+file, 'r', encoding='utf-8') as f, open(TAR_PATH+fname, 'w', encoding='utf-8') as t:
        for line in f.readlines():
            t.write(line.strip())

""" Read article by name """
def read_article(id):
    with open(TAR_PATH+look_up[id], 'r', encoding='utf-8') as f:
        # cut_article = ''
        # for w in f.readlines()[0].split()[:510]:
        #     cut_article = cut_article + w + ' '
        # return cut_article
        return f.readlines()[0][0:500]

text_left = [read_article(str(id)) for id in annot['PMCID']]
print(text_left[0])

df = pd.DataFrame(data={
    'id_left': annot['PMCID'],
    'text_left': text_left,
    'id_right': annot['PromptID'],
    'text_right': prompt['Prompt'],
    'label': annot['Label Code']+1
})

"""Split data pack into train/valid"""
train, valid = train_test_split(df, test_size=0.2)
train_pack = mz.pack(train, task=classification_task)
valid_pack = mz.pack(valid, task=classification_task)
train_pack.frame().head(10) # DataFrame

dp = mz.pack(df, task=classification_task)
# print(type(dp.frame))
frame_slice = dp.frame[0:5]
# print(list(frame_slice.columns))
full_frame = dp.frame()
assert len(full_frame) == len(dp)


""" MODEL & Train """
biobert = '/home/sjy1203/Shane/BIOBERT_DIR/'
preprocessor = mz.models.Bert.get_default_preprocessor(mode=biobert)
train_processed = preprocessor.transform(train_pack)
valid_processed = preprocessor.transform(valid_pack)

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
    batch_size=4
    # num_dup=1,
    # num_neg=4,
    # resample=True
    # callbacks=[ngram_callback]
)
validset = mz.dataloader.Dataset(
    data_pack=valid_processed,
    mode='point',
    batch_size=4
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
model.params['mode'] = biobert
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
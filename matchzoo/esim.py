import torch
import os
import numpy as np
import pandas as pd
import matchzoo as mz
from sklearn.model_selection import train_test_split
print('matchzoo version', mz.__version__)

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
def read_article(id: str) -> str:
    with open(TAR_PATH+look_up[id], 'r', encoding='utf-8') as f:
        return f.readlines()[0]

text_left = [read_article(str(id)) for id in annot['PMCID']]
# print(text_left[0:5])

df = pd.DataFrame(data={
    'id_left': annot['PMCID'],
    'text_left': text_left,
    'id_right': annot['PromptID'],
    'text_right': prompt['Prompt'],
    'label': annot['Label Code']+1
})

""" Split data pack into train/valid """
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
preprocessor = mz.models.ESIM.get_default_preprocessor(
    truncated_length_left=30,
    truncated_length_right=30,
    ngram_size=1
)
train_processed = preprocessor.fit_transform(train_pack)
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
    mode='pair',
    batch_size=16,
    num_dup=1,
    num_neg=4,
    resample=True
    # callbacks=[ngram_callback]
)
validset = mz.dataloader.Dataset(
    data_pack=valid_processed,
    mode='point',
    batch_size=16,
    resample=False
    # callbacks=[ngram_callback]
)

padding_callback = mz.models.ESIM.get_default_padding_callback(
    # fixed_length_left=30,
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

model = mz.models.ESIM()
model.params['task'] = classification_task
# model.params['embedding'] = embedding_matrix
model.params['mask_value'] = 0
model.params['hidden_size'] = 200
model.params['lstm_layer'] = 1
model.params['embedding_output_dim'] = 100
model.params['embedding_input_dim'] = preprocessor.context['embedding_input_dim']
# model.params['char_embedding_input_dim'] = preprocessor.context['ngram_vocab_size']
# model.params['dropout_rate'] = 0.2
model.guess_and_fill_missing_params()
model.build()

# print(model)
print('Trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = torch.optim.Adam(model.parameters())

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=validloader,
    epochs=10
)

trainer.run()

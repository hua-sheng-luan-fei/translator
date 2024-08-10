<h1><center>report in sentiment analysis



## 模型结构

### 模型简介

本项目采用预训练的BERT模型来进行中文到俄语的翻译任务。具体使用的模型为 `bert-base-multilingual-cased`，支持多种语言，包括中文和俄语。该模型旨在解决文本分类问题，并且通过微调的方法应用于翻译任务。

### 模型结构

- **输入层**：文本输入，通过预训练的BERT分词器进行分词和编码。
- **嵌入层**：BERT模型对输入进行编码，将词汇转化为高维向量。
- **Transformer层**：12层Transformer，每层包含多头自注意力机制和前馈神经网络。
- **输出层**：全连接层，将Transformer层的输出用于标签分类，本项目中设置为3个分类标签（正面、负面、中立）。

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = 'bert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

```



## 训练方法描述与收敛曲线

### 数据准备

我们从JSON文件中读取数据，并进行预处理。划分训练集和测试集，使用BERT分词器对文本进行编码。

```python
import json
from sklearn.model_selection import train_test_split
import torch

def get_data(filepath):
    sentimap = {"positive": 1, "negative": 0, "neutral": 2}
    with open(filepath, 'r', encoding='utf-8') as file:
        dic = json.load(file)
    texts, labels = [], []
    for item in dic:
        m = item['sentiment']
        if m not in sentimap:
            continue
        text = item['text'].replace('\n', '').replace('Ctrl+Enter', '')
        texts.append(text)
        labels.append(sentimap[m])
    return texts, labels

texts, labels = get_data(filepath)
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

tokenizer = AutoTokenizer.from_pretrained(model_name)
train_encodings = tokenizer(train_texts, padding=True, truncation=True, return_tensors='pt')
test_encodings = tokenizer(test_texts, padding=True, truncation=True, return_tensors='pt')

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, train_labels)
test_dataset = SentimentDataset(test_encodings, test_labels)

```



### 训练过程

使用Hugging Face的Trainer API进行训练，设置必要的训练参数，如训练周期数、批次大小、预热步骤等。

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./data/results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./data/logs',
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

```



### 收敛曲线

训练过程中记录损失值并绘制收敛曲线。通过这些曲线可以观察模型的训练效果和收敛速度。

![](E:\translator\sentiment.png)

## 模型效果

### 示例1

- **俄语（输入）：** Мне сегодня очень весело!（我今天玩得很开心！）

- **情感分析（输出）：** 

  ```
  Text: Мне сегодня очень весело!
  Sentiment: Score: 0.5730974674224854
  ```

### 示例2

- **俄语（输入）：** Сегодня хорошая погода.（今天天气很好。）

- **情感分析（输出）：** 

  ```
  Text: Сегодня хорошая погода!
  Sentiment: Score: 0.5570529103279114
  ```

### 示例3

- **俄语（输入）：** Я ненавижу этот фильм!（我讨厌这部电影！）

- **情感分析（输出）：** 

  ```
  Text: Я ненавижу этот фильм!
  Sentiment: Score: 0.8691123604774475
  ```

  
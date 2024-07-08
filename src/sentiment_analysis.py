import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
import json
from transformers import Trainer, TrainingArguments

filepath = './data/senti_train.json'

#获取数据
def get_data(filepath):
    sentimap = {"positive": 1, "negative": 0, "neutral": 2}
    file = open(filepath, 'r', encoding='utf-8')
    dic = json.load(file)
    texts, labels = [], []
    for i in range(len(dic)):
        m = dic[i]['sentiment']
        if(m not in sentimap.keys()):
            continue
        text = dic[i]['text']
        text = text.replace('\n', '')
        text = text.replace('Ctrl+Enter', '')
        texts.append(text)
        labels.append(sentimap[m])
        # print(sentimap[m])
    return texts, labels

texts, labels = get_data(filepath)


# 划分训练集和测试集
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 加载预训练的BERT模型和分词器
model_name = 'bert-base-multilingual-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# 数据预处理
train_encodings = tokenizer(train_texts, padding=True, truncation=True, return_tensors='pt')
test_encodings = tokenizer(test_texts, padding=True, truncation=True, return_tensors='pt')

# 准备PyTorch数据集
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

# 训练模型


training_args = TrainingArguments(
    output_dir='./data/results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./data/logs',
    warmup_steps=500,                # 预热步骤
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

# 评估模型
predictions = trainer.predict(test_dataset).predictions
predicted_labels = np.argmax(predictions, axis=1)

print("Accuracy: ", accuracy_score(test_labels, predicted_labels))
print(classification_report(test_labels, predicted_labels))

sentiment_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

sample_text = "Мне сегодня очень весело!"
result = sentiment_pipeline(sample_text)[0]
print(f"Text: {sample_text}")
print(f"Sentiment: {result['label']}, Score: {result['score']}")

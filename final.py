#coding=gbk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch

def translate_zh_to_ru(text):#实现中文到俄语的翻译

    # 定义模型和分词器的名称
    model_name = "joefox/mbart-large-ru-zh-ru-many-to-many-mmt"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    #翻译
    tokenizer.src_lang = "zh_CN"
    encoded_zh = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded_zh,
        forced_bos_token_id=tokenizer.lang_code_to_id["ru_RU"]
    )
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return result


def sentiment_analysis_ru(text): #对俄语进行情感分析

    model_name = "MonoHime/rubert-base-cased-sentiment-new"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # 模型推理
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # 计算概率
    probs = softmax(logits, dim=1)
    
    # 返回情感分析结果
    return probs

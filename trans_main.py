import os
import math

import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import pad, log_softmax
from pathlib import Path
from tqdm import tqdm
import torchtext
torchtext.disable_torchtext_deprecation_warning()

# 工作目录，缓存文件盒模型会放在该目录下
work_dir = Path("../data")
# 训练好的模型会放在该目录下
model_dir = Path("../transformer_checkpoints")
# 上次运行到的地方，如果是第一次运行，为None，如果中途暂停了，下次运行时，指定目前最新的模型即可。
model_checkpoint = None # 'model_10000.pt'

# 如果工作目录不存在，则创建一个
if not os.path.exists(work_dir):
    os.makedirs(work_dir)

# 如果工作目录不存在，则创建一个
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 俄语句子的文件路径
ru_filepath = '../data/train.ru'
# 中文句子的文件路径
zh_filepath = '../data/train.zh_CN'


# 定义一个获取文件行数的方法。
def get_row_count(filepath):
    count = 0
    for _ in open(filepath, encoding='utf-8'):
        count += 1
    return count


# 俄语句子数量
ru_row_count = get_row_count(ru_filepath)
# 中文句子数量
zh_row_count = get_row_count(zh_filepath)
assert ru_row_count == zh_row_count, "英文和中文文件行数不一致！"
# 句子数量，主要用于后面显示进度。
row_count = zh_row_count

# 定义句子最大长度，如果句子不够这个长度，则填充，若超出该长度，则裁剪
max_length = 72
# print("句子数量为：", ru_row_count)
# print("句子最大长度为：", max_length)

# 定义俄语和中文词典，都为Vocab类对象，后面会对其初始化
ru_vocab = None
zh_vocab = None

# 定义batch_size，由于是训练文本，占用内存较小，可以适当大一些
batch_size = 64
# epochs数量，不用太大，因为句子数量较多
epochs = 10
# 多少步保存一次模型，防止程序崩溃导致模型丢失。
save_after_step = 5000

# 是否使用缓存，由于文件较大，初始化动作较慢，所以将初始化好的文件持久化
use_cache = True

# 定义训练设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# print("batch_size:", batch_size)
# print("每{}步保存一次模型".format(save_after_step))
# print("Device:", device)
# 加载基础的分词器模型，使用的是基础的bert模型。`uncased`意思是不区分大小写
tokenizer = Tokenizer.from_pretrained("bert-base-multilingual-cased")

def ru_tokenizer(line):
    """
    定义俄语分词器，后续也要使用
    """
    # 使用bert进行分词，并获取tokens。add_special_tokens是指不要在结果中增加‘<bos>’和`<eos>`等特殊字符
    return tokenizer.encode(line, add_special_tokens=False).tokens

# text = "Привет, как дела?"
# print(ru_tokenizer(text))

def yield_ru_tokens():
    """
    每次yield一个分词后的俄语句子，之所以yield方式是为了节省内存。
    如果先分好词再构造词典，那么将会有大量文本驻留内存，造成内存溢出。
    """
    file = open(ru_filepath, encoding='utf-8')
    print("-------开始构建俄语词典-----------")
    for line in tqdm(file, desc="构建俄语词典", total=row_count):
        yield ru_tokenizer(line)
    file.close()

ru_vocab
# 指定俄语词典缓存文件路径
ru_vocab_file = work_dir / "vocab_ru.pt"
# 如果使用缓存，且缓存文件存在，则加载缓存文件
if use_cache and os.path.exists(ru_vocab_file):
    ru_vocab = torch.load(ru_vocab_file, map_location="cpu")
# 否则就从0开始构造词典
else:
    # 构造词典
    ru_vocab = build_vocab_from_iterator(
        # 传入一个可迭代的token列表。例如[['i', 'am', ...], ['machine', 'learning', ...], ...]
        yield_ru_tokens(),
        # 最小频率为2，即一个单词最少出现两次才会被收录到词典
        min_freq=2,
        # 在词典的最开始加上这些特殊token
        specials=["<s>", "</s>", "<pad>", "<unk>"],
    )
    # 设置词典的默认index，后面文本转index时，如果找不到，就会用该index填充
    ru_vocab.set_default_index(ru_vocab["<unk>"])
    # 保存缓存文件
    if use_cache:
        torch.save(ru_vocab, ru_vocab_file)

# print("俄语词典大小:", len(ru_vocab))
# print(dict((i, ru_vocab.lookup_token(i)) for i in range(10)))


def zh_tokenizer(line):
    """
    定义中文分词器
    :param line: 中文句子，例如：机器学习
    :return: 分词结果，例如['机','器','学','习']
    """
    return list(line.strip().replace(" ", ""))


def yield_zh_tokens():
    file = open(zh_filepath, encoding='utf-8')
    for line in tqdm(file, desc="构建中文词典", total=row_count):
        yield zh_tokenizer(line)
    file.close()

zh_vocab_file = work_dir / "vocab_zh.pt"
zh_vocab
if use_cache and os.path.exists(zh_vocab_file):
    zh_vocab = torch.load(zh_vocab_file, map_location="cpu")
else:
    zh_vocab = build_vocab_from_iterator(
        yield_zh_tokens(),
        min_freq=1,
        specials=["<s>", "</s>", "<pad>", "<unk>"],
    )
    zh_vocab.set_default_index(zh_vocab["<unk>"])
    torch.save(zh_vocab, zh_vocab_file)

# # 打印看一下效果
# print("中文词典大小:", len(zh_vocab))
# print(dict((i, zh_vocab.lookup_token(i)) for i in range(100)))



class TranslationDataset(Dataset):

    def __init__(self):
        # 加载中文tokens
        self.zh_tokens = self.load_tokens(zh_filepath, zh_tokenizer, zh_vocab, "构建中文tokens", 'zh')
        # 加载俄语tokens
        self.ru_tokens = self.load_tokens(ru_filepath, ru_tokenizer, ru_vocab, "构建俄语tokens", 'ru')

    def __getitem__(self, index):
        return self.zh_tokens[index], self.ru_tokens[index]

    def __len__(self):
        return row_count

    def load_tokens(self, file, tokenizer, vocab, desc, lang):
        """
        加载tokens，即将文本句子们转换成index们。
        :param file: 文件路径，例如"./dataset/train.en"
        :param tokenizer: 分词器，例如en_tokenizer函数
        :param vocab: 词典, Vocab类对象。例如 en_vocab
        :param desc: 用于进度显示的描述，例如：构建英文tokens
        :param lang: 语言。用于构造缓存文件时进行区分。例如：’en‘
        :return: 返回构造好的tokens。例如：[[6, 8, 93, 12, ..], [62, 891, ...], ...]
        """

        # 定义缓存文件存储路径
        cache_file = work_dir / "tokens_list.{}.pt".format(lang)
        # 如果使用缓存，且缓存文件存在，则直接加载
        if use_cache and os.path.exists(cache_file):
            print(f"正在加载缓存文件{cache_file}, 请稍后...")
            return torch.load(cache_file, map_location="cpu")

        # 从0开始构建，定义tokens_list用于存储结果
        tokens_list = []
        # 打开文件
        with open(file, encoding='utf-8') as file:
            # 逐行读取
            for line in tqdm(file, desc=desc, total=row_count):
                # 进行分词
                tokens = tokenizer(line)
                # 将文本分词结果通过词典转成index
                tokens = vocab(tokens)
                # append到结果中
                tokens_list.append(tokens)
        # 保存缓存文件
        if use_cache:
            torch.save(tokens_list, cache_file)

        return tokens_list

dataset = TranslationDataset()
# print(dataset.__getitem__(0))

def collate_fn(batch):
    """
    将dataset的数据进一步处理，并组成一个batch。
    :param batch: 一个batch的数据，例如：
                  [([6, 8, 93, 12, ..], [62, 891, ...]),
                  ....
                  ...]
    :return: 填充后的且等长的数据，包括src, tgt, tgt_y, n_tokens
             其中src为原句子，即要被翻译的句子
             tgt为目标句子：翻译后的句子，但不包含最后一个token
             tgt_y为label：翻译后的句子，但不包含第一个token，即<bos>
             n_tokens：tgt_y中的token数，<pad>不计算在内。
    """

    # 定义'<bos>'的index，在词典中为0，所以这里也是0
    bs_id = torch.tensor([0])
    # 定义'<eos>'的index
    eos_id = torch.tensor([1])
    # 定义<pad>的index
    pad_id = 2

    # 用于存储处理后的src和tgt
    src_list, tgt_list = [], []

    # 循环遍历句子对儿
    for (_src, _tgt) in batch:
        """
        _src: 中文句子
        _tgt: 俄语句子
        """

        processed_src = torch.cat(
            # 将<bos>，句子index和<eos>拼到一块
            [
                bs_id,
                torch.tensor(
                    _src,
                    dtype=torch.int64,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    _tgt,
                    dtype=torch.int64,
                ),
                eos_id,
            ],
            0,
        )

        """
        将长度不足的句子进行填充到max_padding的长度的，然后增添到list中

        pad：假设processed_src为[0, 1136, 2468, 1349, 1]
             第二个参数为: (0, 72-5)
             第三个参数为：2
        则pad的意思表示，给processed_src左边填充0个2，右边填充67个2。
        最终结果为：[0, 1136, 2468, 1349, 1, 2, 2, 2, ..., 2]
        """
        src_list.append(
            pad(
                processed_src,
                (0, max_length - len(processed_src),),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_length - len(processed_tgt),),
                value=pad_id,
            )
        )

    # 将多个src句子堆叠到一起
    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)

    # tgt_y是目标句子去掉第一个token，即去掉<bos>
    tgt_y = tgt[:, 1:]
    # tgt是目标句子去掉最后一个token
    tgt = tgt[:, :-1]

    # 计算本次batch要预测的token数
    n_tokens = (tgt_y != 2).sum()

    # 返回batch后的结果
    return src, tgt, tgt_y, n_tokens


train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
src, tgt, tgt_y, n_tokens = next(iter(train_loader))
src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)

# print("src.size:", src.size())
# print("tgt.size:", tgt.size())
# print("tgt_y.size:", tgt_y.size())
# print("n_tokens:", n_tokens)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model).to(device)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-(math.log(10000.0) / d_model))
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class TranslationModel(nn.Module):

    def __init__(self, d_model, src_vocab, tgt_vocab, dropout=0.1):
        super(TranslationModel, self).__init__()

        # 定义原句子的embedding
        # print(len(src_vocab))
        self.src_embedding = nn.Embedding(len(src_vocab), d_model, padding_idx=2)
        # 定义目标句子的embedding
        self.tgt_embedding = nn.Embedding(len(tgt_vocab), d_model, padding_idx=2)
        # 定义posintional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=max_length)
        # 定义Transformer
        self.transformer = nn.Transformer(d_model, dropout=dropout, batch_first=True)

        # 定义最后的预测层，这里并没有定义Softmax，而是把他放在了模型外。
        self.predictor = nn.Linear(d_model, len(tgt_vocab))

    def forward(self, src, tgt):
        """
        进行前向传递，输出为Decoder的输出。注意，这里并没有使用self.predictor进行预测，
        因为训练和推理行为不太一样，所以放在了模型外面。
        :param src: 原batch后的句子，例如[[0, 12, 34, .., 1, 2, 2, ...], ...]
        :param tgt: 目标batch后的句子，例如[[0, 74, 56, .., 1, 2, 2, ...], ...]
        :return: Transformer的输出，或者说是TransformerDecoder的输出。
        """

        """
        生成tgt_mask，即阶梯型的mask，例如：
        [[0., -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0.]]
        tgt.size()[-1]为目标句子的长度。
        """
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1]).to(device)
        # 掩盖住原句子中<pad>的部分，例如[[False,False,False,..., True,True,...], ...]
        src_key_padding_mask = TranslationModel.get_key_padding_mask(src)
        # 掩盖住目标句子中<pad>的部分
        tgt_key_padding_mask = TranslationModel.get_key_padding_mask(tgt)

        # 对src和tgt进行编码
        # print(f"Max index in src: {src.max().item()}, Min index in src: {src.min().item()}")
        src = self.src_embedding(src)
        # print(f"Max index in tgt: {tgt.max().item()}, Min index in tgt: {tgt.min().item()}")
        tgt = self.tgt_embedding(tgt)
        # 给src和tgt的token增加位置信息
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # 将准备好的数据送给transformer
        out = self.transformer(src, tgt,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)

        """
        这里直接返回transformer的结果。因为训练和推理时的行为不一样，
        所以在该模型外再进行线性层的预测。
        """
        return out

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        return tokens == 2



model = TranslationModel(512, zh_vocab, ru_vocab).to(device)
# output = model(src, tgt)
# print(output)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


class TranslationLoss(nn.Module):

    def __init__(self):
        super(TranslationLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = 2

    def forward(self, x, target):
        """
        损失函数的前向传递
        :param x: 将Decoder的输出再经过predictor线性层之后的输出。
                  也就是Linear后、Softmax前的状态
        :param target: tgt_y。也就是label，例如[[1, 34, 15, ...], ...]
        :return: loss
        """

        """
        由于KLDivLoss的input需要对softmax做log，所以使用log_softmax。
        等价于：log(softmax(x))
        """
        x = log_softmax(x, dim=-1)

        """
        构造Label的分布，也就是将[[1, 34, 15, ...]] 转化为:
        [[[0, 1, 0, ..., 0],
          [0, ..., 1, ..,0],
          ...]],
        ...]
        """
        # 首先按照x的Shape构造出一个全是0的Tensor
        true_dist = torch.zeros(x.size()).to(device)
        # 将对应index的部分填充为1
        true_dist.scatter_(1, target.data.unsqueeze(1), 1)
        # 找出<pad>部分，对于<pad>标签，全部填充为0，没有1，避免其参与损失计算。
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)

        # 计算损失
        return self.criterion(x, true_dist.clone().detach())

criteria = TranslationLoss()
writer = SummaryWriter(log_dir='../runs/transformer_loss')




from transformers import BertTokenizer, BertModel

# 1. 定义要下载的模型名称 + 本地保存路径
model_name = "bert-base-chinese"
local_save_dir = "./"  # 你想保存到的本地文件夹

# 2. 下载tokenizer并保存到本地
tokenizer = BertTokenizer.from_pretrained(
    model_name,
    cache_dir=local_save_dir  # 强制下载到这个目录
)

# 3. 下载模型权重并保存到本地
model = BertModel.from_pretrained(
    model_name,
    cache_dir=local_save_dir
)

# 下载完成后，local_save_dir下会出现模型文件（vocab.txt、config.json、pytorch_model.bin等）
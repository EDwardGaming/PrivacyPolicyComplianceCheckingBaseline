# 设置lstm训练参数
class TrainingConfig(object):
    batch_size = 64  
    lstm_batch_size = 512
    # 学习速率
    lr = 0.004
    epoches = 10  # 轮数
    print_step = 1
    bert_epoches = 3  # BERT训练的轮数
    bert_lr = 2e-5  # BERT的学习率

class LSTMConfig(object):
    emb_size = 128  # 词向量的维数
    hidden_size = 256  # lstm隐向量的维数
    dropout = 0.5

class BERTConfig:
    max_len = 256  # BERT输入的最大长度
    dropout = 0.3  # Dropout的比例

class CNNConfig(object):
    kernel_size = 5
    num_filters = 256


# ==================== 配置参数 ====================
class gptConfig:    
    LLM_API_KEY = "sk-lDM44EPTt9vFx4DX24C597A5CdC346D8Ba006d1f3d4bAf10"  
    LLM_BASE_URL = "https://kfcv50.link/v1" 
    LLM_MODEL_ID = "gpt-5"
    
    # 批处理配置
    LLM_API_BATCH_SIZE = 1    # 一次发送 条数据
    LLM_MAX_RETRIES = 3
    LLM_TIMEOUT = 300          # 批处理处理时间较长，增加超时时间
    LLM_DELAY = 1              # 批次间的间隔

    # 测试时抽样大小 (None表示跑全量)
    TEST_SAMPLE_SIZE = 300
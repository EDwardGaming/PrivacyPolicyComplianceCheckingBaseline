import torch

# ==================== 配置参数 ====================
class Config:    
    # 数据路径
    DATA_PATH = "./dataset/data.tsv"
    
    # 标签映射
    LABEL_NAMES = {
        0: "Other",
        1: "Collect Personal Information (CPI)",
        2: "Data Retention Period (DRP)",
        3: "Data Processing Purposes (DPP)",
        4: "Contact Details (CD)",
        5: "Right to Access (RA)",
        6: "Right to Rectify or Erase (RRE)",
        7: "Right to Restrict of Processing (RRP)",
        8: "Right to Object to Processing (ROP)",
        9: "Right to Data Portability (RDP)",
        10: "Right to Lodge a Complaint (RLC)"
    }
    
    # 合规性规则
    COMPLIANCE_RULES = {
        "Rule 1": (1, 2),   # CPI -> DRP
        "Rule 2": (1, 3),   # CPI -> DPP
        "Rule 3": (1, 4),   # CPI -> CD
        "Rule 4": (1, 5),   # CPI -> RA
        "Rule 5": (1, 6),   # CPI -> RRE
        "Rule 6": (1, 7),   # CPI -> RRP
        "Rule 7": (1, 8),   # CPI -> ROP
        "Rule 8": (1, 9),   # CPI -> RDP
        "Rule 9": (1, 10),  # CPI -> RLC
    }
    
    # 模型通用参数
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 训练模型参数 (BiLSTM/BERT)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    LSTM_LAYERS = 2
    DROPOUT = 0.5
    BERT_MODEL = "./bert-base-uncased"
    MAX_LENGTH = 128
    BATCH_SIZE = 8
    EPOCHS = 10
    LR_BILSTM = 2e-4
    LR_BERT = 2e-5
    
    # ==================== Llama API 配置 ====================
    # 注意：这里需要填入支持 Llama 3.1 405B 的服务商配置 (如 DeepInfra, OpenRouter, SiliconFlow 等)
    LLM_API_KEY = "sk-lDM44EPTt9vFx4DX24C597A5CdC346D8Ba006d1f3d4bAf10"  
    LLM_BASE_URL = "https://kfcv50.link/v1" 
    LLM_MODEL_ID = "gpt-5"
    
    # 批处理配置
    LLM_API_BATCH_SIZE = 20    # 一次发送20条数据
    LLM_MAX_RETRIES = 3
    LLM_TIMEOUT = 120          # 批处理处理时间较长，增加超时时间
    LLM_DELAY = 1              # 批次间的间隔
    
    # 测试时抽样大小 (None表示跑全量)
    TEST_SAMPLE_SIZE = 2000
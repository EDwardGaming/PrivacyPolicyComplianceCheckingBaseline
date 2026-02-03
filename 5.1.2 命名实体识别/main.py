import os
#os.environ["http_proxy"] = "http://127.0.0.1:7890"
#os.environ["https_proxy"] = "http://127.0.0.1:7890"

from data import read_all_corpus, split_dataset, build_map, count_entity_spans
from utils import extend_maps, prepocess_data_for_lstmcrf
from evaluate import hmm_train_eval, crf_train_eval, bilstm_train_and_eval, bert_train_and_eval, gptner_train_and_eval, ensemble_evaluate, cnn_train_and_eval, bilstm_cnn_train_and_eval
import models.config



def main():
    """训练模型，评估结果"""
    
    # ==================== 步骤1: 读取所有数据 ====================
    print("\n" + "=" * 80)
    print("步骤1: 读取所有NER标注数据")
    print("=" * 80)
    all_word_lists, all_tag_lists = read_all_corpus()
    
    # ==================== 数据集抽样打印 ====================
    print("\n" + "=" * 80)
    print("数据集抽样展示（随机50条）")
    print("=" * 80)
    import random
    sample_indices = random.sample(range(len(all_word_lists)), min(10, len(all_word_lists)))
    for idx, sample_idx in enumerate(sample_indices, 1):
        words = all_word_lists[sample_idx]
        tags = all_tag_lists[sample_idx]
        sentence = ''.join(words)
        print(f"\n样本 {idx}:")
        print(f"  句子: {sentence}")
        print(f"  标签: {tags}")
    
    # ==================== 步骤2: 统计实体类型 ====================
    print("\n" + "=" * 80)
    print("步骤2: 统计所有数据中的实体类型")
    print("=" * 80)
    all_counts = count_entity_spans(all_tag_lists)
    print("所有数据实体统计:")
    for entity_type, count in sorted(all_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {entity_type}: {count} 个")
    print(f"  总计: {sum(all_counts.values())} 个实体")
    
    # ==================== 步骤3: 划分数据集 ====================
    print("\n" + "=" * 80)
    print("步骤3: 划分训练集/验证集/测试集")
    print("=" * 80)
    (train_word_lists, train_tag_lists), \
    (dev_word_lists, dev_tag_lists), \
    (test_word_lists, test_tag_lists) = split_dataset(
        all_word_lists, all_tag_lists,
        train_ratio=0.7,
        dev_ratio=0.15,
        test_ratio=0.15,
        shuffle=True,
        random_seed=42
    )
    
    # ==================== 步骤4: 统计各数据集实体分布 ====================
    print("\n" + "=" * 80)
    print("步骤4: 统计各数据集的实体分布")
    print("=" * 80)
    train_counts = count_entity_spans(train_tag_lists)
    dev_counts = count_entity_spans(dev_tag_lists)
    test_counts = count_entity_spans(test_tag_lists)
    
    print("训练集:", dict(train_counts))
    print("验证集:", dict(dev_counts))
    print("测试集:", dict(test_counts))
    
    # ==================== 步骤5: 构建词表和标签表 ====================
    print("\n" + "=" * 80)
    print("步骤5: 构建词表和标签表")
    print("=" * 80)
    word2id = build_map(train_word_lists)
    tag2id = build_map(train_tag_lists)
    print(f"词表大小: {len(word2id)}")
    print(f"标签数量: {len(tag2id)}")
    print(f"标签列表: {list(tag2id.keys())}")
    
    # ==================== 步骤6: 开始训练模型 ====================
    print("\n" + "=" * 80)
    print("步骤6: 训练和评估模型")
    print("=" * 80)
    
    '''
    # 训练评估HMM模型
    print("\n正在训练评估HMM模型...")
    hmm_pred = hmm_train_eval(
        (train_word_lists, train_tag_lists),
        (test_word_lists, test_tag_lists),
        word2id,
        tag2id
    )


    # 训练评估CRF模型
    print("\n正在训练评估CRF模型...")
    crf_pred = crf_train_eval(
        (train_word_lists, train_tag_lists),
        (test_word_lists, test_tag_lists)
    )
    
    
    # 训练评估BI-LSTM模型
    print("\n正在训练评估双向LSTM模型...")
    
    # LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
    bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
    
    
    lstm_pred = bilstm_train_and_eval(
        (train_word_lists, train_tag_lists),
        (dev_word_lists, dev_tag_lists),
        (test_word_lists, test_tag_lists),
        bilstm_word2id, bilstm_tag2id,
        crf=False
    )

    print("\n正在训练评估Bi-LSTM+CRF模型...")
    # 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
    # 还需要额外的一些数据处理
    train_word_lists_crf, train_tag_lists_crf = prepocess_data_for_lstmcrf(
        train_word_lists, train_tag_lists
    )
    dev_word_lists_crf, dev_tag_lists_crf = prepocess_data_for_lstmcrf(
        dev_word_lists, dev_tag_lists
    )
    test_word_lists_crf, test_tag_lists_crf = prepocess_data_for_lstmcrf(
        test_word_lists, test_tag_lists, test=True
    )
    lstmcrf_pred = bilstm_train_and_eval(
        (train_word_lists_crf, train_tag_lists_crf),
        (dev_word_lists_crf, dev_tag_lists_crf),
        (test_word_lists_crf, test_tag_lists_crf),
        crf_word2id, crf_tag2id
    )
    
    # 训练评估CNN模型
    print("\n正在训练评估CNN模型...")
    # CNN模型使用与BiLSTM相同的word2id (含PAD/UNK)
    cnn_pred = cnn_train_and_eval(
        (train_word_lists, train_tag_lists),
        (dev_word_lists, dev_tag_lists),
        (test_word_lists, test_tag_lists),
        bilstm_word2id, bilstm_tag2id
    )

    # 训练评估BiLSTM+CNN模型
    print("\n正在训练评估BiLSTM+CNN模型...")
    bilstm_cnn_pred = bilstm_cnn_train_and_eval(
        (train_word_lists, train_tag_lists),
        (dev_word_lists, dev_tag_lists),
        (test_word_lists, test_tag_lists),
        bilstm_word2id, bilstm_tag2id
    )
    
    # ==================== 使用 BERT 模型预测 ====================
    print("\n正在训练评估BERT模型...")
    bert_pred = bert_train_and_eval(
        (train_word_lists, train_tag_lists),
        (dev_word_lists, dev_tag_lists),
        (test_word_lists, test_tag_lists),
        word2id, tag2id
    )
    '''
    
    # ==================== 使用 GPT 模型预测 ====================
    print("\n" + "=" * 80)
    print("步骤7: 使用 大语言模型 预测")
    print("=" * 80)
    # 从测试集中抽取100条数据进行测试
    sample_size = models.config.gptConfig.TEST_SAMPLE_SIZE
    sampled_test_word_lists = test_word_lists[:sample_size]
    sampled_test_tag_lists = test_tag_lists[:sample_size]

    # 使用 GPT 模型预测
    gpt_predictions = gptner_train_and_eval(
        test_data=(sampled_test_word_lists, sampled_test_tag_lists)
    )
    print("大语言模型预测完成")
    
    '''
    # ==================== 使用 Ensemble 模型预测 ====================
    print("\n正在进行集成学习评估...")
    ensemble_evaluate(
        [hmm_pred, crf_pred, lstm_pred, lstmcrf_pred, bert_pred, cnn_pred, bilstm_cnn_pred],
        test_tag_lists
    )
    '''
    
    print("\n" + "=" * 80)
    print("所有训练和评估完成！")
    print("=" * 80)



if __name__ == "__main__":
    main()
import os
import random
from collections import Counter


def read_all_corpus(data_dir="data"):
    """
    读取所有NER标注语料，返回所有句子和标签
    """
    word_lists = []
    tag_lists = []
    target_extensions = ["bmes"]
    
    root_dir = os.path.abspath(os.path.join(os.getcwd(), data_dir))
    print(f"开始遍历根目录：{root_dir}\n")
    
    encodings_to_try = ["gb18030", "gbk", "utf-8", "gb2312", "utf-8-sig", "iso-8859-1"]
    
    file_count = 0
    error_count = 0
    
    for root, _, files in os.walk(root_dir):
        for file_name in files:
            if any(file_name.endswith(ext) for ext in target_extensions):
                file_path = os.path.join(root, file_name)
                f_obj = None
                used_encoding = None
                
                # 尝试多种编码打开文件
                for enc in encodings_to_try:
                    try:
                        with open(file_path, 'r', encoding=enc) as temp_f:
                            temp_f.read()
                        f_obj = open(file_path, 'r', encoding=enc)
                        used_encoding = enc
                        break
                    except (UnicodeDecodeError, LookupError, OSError):
                        continue
                
                if f_obj is None:
                    print(f"警告：无法以已知编码读取文件 {file_path}，已跳过。")
                    error_count += 1
                    continue
                
                file_count += 1
                word_list = []
                tag_list = []
                
                for line_num, raw_line in enumerate(f_obj, 1):
                    clean_line = raw_line.rstrip('\n\r')
                    
                    # 处理空行（句子分隔符）
                    if not clean_line:
                        if word_list:
                            word_lists.append(word_list.copy())
                            tag_lists.append(tag_list.copy())
                            word_list.clear()
                            tag_list.clear()
                        continue
                    
                    # 从右边分割最后一个空格
                    parts = clean_line.rsplit(' ', 1)
                    
                    if len(parts) != 2:
                        print(f"警告：解析行失败（跳过）: {file_path} 第{line_num}行")
                        continue
                    
                    word, tag = parts[0], parts[1]
                    
                    if tag:
                        word_list.append(word)
                        tag_list.append(tag)
                
                # 文件结束，保存残留句子
                if word_list:
                    word_lists.append(word_list)
                    tag_lists.append(tag_list)
                
                try:
                    f_obj.close()
                except OSError:
                    pass
    
    print(f"读取完成: {file_count} 个文件，{len(word_lists)} 个句子")
    if error_count > 0:
        print(f"跳过文件: {error_count} 个\n")
    return word_lists, tag_lists


def count_entity_spans(tag_lists):
    """
    统计实体数量（按实体span计数）
    支持 BMES/BIO 标注格式
    """
    counts = Counter()
    for seq in tag_lists:
        prev_prefix = 'O'
        prev_type = None
        for tag in seq:
            if tag is None:
                prev_prefix, prev_type = 'O', None
                continue
            tag = tag.strip()
            if tag == '' or tag.upper() == 'O':
                prev_prefix, prev_type = 'O', None
                continue
            if '-' not in tag:
                prev_prefix, prev_type = 'O', None
                continue
            
            prefix, typ = tag.split('-', 1)
            prefix = prefix.upper()
            
            # 单字实体 (BMES中的S)
            if prefix == 'S':
                counts[typ] += 1
                prev_prefix, prev_type = 'S', typ
            # 实体开始 (BMES中的B，BIO中的B)
            elif prefix == 'B':
                counts[typ] += 1
                prev_prefix, prev_type = 'B', typ
            # 实体中间或结束 (BMES中的M/E，BIO中的I)
            elif prefix in ('I', 'M', 'E'):
                # 如果前一个是同类型的B/I/M，视为续接
                if prev_prefix in ('B', 'I', 'M') and prev_type == typ:
                    prev_prefix, prev_type = prefix, typ
                else:
                    # 否则视为新实体（容错处理）
                    counts[typ] += 1
                    prev_prefix, prev_type = prefix, typ
            else:
                prev_prefix, prev_type = 'O', None
    return counts


def split_dataset(word_lists, tag_lists, train_ratio=0.7, dev_ratio=0.15, test_ratio=0.15, shuffle=True, random_seed=42):
    """
    将数据集划分为训练集、验证集和测试集
    
    参数:
        word_lists: 所有句子的词列表
        tag_lists: 所有句子的标签列表
        train_ratio: 训练集比例（默认70%）
        dev_ratio: 验证集比例（默认15%）
        test_ratio: 测试集比例（默认15%）
        shuffle: 是否打乱数据（默认True）
        random_seed: 随机种子（默认42）
    
    返回:
        (train_data, dev_data, test_data)
        每个data是(word_lists, tag_lists)元组
    """
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    # 创建索引列表
    total_size = len(word_lists)
    indices = list(range(total_size))
    
    # 打乱数据
    if shuffle:
        random.seed(random_seed)
        random.shuffle(indices)
    
    # 计算划分点
    train_size = int(total_size * train_ratio)
    dev_size = int(total_size * dev_ratio)
    
    train_indices = indices[:train_size]
    dev_indices = indices[train_size:train_size + dev_size]
    test_indices = indices[train_size + dev_size:]
    
    # 划分数据
    train_words = [word_lists[i] for i in train_indices]
    train_tags = [tag_lists[i] for i in train_indices]
    
    dev_words = [word_lists[i] for i in dev_indices]
    dev_tags = [tag_lists[i] for i in dev_indices]
    
    test_words = [word_lists[i] for i in test_indices]
    test_tags = [tag_lists[i] for i in test_indices]
    
    print(f"数据集划分:")
    print(f"  训练集: {len(train_words)} 句 ({train_ratio*100:.1f}%)")
    print(f"  验证集: {len(dev_words)} 句 ({dev_ratio*100:.1f}%)")
    print(f"  测试集: {len(test_words)} 句 ({test_ratio*100:.1f}%)")
    
    return (train_words, train_tags), (dev_words, dev_tags), (test_words, test_tags)


def build_map(lists):
    """构建词/标签到id的映射"""
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)
    return maps


# ============ 保留旧的 build_corpus 函数（兼容性） ============
def build_corpus(split, make_vocab=True, data_dir="隐私协议人工标注语料库-NER模型训练数据"):
    """
    旧版本函数，保留用于兼容性
    建议使用新的 read_all_corpus + split_dataset 组合
    """
    print("警告: build_corpus 已过时，建议使用 read_all_corpus + split_dataset")
    word_lists, tag_lists = read_all_corpus(data_dir)
    
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists
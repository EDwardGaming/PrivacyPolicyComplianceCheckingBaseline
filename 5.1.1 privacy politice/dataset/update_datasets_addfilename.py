import json
import csv
import os
from collections import Counter

def main():
    # 配置相关文件路径
    tsv_path = 'data.tsv'
    target_files = [
        #'rag_test.jsonl',
        'rag_train_dedup.jsonl',
        #'ablation_test.jsonl',
        'ablation_train_dedup.jsonl'
    ]

    # 1. 加载 data.tsv 映射
    # 假设 data.tsv 包含表头 'sentence' 和 'filename'
    print(f"正在加载映射文件: {tsv_path} ...")
    sentence_to_filename = {}
    
    if not os.path.exists(tsv_path):
        print(f"错误: 找不到文件 {tsv_path}")
        return

    with open(tsv_path, 'r', encoding='utf-8') as f:
        # 使用 csv 模块处理 tsv，delimiter 设置为制表符
        reader = csv.DictReader(f, delimiter='\t')
        
        # 简单的表头检查
        if 'sentence' not in reader.fieldnames:
            print("错误: data.tsv 中缺少 'sentence' 列")
            return
        
        # 如果 data.tsv 没有 filename 列头，请根据实际情况调整，这里假设有
        filename_key = 'filename' if 'filename' in reader.fieldnames else reader.fieldnames[1]

        for row in reader:
            s = row['sentence']
            fn = row[filename_key]
            sentence_to_filename[s] = fn

    print(f"映射加载完成，共 {len(sentence_to_filename)} 条数据。")

    # 2. 处理每个 JSONL 文件
    for file_path in target_files:
        if not os.path.exists(file_path):
            print(f"\n警告: 文件 {file_path} 不存在，跳过。")
            continue

        print(f"\n正在处理文件: {file_path} ...")
        
        updated_data = []
        label_counts = Counter()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    print("警告: 发现无法解析的 JSON 行，已跳过")
                    continue

                # 添加 filename 字段
                input_text = item.get('Input')
                if input_text in sentence_to_filename:
                    item['filename'] = sentence_to_filename[input_text]
                else:
                    # 如果找不到映射，根据需求可以置为 None 或记录日志
                    item['filename'] = None
                
                # 统计 Response 标签频率
                response = item.get('Response')
                label_counts[response] += 1
                
                updated_data.append(item)
        
        # 将修改后的数据写回文件 (先写临时文件再替换，防止写入中断导致数据丢失)
        temp_path = file_path + '.tmp'
        with open(temp_path, 'w', encoding='utf-8') as f:
            for item in updated_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        os.replace(temp_path, file_path)
        
        # 3. 输出统计结果
        print(f"文件 {file_path} 处理完成。")
        print("标签频率统计 (Response):")
        
        # 对标签进行排序以便查看 (处理可能的类型不一致)
        def sort_key(k):
            try:
                return int(k)
            except (ValueError, TypeError):
                return -1 # None 或非数字排在最前

        for label in sorted(label_counts.keys(), key=sort_key):
            print(f"  标签 {label}: {label_counts[label]} 次")

if __name__ == "__main__":
    main()

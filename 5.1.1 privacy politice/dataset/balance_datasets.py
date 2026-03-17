import json
import os
import random
from collections import Counter

def main():
    # 仅处理指定的训练集文件
    target_files = [
        'rag_train_dedup.jsonl',
        'ablation_train_dedup.jsonl'
    ]
    
    # 配置阈值
    MIN_COUNT = 150   # 最小数量，不足则复制
    MAX_TOTAL = 5000  # 数据集总数量上限
    MIN_LABEL_0 = 2000 # 标签 0 的最小保留数量
    
    # 设置随机种子，保证结果可复现
    random.seed(42)

    for file_path in target_files:
        if not os.path.exists(file_path):
            print(f"错误: 找不到文件 {file_path}")
            continue

        print(f"正在处理文件: {file_path} ...")
        
        # 1. 读取所有数据
        all_items = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    all_items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        # 2. 按标签分组
        grouped_data = {}
        for item in all_items:
            label = item.get('Response')
            if label not in grouped_data:
                grouped_data[label] = []
            grouped_data[label].append(item)
            
        balanced_items = []
        
        # 获取排序后的标签列表，方便日志输出
        sorted_labels = sorted(grouped_data.keys(), key=lambda x: int(x) if isinstance(x, (int, str)) and str(x).isdigit() else -1)
        
        # 3. 执行平衡策略 - 第一步：过采样 (Upsampling)
        processed_groups = {}
        
        for label in sorted_labels:
            items = grouped_data[label]
            count = len(items)
            
            # 策略 1: 数量少于 200 -> 自我复制
            if count < MIN_COUNT:
                needed = MIN_COUNT - count
                print(f"  标签 {label}: 原数量 {count} < {MIN_COUNT}，需补充 {needed} 条。")
                
                # 循环复制直到满足数量
                supplement = []
                while len(supplement) < needed:
                    # 每次都打乱顺序再复制，增加一点随机性（虽然内容是一样的）
                    pool = list(items)
                    random.shuffle(pool)
                    supplement.extend(pool)
                
                # 截取刚好需要的数量
                items.extend(supplement[:needed])
                print(f"    -> 补充后数量: {len(items)}")
            
            processed_groups[label] = items

        # 4. 执行平衡策略 - 第二步：检查总数并削减标签 0 (Downsampling)
        current_total = sum(len(items) for items in processed_groups.values())
        
        if current_total > MAX_TOTAL:
            excess = current_total - MAX_TOTAL
            print(f"  当前总数据量 {current_total} > {MAX_TOTAL}，需减少 {excess} 条 (仅针对标签 0)。")
            
            # 查找标签 0 (可能是整数 0 或字符串 "0")
            label_0_key = None
            if 0 in processed_groups:
                label_0_key = 0
            elif "0" in processed_groups:
                label_0_key = "0"
                
            if label_0_key is not None:
                items_0 = processed_groups[label_0_key]
                count_0 = len(items_0)
                
                # 计算允许减少的最大数量 (保留至少 MIN_LABEL_0)
                can_reduce = max(0, count_0 - MIN_LABEL_0)
                # 实际减少数量取 excess 和 can_reduce 的较小值
                reduction = min(excess, can_reduce)
                
                if reduction > 0:
                    new_count_0 = count_0 - reduction
                    print(f"  标签 {label_0_key}: 原数量 {count_0} -> 减少至 {new_count_0} (保留底线 {MIN_LABEL_0})")
                    processed_groups[label_0_key] = random.sample(items_0, new_count_0)
                    
                    if reduction < excess:
                        print(f"  注意: 为保证标签 0 数量不低于 {MIN_LABEL_0}，总数据量将超过 {MAX_TOTAL} (当前: {current_total - reduction})。")
                else:
                    print(f"  警告: 标签 0 数量 ({count_0}) 已接近或低于 {MIN_LABEL_0}，无法执行削减。总数据量将保持为 {current_total}。")
            else:
                print("  警告: 未找到标签 0，无法执行削减操作以满足总数限制。")
        else:
            print(f"  当前总数据量 {current_total} <= {MAX_TOTAL}，无需削减。")

        # 5. 合并数据
        for label in sorted_labels:
            if label in processed_groups:
                balanced_items.extend(processed_groups[label])
            
        # 4. 打乱最终数据集并写回文件
        random.shuffle(balanced_items)
        
        # 先写临时文件，确保写入成功后再替换
        temp_path = file_path + '.tmp'
        with open(temp_path, 'w', encoding='utf-8') as f:
            for item in balanced_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        os.replace(temp_path, file_path)
        
        print(f"文件 {file_path} 处理完毕。总数据量: {len(balanced_items)}")
        
        # 打印最终统计
        final_counts = Counter([item.get('Response') for item in balanced_items])
        print("最终标签频率统计:")
        for label in sorted(final_counts.keys(), key=lambda x: int(x) if isinstance(x, (int, str)) and str(x).isdigit() else -1):
            print(f"  标签 {label}: {final_counts[label]}")
        print("-" * 40 + "\n")

if __name__ == "__main__":
    main()

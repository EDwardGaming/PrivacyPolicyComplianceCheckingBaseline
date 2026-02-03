import os
import pandas as pd
from collections import defaultdict

def count_label_fields(root_dir):
    """
    统计指定目录下所有TSV文件中的label字段分布
    
    参数:
        root_dir: 要搜索TSV文件的根目录
    返回:
        包含所有文件label统计结果的字典
    """
    # 存储总体统计结果
    total_counts = defaultdict(int)
    # 存储每个文件的统计结果
    file_counts = {}
    
    # 遍历目录下所有文件
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # 只处理TSV文件
            if filename.lower().endswith('.tsv'):
                file_path = os.path.join(dirpath, filename)
                try:
                    # 读取TSV文件，使用\t作为分隔符
                    df = pd.read_csv(file_path, sep='\t')
                    
                    # 检查是否包含label字段
                    if 'label' not in df.columns:
                        print(f"警告: 文件 {file_path} 不包含 'label' 字段，已跳过")
                        continue
                    
                    # 统计当前文件的label分布
                    counts = df['label'].value_counts().to_dict()
                    file_counts[file_path] = counts
                    
                    # 更新总体统计
                    for label, count in counts.items():
                        total_counts[label] += count
                        
                    print(f"已处理: {file_path}")
                    
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {str(e)}")
    
    return total_counts, file_counts

def print_statistics(total_counts, file_counts):
    """打印统计结果"""
    print("\n===== 总体统计结果 =====")
    for label, count in sorted(total_counts.items()):
        print(f"label {label}: {count} 次")
    
    print("\n===== 各文件统计结果 =====")
    for file_path, counts in file_counts.items():
        print(f"\n文件: {file_path}")
        for label, count in sorted(counts.items()):
            print(f"  label {label}: {count} 次")

if __name__ == "__main__":
    # 指定要搜索的目录（当前目录）
    target_directory = "./dataset"  # 可替换为具体路径，如 "data/tsv_files"
    
    print(f"开始统计 {target_directory} 下所有TSV文件的label字段...")
    total, files = count_label_fields(target_directory)
    print_statistics(total, files)
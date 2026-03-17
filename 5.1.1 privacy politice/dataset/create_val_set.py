import json
import random
import os

def main():
    # 定义输入和输出文件
    rag_test_file = 'rag_test.jsonl'
    ablation_test_file = 'ablation_test.jsonl'
    
    rag_val_file = 'rag_val.jsonl'
    ablation_val_file = 'ablation_val.jsonl'
    
    # 检查输入文件是否存在
    if not os.path.exists(rag_test_file):
        print(f"错误: 找不到文件 {rag_test_file}")
        return
        
    if not os.path.exists(ablation_test_file):
        print(f"错误: 找不到文件 {ablation_test_file}")
        return

    print("正在读取文档列表...")
    
    # 1. 获取所有唯一的 filename
    # 我们以 rag_test.jsonl 为基准获取文档列表，确保两个验证集包含的是同一批文档
    all_filenames = set()
    
    try:
        with open(rag_test_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    item = json.loads(line)
                    if 'filename' in item:
                        all_filenames.add(item['filename'])
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"读取文件出错: {e}")
        return
    
    unique_files = list(all_filenames)
    total_docs = len(unique_files)
    print(f"共发现 {total_docs} 个唯一文档。")
    
    # 2. 计算 10% 的数量
    val_count = int(total_docs * 0.1)
    if val_count < 1 and total_docs > 0:
        val_count = 1 # 如果文档很少，至少取一个
        
    print(f"计划抽取 {val_count} 个文档作为验证集 (10%)。")
    
    # 3. 随机抽取
    random.seed(42) # 固定种子以便复现
    val_filenames = set(random.sample(unique_files, val_count))
    
    # 4. 定义处理函数
    def extract_to_val(input_path, output_path, target_filenames):
        print(f"正在处理 {input_path} -> {output_path} ...")
        count = 0
        try:
            with open(input_path, 'r', encoding='utf-8') as f_in, \
                 open(output_path, 'w', encoding='utf-8') as f_out:
                for line in f_in:
                    line = line.strip()
                    if not line: continue
                    try:
                        item = json.loads(line)
                        # 如果该行的 filename 在我们选中的验证集列表中，则写入验证集文件
                        # 注意：这里是复制(Copy)，原文件内容不会被删除
                        if item.get('filename') in target_filenames:
                            f_out.write(line + '\n')
                            count += 1
                    except json.JSONDecodeError:
                        continue
            print(f"  已写入 {count} 条数据到 {output_path}")
        except Exception as e:
            print(f"处理文件 {input_path} 时出错: {e}")

    # 5. 执行抽取
    extract_to_val(rag_test_file, rag_val_file, val_filenames)
    extract_to_val(ablation_test_file, ablation_val_file, val_filenames)
    
    print("验证集生成完毕。")

if __name__ == "__main__":
    main()

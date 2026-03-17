import os

def extract_loss_logs(input_file, output_file="train_loss.txt"):
    if not os.path.exists(input_file):
        print(f"找不到日志文件: {input_file}")
        return
        
    extracted_count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            # 考虑到可能混杂着 tqdm 进度条等前缀，我们提取 { 开始到 } 结束的子串
            start_idx = line.find('{')
            end_idx = line.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                dict_str = line[start_idx:end_idx+1]
                f_out.write(dict_str + '\n')
                extracted_count += 1
                
    print(f"提取完成！共提取 {extracted_count} 行记录，已保存至 {output_file}")

if __name__ == "__main__":
    # 运行前，请将 'console_output.txt' 替换为您保存有训练输出的 txt 日志文件名称
    extract_loss_logs('console_output.txt', 'train_loss.txt')
"""检查数据和tag2id映射"""
from data import build_corpus
from collections import Counter

# 读取数据
print("读取训练数据...")
train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train")
print("读取测试数据...")
test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

print("\n" + "="*80)
print("tag2id映射")
print("="*80)
print(f"标签总数: {len(tag2id)}")
for tag, idx in sorted(tag2id.items(), key=lambda x: x[1]):
    print(f"  {tag:10s} -> {idx:2d}")

# 统计训练集标签分布
train_tags = []
for tag_list in train_tag_lists:
    train_tags.extend(tag_list)

train_counter = Counter(train_tags)

print("\n" + "="*80)
print("训练集标签分布")
print("="*80)
print(f"总标签数: {len(train_tags)}")
for tag, count in sorted(train_counter.items(), key=lambda x: -x[1]):
    percentage = count / len(train_tags) * 100
    tag_id = tag2id.get(tag, -1)
    print(f"  {tag:10s} (ID={tag_id:2d}): {count:6d} ({percentage:5.2f}%)")

# 统计测试集标签分布
test_tags = []
for tag_list in test_tag_lists:
    test_tags.extend(tag_list)

test_counter = Counter(test_tags)

print("\n" + "="*80)
print("测试集标签分布")
print("="*80)
print(f"总标签数: {len(test_tags)}")
for tag, count in sorted(test_counter.items(), key=lambda x: -x[1])[:10]:
    percentage = count / len(test_tags) * 100
    tag_id = tag2id.get(tag, -1)
    print(f"  {tag:10s} (ID={tag_id:2d}): {count:6d} ({percentage:5.2f}%)")

# 检查是否有测试集中出现但训练集中没有的标签
test_only_tags = set(test_tags) - set(train_tags)
if test_only_tags:
    print("\n⚠️ WARNING: 以下标签只在测试集出现:")
    for tag in test_only_tags:
        print(f"  {tag}")

print("\n" + "="*80)
print("样本检查")
print("="*80)
print(f"\n第一个训练样本:")
print(f"字符: {train_word_lists[0][:20]}")
print(f"标签: {train_tag_lists[0][:20]}")
print(f"长度: {len(train_word_lists[0])}")

print(f"\n第一个测试样本:")
print(f"字符: {test_word_lists[0][:20]}")
print(f"标签: {test_tag_lists[0][:20]}")
print(f"长度: {len(test_word_lists[0])}")

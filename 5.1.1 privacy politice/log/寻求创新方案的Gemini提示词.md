我们想发SCI论文，我们决定攻克解决“法律隐私政策多标签分类(极度不平衡)“,我们执行了基线实验：

```
分类性能 (10个GDPR标签平均):
Model           Precision    Recall       F1-Score    
---------------------------------------------------
SVM             0.7815       0.5919       0.6690
BiLSTM          0.8001       0.6107       0.6813
BiLSTM+LW       0.7243       0.6543       0.6785
BERT            0.7398       0.6616       0.6917
BERT+LW         0.6685       0.7516       0.7006
Gemini-2.5-pro  0.3565       0.8661       0.4817
ChatGpt-4       0.3487       0.8660       0.4767
Llama3.1-8B     0.4087       0.8315       0.5534   (finetuned+HybridRAG) 

合规性检测性能 (所有模型):

Model           Precision    Recall       F1-Score    
---------------------------------------------------
SVM             0.7181       0.6093       0.6593      
BiLSTM          0.7093       0.6430       0.6745      
BiLSTM+LW       0.6299       0.7318       0.6770      
BERT            0.7470       0.6458       0.6927      
BERT+LW         0.7091       0.7290       0.7189 
Llama3.1-8B     0.6033       0.728        0.6813    (finetuned+HybridRAG) 
```



我们原本提出的创新RAG架构(finetuned+HybridRAG：即对句子分类时投送句子所属的文档的摘要和该句字的上下文部分，以协助大模型进行推理)，性能提升的很有限。 



事实上，由于标签的严重不平衡：

```
标签分布:
  0 (Other): 30699
  1 (Collect Personal Information (CPI)): 1542
  2 (Data Retention Period (DRP)): 448
  3 (Data Processing Purposes (DPP)): 1839
  4 (Contact Details (CD)): 721
  5 (Right to Access (RA)): 115
  6 (Right to Rectify or Erase (RRE)): 562
  7 (Right to Restrict of Processing (RRP)): 127
  8 (Right to Object to Processing (ROP)): 245
  9 (Right to Data Portability (RDP)): 167
  10 (Right to Lodge a Complaint (RLC)): 145
```

 在对大模型进行微调时引入了很多噪声，大模型一直在输出0 (Other)。而且大模型无法像BERT一样采用LW加权损失学习。我们决定重新分析这个任务，我们观察到BERT+LW方法分类性能很好，而LLMs强在推理而不是直接输出分类，为了决定扬长避短，发挥两者优势，我们想提出两阶段任务：

1. 训练一个分类器(比如BERT+LW)，只输出0和1(如果输入句子属于Other输出0，否则输出1)，如果输出0则任务直接结束
2. 如果输出1则把句子喂给LLMs进行推理输出(1-10)。

这样，在LLMs微调训练过程中，可以舍弃含有大量的噪声(0:Other)标签，构建相对平衡的不含o的高质量数据集，让大模型充分学习推理,以提升该分类任务的性能。

现在需要你解决以下问题：

1. 为什么大模型无法像BERT一样采用LW加权损失学习？
2. 分类器用什么合适？BERT+LW吗？请你推荐
3. 需要准备什么消融实验？构建相对平衡的不含o的高质量数据集后，微调LLMs效果相比再微调BERT(用于1-10少数量标签分类)效果会如何？
4. 我们的魔改RAG技术包含了Summary和上下文，你觉得有意义吗是否需要融合后续实验或进行消融？
5. 请你设计实验流程实现方案(比如哪些架构、模型、python库)
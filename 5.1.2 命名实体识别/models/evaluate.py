import time
from collections import Counter
import torch
import torch.optim as optim

from models.hmm import HMM
from models.crf import CRFModel
from models.bilstm_crf import BILSTM_Model
from models.bert_model import BERT_Model
from models.rankgpt import GPTNER
from models.cnn import CNN
from models.bilstm_cnn import BiLSTM_CNN
from models.config import TrainingConfig, LSTMConfig, CNNConfig
from models.util import tensorized, sort_by_lengths, cal_loss
from utils import save_model, flatten_lists
from evaluating import Metrics


def hmm_train_eval(train_data, test_data, word2id, tag2id, remove_O=False):
    """训练并评估hmm模型"""
    # 训练HMM模型
    train_word_lists, train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data

    hmm_model = HMM(len(tag2id), len(word2id))
    hmm_model.train(train_word_lists,
                    train_tag_lists,
                    word2id,
                    tag2id)
    save_model(hmm_model, "./ckpts/hmm.pkl")

    # 评估hmm模型
    pred_tag_lists = hmm_model.test(test_word_lists,
                                    word2id,
                                    tag2id)

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists


def crf_train_eval(train_data, test_data, remove_O=False):

    # 训练CRF模型
    train_word_lists, train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data

    crf_model = CRFModel()
    crf_model.train(train_word_lists, train_tag_lists)
    save_model(crf_model, "./ckpts/crf.pkl")

    pred_tag_lists = crf_model.test(test_word_lists)

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists


def bilstm_train_and_eval(train_data, dev_data, test_data,
                          word2id, tag2id, crf=True, remove_O=False):
    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists, test_tag_lists = test_data

    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    bilstm_model = BILSTM_Model(vocab_size, out_size, crf=crf)
    bilstm_model.train(train_word_lists, train_tag_lists,
                       dev_word_lists, dev_tag_lists, word2id, tag2id)

    model_name = "bilstm_crf" if crf else "bilstm"
    save_model(bilstm_model, "./ckpts/"+model_name+".pkl")

    print("训练完毕,共用时{}秒.".format(int(time.time()-start)))
    print("评估{}模型中...".format(model_name))
    pred_tag_lists, test_tag_lists = bilstm_model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists


def bert_train_and_eval(train_data, dev_data, test_data,
                        word2id, tag2id, remove_O=False):
    """训练并评估BERT模型"""
    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists, test_tag_lists = test_data

    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    bert_model = BERT_Model(vocab_size, out_size)
    bert_model.train(train_word_lists, train_tag_lists,
                     dev_word_lists, dev_tag_lists, word2id, tag2id)

    save_model(bert_model, "./ckpts/bert.pkl")

    print("训练完毕,共用时{}秒.".format(int(time.time()-start)))
    print("评估BERT模型中...")
    pred_tag_lists, test_tag_lists = bert_model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists

def cnn_train_and_eval(train_data, dev_data, test_data, word2id, tag2id, remove_O=False):
    """训练并评估CNN模型"""
    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists, test_tag_lists = test_data

    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    
    model = CNN(vocab_size, LSTMConfig.emb_size, LSTMConfig.hidden_size, out_size, 
                kernel_size=CNNConfig.kernel_size, num_filters=CNNConfig.num_filters)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=TrainingConfig.lr)
    
    # 训练
    train_word_lists, train_tag_lists, _ = sort_by_lengths(train_word_lists, train_tag_lists)
    B = TrainingConfig.lstm_batch_size
    
    for e in range(1, TrainingConfig.epoches+1):
        model.train()
        losses = 0.
        step = 0
        for ind in range(0, len(train_word_lists), B):
            batch_sents = train_word_lists[ind:ind+B]
            batch_tags = train_tag_lists[ind:ind+B]
            
            tensorized_sents, lengths = tensorized(batch_sents, word2id)
            tensorized_sents = tensorized_sents.to(device)
            targets, lengths = tensorized(batch_tags, tag2id)
            targets = targets.to(device)
            
            scores = model(tensorized_sents)
            
            optimizer.zero_grad()
            loss = cal_loss(scores, targets, tag2id).to(device)
            loss.backward()
            optimizer.step()
            
            losses += loss.item()
            step += 1
            
        print("Epoch {}, Loss:{:.4f}".format(e, losses/step))
    
    save_model(model, "./ckpts/cnn.pkl")
    print("训练完毕,共用时{}秒.".format(int(time.time()-start)))
    
    # 评估
    print("评估CNN模型中...")
    model.eval()
    pred_tag_lists = []
    test_word_lists, test_tag_lists, indices = sort_by_lengths(test_word_lists, test_tag_lists)
    
    with torch.no_grad():
        for ind in range(0, len(test_word_lists), B):
            batch_sents = test_word_lists[ind:ind+B]
            tensorized_sents, lengths = tensorized(batch_sents, word2id)
            tensorized_sents = tensorized_sents.to(device)
            
            scores = model(tensorized_sents)
            _, batch_tagids = torch.max(scores, dim=2)
            
            id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
            for i, ids in enumerate(batch_tagids):
                tag_list = [id2tag[ids[j].item()] for j in range(lengths[i])]
                pred_tag_lists.append(tag_list)
    
    # 恢复顺序
    ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
    indices, _ = list(zip(*ind_maps))
    pred_tag_lists = [pred_tag_lists[i] for i in indices]
    test_tag_lists = [test_tag_lists[i] for i in indices]

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists

def bilstm_cnn_train_and_eval(train_data, dev_data, test_data, word2id, tag2id, remove_O=False):
    """训练并评估BiLSTM+CNN模型"""
    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists, test_tag_lists = test_data

    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    
    model = BiLSTM_CNN(vocab_size, LSTMConfig.emb_size, LSTMConfig.hidden_size, out_size, 
                       kernel_size=CNNConfig.kernel_size, num_filters=CNNConfig.num_filters)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=TrainingConfig.lr)
    
    # 训练
    train_word_lists, train_tag_lists, _ = sort_by_lengths(train_word_lists, train_tag_lists)
    B = TrainingConfig.lstm_batch_size
    
    for e in range(1, TrainingConfig.epoches+1):
        model.train()
        losses = 0.
        step = 0
        for ind in range(0, len(train_word_lists), B):
            batch_sents = train_word_lists[ind:ind+B]
            batch_tags = train_tag_lists[ind:ind+B]
            
            tensorized_sents, lengths = tensorized(batch_sents, word2id)
            tensorized_sents = tensorized_sents.to(device)
            targets, lengths = tensorized(batch_tags, tag2id)
            targets = targets.to(device)
            
            # BiLSTM_CNN forward needs lengths
            scores = model(tensorized_sents, lengths)
            
            optimizer.zero_grad()
            loss = cal_loss(scores, targets, tag2id).to(device)
            loss.backward()
            optimizer.step()
            
            losses += loss.item()
            step += 1
            
        print("Epoch {}, Loss:{:.4f}".format(e, losses/step))
    
    save_model(model, "./ckpts/bilstm_cnn.pkl")
    print("训练完毕,共用时{}秒.".format(int(time.time()-start)))
    
    # 评估
    print("评估BiLSTM+CNN模型中...")
    model.eval()
    pred_tag_lists = []
    test_word_lists, test_tag_lists, indices = sort_by_lengths(test_word_lists, test_tag_lists)
    
    with torch.no_grad():
        for ind in range(0, len(test_word_lists), B):
            batch_sents = test_word_lists[ind:ind+B]
            tensorized_sents, lengths = tensorized(batch_sents, word2id)
            tensorized_sents = tensorized_sents.to(device)
            
            scores = model(tensorized_sents, lengths)
            _, batch_tagids = torch.max(scores, dim=2)
            
            id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
            for i, ids in enumerate(batch_tagids):
                tag_list = [id2tag[ids[j].item()] for j in range(lengths[i])]
                pred_tag_lists.append(tag_list)
    
    # 恢复顺序
    ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
    indices, _ = list(zip(*ind_maps))
    pred_tag_lists = [pred_tag_lists[i] for i in indices]
    test_tag_lists = [test_tag_lists[i] for i in indices]

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists

def gptner_train_and_eval(test_data, remove_O=False, batch_size=5):
    test_word_lists, test_tag_lists = test_data
    rankgpt_model = GPTNER()

    pred_tag_lists = []

    for i in range(0, len(test_word_lists), batch_size):
        batch_words = test_word_lists[i:i + batch_size]

        batch_pred_tags = rankgpt_model.predict(batch_words, start_index=i+1)

        for words, pred_tags in zip(batch_words, batch_pred_tags):
            # The predict method in GPTNER is responsible for ensuring pred_tags has the correct length
            # or providing a fallback. So, we can directly append.
            pred_tag_lists.append(pred_tags)

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists


def ensemble_evaluate(results, targets, remove_O=False):
    """ensemble多个模型"""
    for i in range(len(results)):
        results[i] = flatten_lists(results[i])

    pred_tags = []
    for result in zip(*results):
        ensemble_tag = Counter(result).most_common(1)[0][0]
        pred_tags.append(ensemble_tag)

    targets = flatten_lists(targets)
    assert len(pred_tags) == len(targets)

    print("Ensemble 多个模型的结果如下：")
    metrics = Metrics(targets, pred_tags, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

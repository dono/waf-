import json
import random
import warnings
import numpy as np
from concurrent import futures
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


# jsonl形式のトレーニング用データセットを読み込む
def read_train_dataset(path):
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            # if len(d['words']) != 0: # ペイロードが空のデータは除外する
            yield TaggedDocument(d['words'], [d['label']])

# jsonl形式のテスト用データセットを読み込む
def read_test_dataset(path):
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            # if len(d['words']) != 0: # ペイロードが空のデータは除外する
            yield d

def norm_test(model, norm_test_data):
    TP = FN = 0
    for data in norm_test_data:
        # model.random.seed(0)
        vec = model.infer_vector(data['words'])
        label = model.docvecs.most_similar([vec], topn=1)[0][0]
        if label == 'norm':
            TP += 1
        else:
            FN += 1
    return TP, FN

def anom_test(model, anom_test_data):
    FP = TN = 0
    for data in anom_test_data:
        # model.random.seed(0)
        vec = model.infer_vector(data['words'])
        label = model.docvecs.most_similar([vec], topn=1)[0][0]
        if label == 'norm':
            FP += 1
        else:
            TN += 1
    return FP, TN


def get_score(window, min_count, vector_size, alpha, min_alpha, epochs):
    warnings.filterwarnings('ignore', category=FutureWarning)

    train = list(read_train_dataset('./static/train.jsonl'))
    # model = Doc2Vec(train, dm=1, window=6, min_count=15, vector_size=1000, alpha=0.003, min_alpha=0.001, workers=6, epochs=100)
    model = Doc2Vec(train, dm=1, window=window, min_count=min_count, vector_size=vector_size, alpha=alpha, min_alpha=min_alpha, epochs=epochs, workers=6)

    norms = list(read_test_dataset('./static/norm-test.jsonl'))
    anoms = list(read_test_dataset('./static/anom-test.jsonl'))

    with futures.ProcessPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(norm_test, model, norms)
        f2 = executor.submit(anom_test, model, anoms)
        TP, FN = f1.result()
        FP, TN = f2.result()

        print('TP: {}, FN: {}, FP: {}, TN: {}'.format(TP, FN, FP, TN))

        Accuracy = 0
        Precision = 0
        Recall = 0
        F1 = 0

        if (TP + FP) != 0 and (TP + FN) != 0 and (TP + FP + FN + TN) != 0:
            Accuracy = (TP + TN) / (TP + FP + FN + TN)
            Precision = TP / (TP + FP)
            Recall = TP / (TP + FN)
            F1 = 2 * Recall * Precision / (Recall + Precision)

        result = {'params': {'window': window, 'min_count': min_count, 'vector_size': vector_size, 'alpha': alpha, 'min_alpha': min_alpha, 'epochs': epochs},
                  'score': {'Accuracy': Accuracy, 'Precision': Precision, 'Recall': Recall, 'F1': F1}}
        return result

def gen_params():
    params = []
    for window in range(1, 6): # 5
        for min_count in range(10, 31): # 20
            for vector_size in range(10, 1001, 10): # 100
                for alpha in [x / 1000 for x in range(1, 11)]: # 100
                    for epochs in range(10, 301, 10): # 30
                        params.append((window, min_count, vector_size, alpha, epochs))
    return params


if __name__ == '__main__':
    # params = gen_params()
    # for _ in range(20000):
    #     p = random.choice(params)
    #     result = get_score(*p)
    #     print(json.dumps(result))
    # for min_count in [0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001]:
    result = get_score(2, 13, 200, 0.08, 0.01, 80)
    print(json.dumps(result))
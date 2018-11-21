import json
import random
import warnings
import numpy as np
from concurrent import futures
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.ensemble import RandomForestClassifier


# jsonl形式のトレーニング用データセットを読み込む
def read_train_dataset(path):
    with open(path) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            yield TaggedDocument(d['words'], [d['label']+str(i)])

# jsonl形式のテスト用データセットを読み込む
def read_test_dataset(path):
    with open(path) as f:
        for line in f:
            d = json.loads(line)
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


def gen_params():
    params = []
    for window in range(1, 6): # 5
        for min_count in range(10, 31): # 20
            for vector_size in range(10, 1001, 10): # 100
                for alpha in [x / 1000 for x in range(1, 11)]: # 100
                    for epochs in range(10, 301, 10): # 30
                        params.append((window, min_count, vector_size, alpha, epochs))
    return params


def get_score(window, min_count, vector_size, alpha, min_alpha, epochs):
    warnings.filterwarnings('ignore', category=FutureWarning)

    train = list(read_train_dataset('./static/train.jsonl'))
    model = Doc2Vec(train, dm=1, window=window, min_count=min_count, vector_size=vector_size, alpha=alpha, min_alpha=min_alpha, epochs=epochs, workers=6)

    norms = list(read_test_dataset('./static/norm-test.jsonl'))
    anoms = list(read_test_dataset('./static/anom-test.jsonl'))

    with futures.ProcessPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(norm_test, model, norms)
        f2 = executor.submit(anom_test, model, anoms)
        TP, FN = f1.result()
        FP, TN = f2.result()

        print('TP: {}, FN: {}, FP: {}, TN: {}'.format(TP, FN, FP, TN))

        Accuracy = -1
        Precision = -1
        Recall = -1
        F1 = -1

        if (TP + FP) != 0 and (TP + FN) != 0 and (TP + FP + FN + TN) != 0:
            Accuracy = (TP + TN) / (TP + FP + FN + TN)
            Precision = TP / (TP + FP)
            Recall = TP / (TP + FN)
            F1 = 2 * Recall * Precision / (Recall + Precision)

        result = {'params': {'window': window, 'min_count': min_count, 'vector_size': vector_size, 'alpha': alpha, 'min_alpha': min_alpha, 'epochs': epochs},
                  'score': {'Accuracy': Accuracy, 'Precision': Precision, 'Recall': Recall, 'F1': F1}}
        return result


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=FutureWarning)

    norm_train = list(read_train_dataset('./static/norm-train.jsonl'))
    anom_train = list(read_train_dataset('./static/anom-train.jsonl'))

    # model = Doc2Vec([*norm_train, *anom_train], dm=1, window=2, min_count=13, vector_size=200, alpha=0.08, min_alpha=0.01, epochs=160, workers=6)
    # model.save('./tmp/a.model')
    model = Doc2Vec.load('./tmp/a.model')

    norm_train_vecs = np.array([model.docvecs['norm'+str(i)] for i in range(len(norm_train))])
    anom_train_vecs = np.array([model.docvecs['anom'+str(i)] for i in range(len(anom_train))])

    # clf = RandomForestClassifier(random_state=0)


    # testing
    # norms = list(read_test_dataset('./static/norm-test.jsonl'))
    # for norm in norms:
    #     vec = model.infer_vector(norm['words'])

    # anoms = list(read_test_dataset('./static/anom-test.jsonl'))
    # for anom in anoms:
    #     vec = model.infer_vector(anom['words'])

from collections import Counter
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from torch import nn, optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
np.random.seed(0)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class WebpageClassificationModel(nn.Module):
    def __init__(self, embeddings, n_features=10, hidden_size=100, n_classes=19, dropout_prob=0.5):
        super(WebpageClassificationModel, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.embed_size = embeddings.shape[1]
        self.embed_size = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.pretrained_embeddings = nn.Embedding(embeddings.shape[0], self.embed_size)
        self.pretrained_embeddings.weight = nn.Parameter(torch.tensor(embeddings))

        self.embed_to_hidden = nn.Linear(self.n_features * self.embed_size, self.hidden_size)
        nn.init.xavier_uniform_(self.embed_to_hidden.weight.data)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.hidden_to_logits = nn.Linear(self.hidden_size, self.n_classes)
        nn.init.xavier_uniform_(self.hidden_to_logits.weight.data)

    def embedding_lookup(self, t):
        x = self.pretrained_embeddings(t)
        x = x.view(1, -1)
        return x

    def forward(self, t):
        lookup = self.embedding_lookup(t)
        embeddings = self.embed_to_hidden(lookup)
        relu = nn.ReLU()
        hidden = relu(embeddings)
        hidden = self.dropout(hidden)
        logits = self.hidden_to_logits(hidden)

        return logits


def load_and_preprocess_data(data, embedding_file, embed_size, most_common_n):

    c2i = {}  # Category
    w2i = {}  # word id generator
    w2i["NULL"] = 0
    for d in data:
        for w in d[2]:
            if w not in w2i:
                w2i[w] = len(w2i)
        if d[1] not in c2i:
            c2i[d[1]] = len(c2i)
    # assert len(c2i) == 5
    word_vectors = {}
    for line in open(embedding_file, encoding="utf8").readlines():
        sp = line.strip().split()
        word_vectors[sp[0]] = [float(x) for x in sp[1:]]
    embeddings_matrix = np.asarray(np.random.normal(0, 0.9, (len(w2i), embed_size)), dtype='float32')

    for token in w2i:
        i = w2i[token]
        if token in word_vectors:
            embeddings_matrix[i] = word_vectors[token]

    instances = []
    dd = [" ".join(d[2]) for d in data]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(dd)
    df = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names())
    count = 0
    for d in data:
        c = Counter(d[2])
        indexes = []
        x = zip(df.iloc[count],vectorizer.get_feature_names())
        x = sorted(x, key=lambda tup: (tup[0], tup[1]), reverse=True)
        count+=1
        #print("c : ",c.most_common(most_common_n))
        #print("tfdif : ",[(x[0],x[1]) for x in x[0:10]])
        for _,w in x[0:10]:
            try:
                indexes.append(w2i[w])
            except:
                indexes.append(w2i["NULL"])
        indexes += [w2i["NULL"]] * (most_common_n - len(indexes))
        instances.append([d, np.array(indexes), np.array([c2i[d[1]]])])
    return w2i, c2i, embeddings_matrix, instances





def run(data):
    emb_size = 50
    most_common_n_words = 10
    embeding_file_path = "glove.6B/glove.6B.{}d.txt".format(emb_size)

    w2i, c2i, embeddings_matrix, instances = load_and_preprocess_data(data, embeding_file_path, emb_size, most_common_n_words)
    number_of_cat = len(c2i)
    n_epochs = 20
    lr = 0.0005
    classifier = WebpageClassificationModel(embeddings_matrix, n_classes=number_of_cat)
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()

    train_instances, test_instances = train_test_split(instances, test_size=0.33, random_state=42)
    max_acc = 0
    max_acc_epoch = 0
    for epoch in range(n_epochs):
        print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
        loss_meter = AverageMeter()
        classifier.train()
        for i, (d, train_x, train_y) in enumerate(train_instances):
            optimizer.zero_grad()
            loss = 0.
            train_x = torch.from_numpy(train_x).long()
            train_y = torch.from_numpy(train_y).long()
            logits = classifier(train_x)
            loss = loss_func(logits, train_y)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())

        print ("Average Train Loss: {}".format(loss_meter.avg))
        loss_meter = AverageMeter()
        for i, (d, test_x, test_y) in enumerate(test_instances):
            optimizer.zero_grad()
            loss = 0.
            test_x = torch.from_numpy(test_x).long()
            test_y = torch.from_numpy(test_y).long()
            logits = classifier(test_x)
            loss = loss_func(logits, test_y)
            loss_meter.update(loss.item())
        print ("Average Test Loss: {}".format(loss_meter.avg))

        classifier.eval()
        correct = 0.0
        total = 0

        predictions = np.zeros(len(test_instances))
        groundtruth = np.zeros(len(test_instances))

        for i, (d, test_x, test_y) in enumerate(test_instances):
            test_x = torch.from_numpy(test_x).long()
            pred = classifier(test_x)
            pred = pred.detach().numpy()
            pred = np.argmax(pred, 1)

            predictions[i] = pred[0]
            groundtruth[i] = test_y[0]

            total += 1
            if pred[0] == test_y[0]:
                correct += 1

        print('F1: {}'.format(f1_score(groundtruth, predictions, average="weighted",zero_division=1)))
        acc = correct/total*100

        if acc > max_acc:
            max_acc = acc
            max_acc_epoch = epoch
        print("Accuracy ", acc)
        print("Max Accuracy", max_acc)
        print("Max Accuracy", max_acc_epoch)
        print("-"*50 + "\n\n")
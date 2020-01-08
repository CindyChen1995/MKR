import numpy as np
import os
import pickle
import scipy.sparse as sp

np.random.seed(2019)


class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_rating_file_as_matrix(path + "/train_rating.txt")
        self.testRatings = self.load_rating_file_as_list(path + "/test_rating.txt")
        self.testNegatives = self.load_negative_file(path + "/test_negative.txt")
        assert len(self.testRatings) == len(self.testNegatives)

        self.num_users, self.num_items = self.trainMatrix.shape

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            # while line != None and line != "":
            while line is not None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item, 1])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            # while line != None and line != "":
            while line is not None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            # while line != None and line != "":
            while line is not None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            # while line != None and line != "":
            while line is not None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if rating > 0:
                    mat[user, item] = 1.0
                line = f.readline()
        return mat


def getTrainMatrix(train):
    num_users, num_items = train.shape
    train_matrix = np.zeros([num_users, num_items], dtype=np.int32)
    for (u, i) in train.keys():
        train_matrix[u][i] = 1
    return train_matrix


def get_train_instances(train, num_negatives, num_items):
    user_input, item_input, labels = [], [], []
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train.keys():
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    # shuffled_idx = np.random.permutation(len(user_input))
    # user_input = np.array(user_input)[shuffled_idx]
    # item_input = np.array(item_input)[shuffled_idx]
    # labels = np.array(labels)[shuffled_idx]
    return user_input, item_input, labels


def load_data(args):
    n_entity, n_relation, kg = load_kg(args)
    n_user, n_item, train_data, test_data, test_negative_data = load_rating(args)

    print('data loaded.')

    return n_user, n_item, n_entity, n_relation, train_data, test_data, test_negative_data, kg


def load_rating(args):
    print('reading rating file ...')

    dataset = Dataset(args.path + args.dataset)
    train_data, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train_data.shape
    try:
        users, items, labels = pickle.load(open(args.path + args.dataset + '/data_negative.p', mode='rb'))
    except:
        users, items, labels = get_train_instances(train_data, args.num_neg, num_items)
        pickle.dump((users, items, labels), open(args.path + args.dataset + '/data_negative.p', mode='wb'))
    users = np.reshape(users, [-1, 1])
    items = np.reshape(items, [-1, 1])
    labels = np.reshape(labels, [-1, 1])
    data = np.concatenate([users, items, labels], axis=1)
    return num_users, num_items, data, np.array(testRatings), testNegatives


def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = '../data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg = np.load(kg_file + '.npy')
    else:
        kg = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg)

    n_entity = len(set(kg[:, 0]) | set(kg[:, 2]))
    n_relation = len(set(kg[:, 1]))

    # # get kg head dict
    # kg_dict = dict()
    # for i in kg:
    #     if i[0] not in kg_dict:
    #         kg_dict[i] = set()
    #     kg_dict[i].add((i[1], i[2]))

    return n_entity, n_relation, kg

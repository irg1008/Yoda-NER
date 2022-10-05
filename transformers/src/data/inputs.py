from tqdm import tqdm


def flatten(*lists, unique=False):
    flattened = [item for l in lists for sublist in l for item in sublist]
    return list(set(flattened)) if unique else flattened


def get_inputs_and_labels(corpus):
    X, Y = [], []

    print("Getting inputs and labels")
    for line in tqdm(corpus):
        x = list(line.keys())
        y = list(line.values())
        X.append(x)
        Y.append(y)

    return X, Y


def get_vectors(corpus):
    X, Y = get_inputs_and_labels(corpus)

    def get_vector(x, surplus=0):
        flattened = flatten(x, unique=True)
        idx = {word: i + surplus for i, word in enumerate(flattened)}
        vector = [[idx[word] for word in row] for row in x]
        return vector, idx

    vector_X, dict_X = get_vector(X)
    vector_Y, dict_Y = get_vector(Y)

    return X, Y, vector_X, vector_Y, dict_X, dict_Y

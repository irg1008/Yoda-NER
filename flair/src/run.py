import split_data, create_corpus, read_corpus, train, predict


def log(text: str, inverse: bool = False):
    l = 100
    out = [f"{text:^{l}}", "-" * l]
    if inverse:
        out.reverse()

    print("\n")
    print(out[0])
    print(out[1])


def main():
    log("Starting data splitting")
    split_data.main()

    log("Creating corpus")
    create_corpus.main()

    log("Reading corpus")
    corpus = read_corpus.main()

    log("Starting trainning")
    train.main(corpus)

    log("Prediction example")
    predict.main("Cremallera roja talla 56")


if __name__ == "__main__":
    main()

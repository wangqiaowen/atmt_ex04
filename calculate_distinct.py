from nltk.util import ngrams


def distinct_n_sentence_level(sentence, n):
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)


def distinct_n_corpus_level(sentences, n):
    """
    Compute average distinct-N of a list of sentences (the corpus).
    :param sentences: a list of sentence.
    :param n: int, ngram.
    :return: float, the average value.
    """
    return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(sentences)


def load_data(infile):
    sent_list = []
    with open(infile, 'r') as infile:
        for line in infile:
            # print(line)
            line = line.rstrip('\n')
            line = line.split()
            # print(line)
            sent_list.append(line)
    return sent_list


def main():
    for i in range(12):
        sent_list = load_data(f'output_task_4/translation_{i}.txt')
        for j in range(1, 4):
            print(f'output_task_4/translation_{i}.txt')
            print(f'distinct-{j} score: {distinct_n_corpus_level(sent_list, j)}')
            print('='*30)


if __name__ == '__main__':
    main()
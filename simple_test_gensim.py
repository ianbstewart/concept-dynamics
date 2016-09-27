"""
Testing gensim with real simple data.
"""
from gensim.models import Word2Vec

if __name__ == '__main__':
    sentences = ['the cat ran', 'the dog ran', 'the cat ate the dog', 'the dog ran after the cat']
    dimensions = 5
    window = 3
    min_count = 1
    model = Word2Vec(size=dimensions, window=window, min_count=min_count)
    vocab = set(' '.join(sentences).split(' '))
    # put each vocab item in its own list!
    vocab = [[v] for v in vocab]
    # build vocab
    model.build_vocab(vocab)
    tokenized_sentences = [s.split(' ') for s in sentences]
    # train model
    model.train(sentences)
    print('model vocab = %s'%(model.vocab))

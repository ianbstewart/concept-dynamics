"""
How to make timestep-specific word
embeddings on a time-separated corpus,
following Kim et al. (2014).
"""
from gensim.models import Word2Vec
import os, codecs
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize.casual import TweetTokenizer

TKNZR = TweetTokenizer(strip_handles=True, preserve_case=False)
def get_write_temporal_embeddings(timestep_files,
                                  timesteps,
                                  out_dir,
                                  n_dims=100,
                                  window=5, 
                                  min_count=5,
                                  ):
    """
    Collect full vocabulary, initialize model
    with vocabulary as "sentences," then 
    train model with each timestep's sentences,
    writing each set of embeddings to file.
    Assumes files are stored in line-by-line sentence format.
    
    params:
    timestep_files = [str]
    timesteps = [str]
    out_dir = str
    """
    # get base model
    cv = CountVectorizer(min_df=min_count)
    all_vocab = set()
    all_sentences = []
    for t in timestep_files:
        for l in codecs.open(t, 'r', encoding='utf-8'):
            words = TKNZR.tokenize(l)
            #all_sentences.append(words)
            all_sentences.extend(words)
    cv.fit_transform(all_sentences)
    all_vocab = set(cv.vocabulary_.keys())
    model = Word2Vec(size=n_dims, window=window, min_count=1)
    model.build_vocab([[v] for v in all_vocab])
    print('just built model with %d vocab'%(len(all_vocab)))
    # now train and write models
    all_vocab = list(all_vocab)
    dims_str = ['%d'%(d) for d in range(n_dims)]
    for timestep, t in zip(timesteps, timestep_files):
        print('bout to train on data from timestep %d'%(timestep))
        all_sentences = []
        for l in codecs.open(t, 'r', encoding='utf-8'):
            words = TKNZR.tokenize(l)
            all_sentences.append(words)
        model.train(all_sentences)
        model_file_name = os.path.join(out_dir, 
                                       'embeddings_%s.tsv'%(timestep))
        # write each model with rows = words, columns = dimensions
        with codecs.open(model_file_name, 'w', encoding='utf-8') as out_file:
            out_file.write('\t'.join([str(d) for d in range(n_dims)])+'\n')
            for v in all_vocab:
                out_file.write((v + '\t' + 
                               '\t'.join([str(d) for d in model[v]]) + '\n'))

if __name__ == '__main__':
    # line-separated sentences
    data_dir = 'timestep_files'
    timestep_files = [os.path.join(data_dir, f) for f in os.listdir('timestep_files/')]
    timesteps = range(len(timestep_files))
    out_dir = 'output'
    n_dims = 100
    min_count = 5
    get_write_temporal_embeddings(timestep_files,
                                  timesteps,
                                  out_dir,
                                  n_dims=100,
                                  window=5, 
                                  min_count=5,
                                  )

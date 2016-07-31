"""
Testing the change in embeddings over time. Assumes
that we've already generated embeddings in output/.
"""
import pandas as pd
import numpy as np
import os, codecs
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

if __name__ == '__main__':
    out_dir = 'output'
    embedding_files = [os.path.join(out_dir, f) for f in os.listdir(out_dir)]
    # test 0: do the embeddings make semantic sense? 
    end_embedding = pd.read_csv(embedding_files[-1], sep='\t', index_col=0)
    test_words = ['you', 'go', 'road', 'give', 'cold']
    for test_word in test_words:
        sims = end_embedding.apply(lambda r: cosine_similarity(r.reshape(1,-1), 
                                                               end_embedding.loc[test_word].reshape(1,-1))[0][0], 
                                   axis=1)
        print('test word %s has top 10 similarities \n%s'%
              (test_word, sims.sort_values(ascending=False)[:10]))
    # TL;DR the embeddings aren't perfect but they work for more common words
    # test 1: how much have embeddings changed from start to end of data?
    start_embedding = pd.read_csv(embedding_files[1], sep='\t', index_col=0)
    embedding_deltas = abs(cosine_distances(end_embedding, start_embedding))
    embedding_deltas = pd.Series(np.diagonal(embedding_deltas), 
                                 index=end_embedding.index).sort_values(ascending=True)
    print('got embedding deltas %s'%(embedding_deltas))

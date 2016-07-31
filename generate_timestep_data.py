"""
Generate dummy timestep data from the Brown
corpus by splitting all sentences and writing
to file.
"""
from nltk.corpus import brown
import os, codecs

if __name__ == '__main__':
    out_dir = 'timestep_files'
    n_timesteps = 5
    # divide into timestep chunks
    all_sents = [' '.join(s) for s in brown.sents()]
    n_sents = len(all_sents)
    chunk_size = n_sents / n_timesteps
    for t in range(n_timesteps):
        with codecs.open(os.path.join(out_dir, 'timestep_%d.txt'%(t)), 
                         'w', encoding='utf-8') as timestep_file:
            sent_chunk = all_sents[t*chunk_size:(t+1)*chunk_size]
            for s in sent_chunk:
                timestep_file.write(s + '\n')

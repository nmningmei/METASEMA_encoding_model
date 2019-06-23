# METASEMA_encoding_model


## METASEMA dataset

## Goals
- [x] cross-validate standard encoding models using features extracted by word embedding models and computer vision models
- [x] compare the model performance among models use different features extracted by different word embedding models and computer vision models

## Encoding Model Pipeline
```
clf             = linear_model.Ridge(
                  alpha        = 1e2,  # L2 penalty, higher means lower the sum of the weights
                  normalize    = True, # normalize the batch features
                  random_state = 12345,# random seeding
)
X # feature representation matrix
y # BOLD signals
cv # cross validation method (indices)

scorer = make_scorer(r2_score,multioutput = "raw_values")

results = cross_validate(clf,X,y,cv = cv, scoring = scorer,)
scores = results["test_score"]
```

## [Word Embedding Models](https://github.com/dccuchile/spanish-word-embeddings)
```
# for example, load fast test model in memory
fasttext_link: http://dcc.uchile.cl/~jperez/word-embeddings/fasttext-sbwc.vec.gz
fasttest_model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(fasttext_downloaded_file_name)
for word in words:
    word_vector_representation = fasttest_model.get_vector(word)
```
![wordembedding](https://cdn-images-1.medium.com/max/800/1*ZNdGa-lpYoZhvSFIcRaewg.png)
Word vector (From [Introduction to Word Vectors](https://medium.com/@jayeshbahire/introduction-to-word-vectors-ea1d4e4b84bf))
### [FastText](https://fasttext.cc/docs/en/crawl-vectors.html), now supports 157 languages
1. Facebook AI Research lab
2. efficient learning for text classification
3. hierarchical classifier
4. Huffman algorithm to build the tree --> depth of frequent words is smaller than for infrequent ones
5. bag of words (BOW) -- ignore the word order
6. ngrams
![fasttextmetasema](https://github.com/nmningmei/fMRI_decoding_benchmarking/blob/master/figures/metasema/word%20embedding/model%20fast%20text-RSA.png)

### [GloVe](https://nlp.stanford.edu/projects/glove/)
1. Stanford
2. nearest neighbors
3. linear substructures
4. non-zero entries of a global word-word co-occurrence matrix
![glovemetasema](https://github.com/nmningmei/fMRI_decoding_benchmarking/blob/master/figures/metasema/word%20embedding/model%20glove-RSA.png)

### [Word2Vec](https://www.tensorflow.org/tutorials/representation/word2vec)
1. skip-gram model with negative-sampling
2. minimum word frequency is 5
3. negative sampling at 20
4. 273 most common words were downsampled
![w2vmetasema](https://github.com/nmningmei/fMRI_decoding_benchmarking/blob/master/figures/metasema/word%20embedding/model%20word2vec-RSA.png)

## [Computer Vision Models](https://keras.io/applications/)

### VGG19

### DenseNet121

### MobileNet_V2

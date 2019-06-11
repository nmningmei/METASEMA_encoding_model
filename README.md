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
```

## Word Embedding Models
![wordembedding](https://cdn-images-1.medium.com/max/800/1*ZNdGa-lpYoZhvSFIcRaewg.png)
Word vector (From [Introduction to Word Vectors](https://medium.com/@jayeshbahire/introduction-to-word-vectors-ea1d4e4b84bf))
### [FastText](https://fasttext.cc/docs/en/crawl-vectors.html), now supports 157 languages
1. 

### GloVe

### Word2Vec

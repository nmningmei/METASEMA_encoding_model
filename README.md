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
## Computing RDM
```
feature_representations # n_word x n_features
# subtract the mean of each "word" but not standardize it, or normalize each row to its unit vector form.
RDM = distance.squareform(distance.pdist(feature_representations - feature_representations.mean(1).reshape(-1,1),
                           metric = 'cosine',))
# fill NaNs for plotting
np.fill_diagonal(RDM,np.nan)
```

## [Word Embedding Models](https://github.com/dccuchile/spanish-word-embeddings)

![basic](https://jaxenter.com/wp-content/uploads/2018/08/image-2-768x632.png)

[source: Tommaso Teofili August 17, 2018](https://jaxenter.com/deep-learning-search-word2vec-147782.html)

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
```
@Article{bojanowski2016a,
  title={Enriching word vectors with subword information},
  author={Bojanowski, Piotr and Grave, Edouard and Joulin, Armand and Mikolov, Tomas},
  journal={Transactions of the Association for Computational Linguistics},
  volume={5},
  pages={135--146},
  year={2017},
  publisher={MIT Press}
}
```
1. Facebook AI Research lab
2. efficient learning for text classification
3. hierarchical classifier
4. Huffman algorithm to build the tree --> depth of frequent words is smaller than for infrequent ones
5. bag of words (BOW) -- ignore the word order
6. ngrams
7. **represntational space = 300**

![fasttextRDM](https://github.com/nmningmei/METASEMA_encoding_model/blob/master/figures/metasema%20word2vec%20RDM%20(fast%20text).png)

### [GloVe](https://nlp.stanford.edu/projects/glove/)
```
@CONFERENCE{Pennnigton2014a,
  title={Glove: Global vectors for word representation},
  author={Pennington, Jeffrey and Socher, Richard and Manning, Christopher},
  booktitle={Proceedings of the 2014 conference on empirical methods in natural language processing (\uppercase{EMNLP})},
  pages={1532--1543},
  year={2014}
}
```
1. Stanford
2. nearest neighbors
3. linear substructures
4. non-zero entries of a global word-word co-occurrence matrix
5. **representational space = 300**

![gloveRDM](https://github.com/nmningmei/METASEMA_encoding_model/blob/master/figures/metasema%20word2vec%20RDM%20(glove).png)

### [Word2Vec - the 2013 paper](https://www.tensorflow.org/tutorials/representation/word2vec)
```
@inproceedings{mikolov2013a,
  title={Distributed representations of words and phrases and their compositionality},
  author={Mikolov, Tomas and Sutskever, Ilya and Chen, Kai and Corrado, Greg S and Dean, Jeff},
  booktitle={Advances in neural information processing systems},
  pages={3111--3119},
  year={2013}
}
```
1. skip-gram model with negative-sampling
2. minimum word frequency is 5
3. negative sampling at 20
4. 273 most common words were downsampled
5. **representational space = 300**

![w2vRMD](https://github.com/nmningmei/METASEMA_encoding_model/blob/master/figures/metasema%20word2vec%20RDM%20(word2vec).png)

## [Computer Vision Models](https://keras.io/applications/)

### VGG19
```
@article{simonyan2014very,
  title={Very deep convolutional networks for large-scale image recognition},
  author={Simonyan, Karen and Zisserman, Andrew},
  journal={arXiv preprint arXiv:1409.1556},
  year={2014}
}
```
1. small convolution filters (3 x 3)
2. well-generalisible feature representations
3. **representational space = 512**

![vgg19](http://www.eneuro.org/content/eneuro/4/3/ENEURO.0113-17.2017/F10.large.jpg)

[source: Kalfas et al., 2017](http://www.eneuro.org/content/4/3/ENEURO.0113-17.2017)

![vgg19ar](https://cdn-images-1.medium.com/max/1600/1*cufAO77aeSWdShs3ba5ndg.jpeg)

[source: Yang et al., 2018](https://www.researchgate.net/publication/325137356_Breast_cancer_screening_using_convolutional_neural_network_and_follow-up_digital_mammography)

![vgg19RDM](https://github.com/nmningmei/METASEMA_encoding_model/blob/master/figures/metasema%20word2vec%20RDM%20(img2vec%20(vgg19)).png)

### DenseNet121
```
@inproceedings{huang2017densely,
  title={Densely connected convolutional networks},
  author={Huang, Gao and Liu, Zhuang and Van Der Maaten, Laurens and Weinberger, Kilian Q},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4700--4708},
  year={2017}
}
```
1. Each layer is receiving a “collective knowledge” from all preceding layers
2. The error signal can be easily propagated to earlier layers more directly. This is a kind of implicit deep supervision as earlier layers can get direct supervision from the final classification layer.
3. DenseNet performs well when training data is insufficient
4. **representational space = 1028**
![concat](https://cdn-images-1.medium.com/max/800/1*9ysRPSExk0KvXR0AhNnlAA.gif)

[source: Tsang, blog Nov 25, 2018](https://towardsdatascience.com/review-densenet-image-classification-b6631a8ef803)

![feature_map](https://cdn-images-1.medium.com/max/800/1*t_orlp67H-odvgMa4LTzzw.png)

[source: Tsang, blog Nov 25, 2018](https://towardsdatascience.com/review-densenet-image-classification-b6631a8ef803)

![densenetRDM](https://github.com/nmningmei/METASEMA_encoding_model/blob/master/figures/metasema%20word2vec%20RDM%20%28img2vec%20%28densenet121%29%29.png)

### MobileNet_V2
```
@article{howard2017mobilenets,
  title={Mobilenets: Efficient convolutional neural networks for mobile vision applications},
  author={Howard, Andrew G and Zhu, Menglong and Chen, Bo and Kalenichenko, Dmitry and Wang, Weijun and Weyand, Tobias and Andreetto, Marco and Adam, Hartwig},
  journal={arXiv preprint arXiv:1704.04861},
  year={2017}
}
```
1. bottle net feature bottle net
2. mobile-oriented design
3. **representational space = 1280**

![bottlenet](https://machinethink.net/images/mobilenet-v2/ResidualBlock@2x.png)

[source: Hollemans, blog, 22 April, 2018](https://machinethink.net/blog/mobilenet-v2/)

![ar](https://yinguobing.com/content/images/2018/03/mobilenet-v2-conv.jpg)

[source: Guobing, blog, 15 March, 2018](https://yinguobing.com/bottlenecks-block-in-mobilenetv2/)

![mobilenetRDM](https://github.com/nmningmei/METASEMA_encoding_model/blob/master/figures/metasema%20word2vec%20RDM%20%28img2vec%20%28mobilenetv2_1%29.png)

## Results
### Average Variance Explained
![folds](https://github.com/nmningmei/METASEMA_encoding_model/blob/master/figures/fig5.png)
### Difference between Computer Vision models and Word Embedding modles
![comparison1](https://github.com/nmningmei/METASEMA_encoding_model/blob/master/figures/fig6.png)
### The difference between CV and WE model contrasted between Shallow and Deep Processing conditions
![comparison2](https://github.com/nmningmei/METASEMA_encoding_model/blob/master/figures/fig7.png)

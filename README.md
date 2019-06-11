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

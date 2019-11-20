## Fetch Gloval Vector Embeddings
Extract glove.6B.300d.txt from http://nlp.stanford.edu/data/glove.6B.zip to data/ folder (needs to be created).

## Create Conda evnironment
```sh
conda env create
```

## Start service
```sh
FLASK_APP=movie_genres flask run
```

## Post a train dataset
```sh
curl \
        --data-binary @train.csv \
        --header "Content-Type: text/csv" \
        localhost:5000/genres/train
```

## Post a test dataset
```sh
curl \
        --data-binary @test.csv \
        --header "Content-Type: text/csv" \
        --header "Accept: text/csv" \
        localhost:5000/genres/predict
``` 

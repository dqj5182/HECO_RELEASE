## Data

You need to follow directory structure of the `data` as below.

```
${ROOT}
|-- data
|   |-- base_data
|   |-- DAMON
|   |   |-- data
|   |   |-- dataset.py
```

* Download `base_data` from [HuggingFace](https://huggingface.co/datasets/dqj5182/heco-data/resolve/main/eval/data/base_data.tar.gz) by running:

```
bash scripts/download_test_base_data.sh
```

#### DAMON dataset

```
${ROOT}
|-- data
|   |-- DAMON
|   |   |-- data
|   |   |   |-- Release_Datasets
|   |   |   |   |-- damon
|   |   |   |   |   |-- hot_dca_test.npz
|   |   |   |-- HOT-Annotated
|   |   |   |   |-- parts
|   |   |-- dataset.py
```

* Download annotations and segmentations by running after registration at the [official website](https://deco.is.tue.mpg.de/):

```
bash scripts/download_official_damon.sh
```

#### HOT dataset

```
${ROOT}
|-- data
|   |-- DAMON
|   |   |-- data
|   |   |   |-- HOT-Annotated
|   |   |   |   |-- images
```

* Download images by running after registration at the [official website](https://hot.is.tue.mpg.de/):

```
bash scripts/download_official_hot.sh
```
## Data

You need to follow the directory structure of `data` as below.

```
${ROOT}
|-- data
|   |-- base_data
|   |   |-- conversions
|   |   |   |-- smpl_to_flame.npz
|   |   |-- pretrained_models
|   |   |   |-- hmr2
|   |   |   |   |-- hmr2.pth
|   |   |   |   |-- config.json
|   |   |   |   |-- smpl_mean_params.npz
|   |   |   |-- human_detector
|   |   |   |   |-- model_final_f05665.pkl
|   |   |-- human_models
|   |   |   |-- flame
|   |   |   |   |-- FLAME_template.npz
|   |   |   |-- smplx
|   |   |   |   |-- SMPL-X__FLAME_vertex_ids.npy
```

- Download `base_data` from [HuggingFace](https://huggingface.co/datasets/dqj5182/heco-data/blob/main/demo/data/base_data.tar.gz) by running:

```
bash scripts/download_demo_base_data.sh
```
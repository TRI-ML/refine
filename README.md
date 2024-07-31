# ReFiNe: Recursive Field Networks for Cross-Modal Multi-Scene Representation

This repository contains the PyTorch implementation of our paper, **ReFiNe**.

<a href="https://www.tri.global/" target="_blank">
 <img align="right" src="assets/tri-logo.png" width="20%"/>
</a>

### [Project Page](https://zakharos.github.io/projects/refine/) | [Paper](https://arxiv.org/pdf/2406.04309)

<a href="https://zakharos.github.io/"><strong>Sergey Zakharov</strong></a>
·
<a href="https://www.thekatherineliu.com/"><strong>Katherine Liu</strong></a>
·
<a href="https://adriengaidon.com/"><strong>Adrien Gaidon</strong></a>
·
<a href="https://www.tri.global/about-us/dr-rares-ambrus"><strong>Rares Ambrus</strong></a>
<br>SIGGRAPH, 2024<br>

<p align="center">
<img src="assets/method.jpg" width="100%">
</p>

### Installation

To set up the environment using docker, execute the following command:

```
make docker-build
```

Once installed, start an interactive session:

```
make docker-interactive
```
Once inside the interactive session, you can train and validate the method within this environment.

### Generate Training Data
To replicate the workflow of our pipeline, first download [3 objects](https://s3.amazonaws.com/tri-ml-public.s3.amazonaws.com/github/refine/hb_objects.zip) from the [HB dataset](https://campar.in.tum.de/personal/ilic/homebreweddb/index.html).
Once downloaded, put them under the `demo/` folder and unzip. 
To generate GT files, run the following command:
```
python -m data.db_generate --config configs/config.yaml
```

### Training and Inference
The `config.yaml` file stores default parameters for training and evaluation and points to the provided 3 models.
To start **training**, run the following script:

```
python train.py --config configs/config.yaml
```

To **visualize** the trained model using Open3D, run:

```
python visualize.py --path_net log/demo
```

This command will extract a dense point cloud from each of the decoded neural fields and visualize them sequentially. Press `9` to visualize normals, `1` to visualize RGB, `-` and `+` to decrease or increase the size of the points, and `q` to proceed to the next object.

Additionally, you can specify the `lod_inc` parameter to apply an increment on top of the default Level of Detail (LoD) 
to further densify the output point cloud. By default, this parameter is set to 1.

```
python visualize.py --config configs/config.yaml --lod_inc 1
```

### Pre-trained Models

We provide pre-trained models on various datasets:

| Dataset (GB)       | # Objects | Latent | Size (MB) | Link                                                                                           |
|--------------------|-----------|--------|-----------|------------------------------------------------------------------------------------------------|
| Thingi32 (0.47)    | 32        | 64     | 3.4       | [model](https://s3.amazonaws.com/tri-ml-public.s3.amazonaws.com/github/refine/thingi32.zip)    |
| ShapeNet150 (0.63) | 150       | 96     | 4.1       | [model](https://s3.amazonaws.com/tri-ml-public.s3.amazonaws.com/github/refine/shapenet150.zip) |
| HB (0.53)          | 33        | 64     | 4         | [model](https://s3.amazonaws.com/tri-ml-public.s3.amazonaws.com/github/refine/hb.zip)          |
| BOP (0.91)         | 201       | 512    | 92.8      | [model](https://s3.amazonaws.com/tri-ml-public.s3.amazonaws.com/github/refine/bop.zip)         |
| GSO (13.6)         | 1024      | 512    | 94.5      | [model](https://s3.amazonaws.com/tri-ml-public.s3.amazonaws.com/github/refine/gso.zip)         |

To **visualize** the pre-trained model, download it under the `pretrained` folder and run:

```
python visualize.py --config pretrained/[model]/config.yaml
```

### Acknowledgements
We used functions from NVIDIA's [Kaolin](https://github.com/NVIDIAGameWorks/kaolin) and [Kaolin Wisp](https://github.com/NVIDIAGameWorks/kaolin-wisp) libraries, as well as from [Open3D](https://www.open3d.org/), in our implementation.

### Reference

```
@inproceedings{refine,
    title={ReFiNe: Recursive Field Networks for Cross-Modal Multi-Scene Representation},
    author={Sergey Zakharov, Katherine Liu, Adrien Gaidon, Rares Ambrus},
    journal={SIGGRAPH},
    year={2024}
}
```

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />
This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
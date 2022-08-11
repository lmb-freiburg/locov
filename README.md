# LocOV: Localized Vision-Language Matching for Open-vocabulary Object Detection

## News
**2022-07** (v0.1): This repository is the official PyTorch implementation of our GCPR 2022 paper:
<a href="https://arxiv.org/pdf/2205.06160.pdf">Localized Vision-Language Matching for Open-vocabulary Object Detection</a>
<!-- published at ([slides](), [poster](), [poster session]() -->

## Table of Contents
* [News](#news)
* [Table of Contents](#table-of-contents)
* [Installation](#installation)
* [Prepare datasets](#prepare-datasets)
  * [Download datasets](#download-datasets)
  * [Precompute the text features](#precompute-the-text-features)
* [Train and validate Open Vocabulary Detection](#train-and-validate-open-vocabulary-detection)
  * [Model Outline](#model-outline)
  * [Useful script commands](#useful-script-commands)
* [Acknowledgements](#acknowledgements)
* [License](#license)
* [Citation](#citation)

## Installation
### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.8.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check the
  PyTorch version matches the one required by Detectron2 and your CUDA version.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).

Originally the code was tested on `python=3.8.13`, `torch=1.10.0`, `cuda=11.2` and OS `Ubuntu 20.04`.

~~~bash
git clone https://github.com/lmb-freiburg/locov.git
cd locov
~~~

## Prepare datasets
### Download datasets
- Download MS COCO training and validation datasets. Download detection and caption annotations for  retrieval from the original [page](https://cocodataset.org/).
- Save the data in datasets_data
- Run the script to create the annotation subsets that include only base and novel categories
~~~bash
python tools/convert_annotations_to_ov_sets.py
~~~

### Precompute the text features
- Run the script to save and calculate the object embeddings.
~~~bash
python tools/coco_bert_embeddings.py
~~~
- Or download the precomputed ones [Embeddings](https://lmb.informatik.uni-freiburg.de/resources/binaries/gcpr2022_locov/coco_nouns_bertemb.json)

### Precomputed generic object proposals
- Train [OLN](https://github.com/mcahny/object_localization_network) on MSCOCO known classes and extract the proposals for all the training set. 
- Or download the precomputed proposals for MSCOCO Train on known classes only [Proposals](https://lmb.informatik.uni-freiburg.de/resources/binaries/gcpr2022_locov/coco_train2017_seen.pkl) (3.9GB)

## Train and validate Open Vocabulary Detection
### Model Outline
<p align="center"><img src="assets/model.pdf" alt="Method" title="LocOV" /></p>
<img src="assets/model.pdf" width="1000"> <br/>

### Useful script commands
#### Train LSM stage
Run the script to train the Localized Semantic Matching stage
~~~bash
python train_ovnet.py --num-gpus 8 --resume --config-file configs/coco_lsm.yaml 
~~~
#### Train STT stage
Run the script to train the Localized Semantic Matching stage
~~~bash
python train_ovnet.py --num-gpus 8 --resume --config-file configs/coco_stt.yaml MODEL.WEIGHTS path_to_final_weights_lsm_stage
~~~

#### Evaluate
~~~bash
python train_ovnet.py --num-gpus 8 --resume --eval-only --config-file configs/coco_stt.yaml \
MODEL.WEIGHTS output/model-weights.pth \
OUTPUT_DIR output/eval_locov
~~~

### Benchmark results
#### Models zoo
Pretrained models can be found in the models directory

|  Model  |  AP-novel  |  AP50-novel  |  AP-known  |  AP50-known  |  AP-general  |  AP50-general  | Weights |
| ------- | ---------- | ------------ | ---------- | ------------ | ------------ | -------------- | ------- |
| LocOv   | 17.219     | 30.109       | 33.499     | 53.383       | 28.129       | 45.719         | [LocOv](https://lmb.informatik.uni-freiburg.de/resources/binaries/gcpr2022_locov/LocOV.pth) |

## Acknowledgements
This work was supported by Deutscher Akademischer Austauschdienst - German Academic Exchange Service (DAAD) Research Grants - Doctoral Programmes in Germany, 2019/20; grant number: 57440921.

The Deep Learning Cluster used in this work is partially funded by the German Research Foundation (DFG) - 417962828.

We especially thank the creators of the following github repositories for providing helpful code:
- Zareian et al. for their open-vocabulary setup and code: [OVR-CNN](https://github.com/alirezazareian/ovr-cnn)

## License
<a rel="license" href="http://creativecommons.org/licenses/by/3.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/3.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/3.0/">Creative Commons Attribution 3.0 Unported License</a>  To view a copy of this license, visit http://creativecommons.org/licenses/by/3.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

## Citation
If you use our repository or find it useful in your research, please cite the following paper:

<pre class='bibtex'>
@InProceedings{Bravo2022locov,
  author       = "M. Bravo and S. Mittal and T. Brox",
  title        = "Localized Vision-Language Matching for Open-vocabulary Object Detection",
  booktitle    = "German Conference on Pattern Recognition (GCPR) 2022",
  year         = "2022"
}
</pre>

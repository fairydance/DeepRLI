<a name="readme-top"></a>



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/fairydance/DeepRLI">
    <img src="img/logo.svg" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">DeepRLI</h3>

  <p align="center">
    A Multi-objective Framework for Universal Protein–Ligand Interaction Prediction
    <br />
    <a href="https://github.com/fairydance/DeepRLI"><strong>GitHub</strong></a>
    ·
    <a href="https://zenodo.org/records/11116386"><strong>Zenodo</strong></a>
    ·
    <a href="https://arxiv.org/abs/2401.10806"><strong>arXiv</strong></a>
    <br />
    <br />
    <a href="https://doi.org/10.5281/zenodo.11116386">Training Datasets</a>
    ·
    <a href="https://github.com/fairydance/DeepRLI/releases">Trained Models</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#preprocessing">Preprocessing</a></li>
        <li><a href="#training">Training</a></li>
        <li><a href="#inference">Inference</a></li>
      </ul>
    </li>
    <li>
      <a href="#all-in-one">All in One</a>
      <ul>
        <li><a href="#inference-aio">Inference AIO</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

In this study, we propose and implement DeepRLI, an interaction prediction framework that is universally applicable across various tasks, leveraging a multi-objective strategy. Innovatively, this work proposes a multi-objective learning strategy that includes scoring, docking, and screening as optimization goals. This allows the deep learning model to have three relatively independent downstream readout networks, which can be optimized separately to enhance the task specificity of each output. The model incorporates an improved graph transformer with a cosine envelope constraint, integrates a novel physical information module, and introduces a new contrastive learning strategy. With these designs, DeepRLI demonstrates superior comprehensive performance, accommodating applications such as binding affinity prediction, binding pose prediction, and virtual screening, showcasing its potential in practical drug development.

The architecture of DeepRLI is illustrated in Figure 1.

![Architecture][architecture]

<div align="center"><b>Figure 1.</b> Schematic representation of the DeepRLI architecture</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

Python virtual environment needs to be created in advance.
* Import from the `.yml` file
  ```sh
  conda env create -n deeprli -f environment.yml
  ```
* Create step by step
  ```sh
  conda create -n deeprli python=3.11
  conda activate deeprli
  conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
  conda install -c dglteam/label/cu118 dgl==1.1.2.cu118
  conda install -c conda-forge rdkit==2023.09.2
  ```

### Installation

1. Clone the repository
   ```sh
   git clone https://github.com/fairydance/DeepRLI.git
   ```
2. Set the environment variable
   ```sh
   export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH}"
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### Preprocessing

Whether it is training or inference, structured data needs to be preprocessed into graph data. Executing preprocessing task first requires building a preset directory structure as follows:
```
${DATA_ROOT_DIR}
├── index
└── raw
```
The directory `raw` should contain the structure files of ligands and proteins. And the `index` directory provides index files for the data to be processed.

The script for this job is deposited in the `${REPO_ROOT}/src/deeprli/preprocess` directory. Run it as below:
```sh
python preprocess.py\
  --data-root "${DATA_ROOT_DIR}"\
  --data-index "${DATA_INDEX_FILE}"\
  --ligand-file-types "sdf,mol2,pdb"\
  --dist-cutoff 6.5
```
The path of `${DATA_INDEX_FILE}` in the above command is relative to the `${DATA_ROOT_DIR}`. After executing, the processed data will be stored in the location `${DATA_ROOT_DIR}/processed`. In addition, this script will also output an index file at the end that stores the successfully processed data.

It should be noted that the `${DATA_INDEX_FILE}` can only contain the processed data for the subsequent training or inference task. If there exist items reside in the `${DATA_INDEX_FILE}` but cannot be found in the `${DATA_ROOT_DIR}/processed` folder, the data preprocessing will be re-executed.

### Training

The script for model training is in the `${REPO_ROOT}/src/deeprli/train` directory. It can not only provide the necessary input with command line parameters, but also obtain the corresponding input by reading the json-formatted configuration file, and the former has a higher priority.
```sh
python train.py --config "${CONFIG_FILE}"
```
An example of the configuration file is as follows:
```json
{
  "train_data_root": "${TRAIN_DATA_ROOT}",
  "train_data_index": "${TRAIN_DATA_INDEX}",
  "train_data_files": "${TRAIN_DATA_FILES}",
  "epoch": 1000,
  "batch": 6,
  "initial_lr": 0.0002,
  "lr_reduction_factor": 0.5,
  "lr_reduction_patience": 15,
  "min_lr": 1e-6,
  "weight_decay": 0,
  "f_dropout_rate": 0.0,
  "g_dropout_rate": 0.0,
  "hidden_dim": 64,
  "num_attention_heads": 8,
  "use_layer_norm": false,
  "use_batch_norm": true,
  "use_residual": true,
  "gpu_id": 0,
  "enable_data_parallel": false,
  "use_all_train_data": true,
  "save_path": "${SAVE_PATH}"
}
```


### Inference

The script for the inference task is in the `${REPO_ROOT}/src/deeprli/infer` directory. It possesses a parameter input method similar to the above training script.
```sh
python infer.heavy.py --config "${CONFIG_FILE}"
```
An example of the configuration file is as follows:
```json
{
  "data_root": "${DATA_ROOT}",
  "data_index": "${DATA_INDEX}",
  "batch": 30,
  "gpu_id": 0,
  "save_path": "${SAVE_PATH}"
}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ALL IN ONE -->
## All in One

To facilitate users, we also provide an all-in-one script. Based on this, users only need to provide the necessary inputs, and the program can automatically complete data processing and all subsequent procedures to obtain the desired results, eliminating the need for manual step-by-step processing.

### Inference AIO

The integrated script for inference, `aio/infer_aio.receptor_in_pdb_and_ligands_in_sdf.py`, enables users to complete the entire process of data processing and inference with a single click by only providing a protein file in PDB format, a small molecule file in SDF format, a known ligand file in SDF format, and a trained model parameter file. The known ligand file contains the ligands already discovered for the target under study, and the program will truncate the binding pocket based on it. An example of executing the script is as follows:

```sh
python aio/infer_aio.receptor_in_pdb_and_ligands_in_sdf.py\
  --receptor-file-path "${RECEPTOR_FILE_PATH}"\
  --ligand-file-path "${LIGAND_FILE_PATH}"\
  --known-ligand-file-path "${KNOWN_LIGAND_FILE_PATH}"\
  --model "${MODEL}"\
  --save-path "${SAVE_PATH}"
```

If these options are not provided, the script will automatically read `receptor.pdb`, `ligands.sdf`, `known_ligands.sdf` and `trained_model.state_dict.pth` in the current directory as inputs. More details about the input options can be viewed through the output of the `--help` option.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

- Haoyu Lin (developer) - hylin@pku.edu.cn
- Jianfeng Pei (supervisor) - jfpei@pku.edu.cn
- Luhua Lai (supervisor) - lhlai@pku.edu.cn

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

We would like to express our gratitude to all members of Luhua Lai's group for their valuable suggestions and insights. We also acknowledge the support of computing resources provided by the high-performance computing platform at the Peking-Tsinghua Center for Life Sciences, Peking University.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[architecture]: img/architecture.svg

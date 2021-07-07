# Environment setting
## 1. Set the env path
In file "env_path", replace "/directory-to-current-folder" with the directory of this folder. For example, "/user/project/human_settlement_segmentation"

## 2. Install conda (conda 4.10.1, where the codes are tested)
[Please refer to the anaconda documentation](https://docs.anaconda.com/anaconda/install/)

## 3. Create the conda env
```bash
conda env create -f environment.yml #Create env from yml file
conda activate settlement_env #activate the conda env
```

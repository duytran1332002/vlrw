# Set up
Install all the neccessary library by the code below:
```
pip install -r requirements.txt
```
# Using GPU
## Install with Conda
You can use conda environment by the code below
```
# Install Miniconda
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Install torch
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

## Install with pip
You can use pip to install torch using gpu
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```
You can see detail herer: https://pytorch.org/get-started/locally/

# Using CPU
## Install with conda
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
## Install with pip
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```

### Setup Conda Python

sudo apt update
# install required dependencies
sudo apt install -y wget bzip2

# download miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# verify download
sha256sum Miniconda3-latest-Linux-x86_64.sh

# run installer
bash Miniconda3-latest-Linux-x86_64.sh

# activate conda
source ~/.bashrc
conda init

# check version
conda --version

# change the base name
conda update -n base -c defaults conda

# Create python environments
conda create --name myenv python=3.10
conda create --name langchain python=3.11
conda create --name autogen python=3.11
conda create --name automation python=3.11

# list the environments
conda env list

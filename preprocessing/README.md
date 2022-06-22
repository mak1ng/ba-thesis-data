# Preprocessing

This directory includes the files used for preprocess the 78 modules used in the evaluation for Pynguin.

`results_78.csv` contains the results of the preprocessing. It lists the number of made predictions and skipped types for each module.

`preprocessing` includes the whole data used for preprocessing except the Deeptyper models. Please insert the two model files (for example `model-5.cntk` and `model-5.cntk.ckp`) inside this folder if you want to recreate the results.
In addition to the preprocessing scripts it also contains a version of [autoimport](https://github.com/lyz-code/autoimport) modified to fit the tasks.

To reproduce the results create the docker image and run the following commands:

```
docker build -t preprocessing . -f ./Dockerfile --no-cache

docker run -it preprocessing:latest bash

mv preprocessing/ root/preprocessing
pip install virtualenv
conda activate deeptyper_3.6
conda install openmpi
apt install git
pip install cntk
pip install requests
pip install libcst
pip install pygments
export LD_LIBRARY_PATH=$HOME/conda/envs/deeptyper_3.6/lib
ln -s $HOME/conda/pkgs/openmpi-4.0.2-hb1b8bf9_1/lib/libmpi_cxx.so.40 $HOME/conda/envs/deeptyper_3.6/lib/libmpi_cxx.so.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/conda/lib/
conda deactivate 

cd root/preprocessing/
ls -la is_type_valid.sh
bash preprocess_project.sh -n 5
```

After the process is finished `results_invalid.csv` includes the skipped-types-statistics and `../../projects` include the files used for Pynguin.

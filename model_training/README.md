# Model Training

The model training and evaluation is based on the work of [DeepTyper](https://github.com/DeepTyper/DeepTyper)

All of these Python commands are executed in Python 3.6, recommended is the use of an anaconda environment.

There should be a directory called `models` containing 10 trained models, but as they need to much space, you have to download it from [Google Drive](https://drive.google.com/drive/folders/1VsOmzFGH3Jo6t8n_QZqM_iCYTrGujbGi?usp=sharing) and insert it manually. 

`data` includes folders `outputs-all` and `outputs-gold` which contain the data for training and testing. Furthermore the token and type vocabulary files and files including information about training, validation and testdata.

To retrain the models follow the instructions in [DeepTyper Repository](https://github.com/DeepTyper/DeepTyper#running-deeptyper), paragraph *Running DeepTyper*. Use the files in this repository, because they are adapted to Python. 

To evaluate a model against the testdata run `evaluation.py` which will create a file including the results. It takes some time.

```
python evaluation.py
```
If you do not want to wait, there is already a results-file for the evaluation of `model-5`, called `evaluation-true-all_07.03_model5.txt`.

After that run `eval_model_results.py`, to get the accuracy, precision and recall values (for the 10 most seen types in training and overall) printed out to console.

```
python eval_model_results.py evaluation-true-all_07.03_model5.txt
```

# rnn-comparative-analysis-sentiment
Comparative analysis of RNN architectures for sentiment classification


### Installing prerequisites - run the below commands from within the project folder
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

### Running the training and evaluation scripts
Directly run the main.py file. The main python file trains the preprocesses the data, craeates config combinations, trains the combination modes

### File structure
main.py - main runner file that calls for preprocess, training and evaluation of datasets
utils.py - variation configurations and dataloader helper functions
models.py - Flexible neural network for RNN, LSTM and BiLSTM
preprocess.py - reads, cleans, tokenizes, pads and sequences the IMDB dataset
train.py - training runner where most of the training code is called



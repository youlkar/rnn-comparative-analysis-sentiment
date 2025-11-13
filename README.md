# rnn-comparative-analysis-sentiment
Comparative analysis of RNN architectures for sentiment classification


### Installing prerequisites - run the below commands from within the project folder
python -m venv .venv <br>
.\.venv\Scripts\Activate.ps1<br>
pip install -r requirements.txt<br>

### Running the training and evaluation scripts
Directly run the main.py file. The main python file trains the preprocesses the data, craeates config combinations, trains the combination modes

### File structure
main.py - main runner file that calls for preprocess, training and evaluation of datasets<br>
utils.py - variation configurations and dataloader helper functions<br>
models.py - Flexible neural network for RNN, LSTM and BiLSTM<br>
preprocess.py - reads, cleans, tokenizes, pads and sequences the IMDB dataset<br>
train.py - training runner where most of the training code is called<br>



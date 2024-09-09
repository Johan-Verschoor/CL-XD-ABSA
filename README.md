# DAT-LCR-Rot-hop++

Cross-Domain (CD) Aspect-Based Sentiment Classification (ABSC) using LCR-Rot-hop++ with Domain Adversarial Training (DAT).

# Project Setup Instructions

1. **Install Python 3.7**: Ensure you have Python 3.7 installed, as newer versions are incompatible with the required packages. Download Python from [here](https://www.python.org/downloads/).
2. **Install Anaconda**: Download and install Anaconda from [this link](https://www.anaconda.com/products/individual).

## Virtual Environment Setup
1. **Create a Virtual Environment**:
   - Open Anaconda and create a virtual environment using Python 3.7:
     ```bash
     conda create --name_of_env python=3.7
     ```
   - Activate the environment:
     ```bash
     conda activate name_of_env
     ```
     
2. **Install Dependencies**:
   - Run the following command to install required packages (including `protobuf 3.19` and `tensorflow 1.15`):
     ```bash
     pip install -r requirements.txt
     ```

3. **Copy Project Files**:
   - Copy all files from this repository into your virtual environment directory.
    
## How to use?
- Adjust the paths in config.py, main$\_$test.py, and main$\_$hyper.py 
- Get raw data for your required domains by running raw$\_$data.py for restaurant, laptop, and book domain.
- Get BERT embeddings by running generate_bert_embeddings.ipynb for your required domains *using Google Colab* to obtain BERT embeddings or download them from https://drive.google.com/drive/folders/10QzWzfGQnAXwdSNUp16241QHHWcKYILn?usp=sharing and put them in correct folder: "\getBERT\dataBERT". 
- Run prepare_bert.py for your required domains.
- Tune hyperparameters to your specific task using main_hyper.py or use hyperparameters as pre-set in main_test.py.
- Adjust the additional settings in config.py. For instance, choosing to add neutral sentiments to positive class
- Adjust in nn_layer.py the structure of the discriminator if wanted
- Select tests to run and run main_test.py (running all tests will take a long time, 5-8 minutes per iteration). Make
  sure write_result is set to True if you want the results to be saved to a text file.

## References.

This code is adapted from Trusca, Wassenberg, Frasincar and Dekker (2020).

https://github.com/mtrusca/HAABSA_PLUS_PLUS

Truşcǎ M.M., Wassenberg D., Frasincar F., Dekker R. (2020) A Hybrid Approach for Aspect-Based Sentiment Analysis Using
Deep Contextual Word Embeddings and Hierarchical Attention. In: 20th International Conference on Web
Engineering. (ICWE 2020). LNCS, vol 12128, pp. 365-380. Springer, Cham.
https://doi.org/10.1007/978-3-030-50578-3_25

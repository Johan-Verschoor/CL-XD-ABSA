# CL-XD-ABSA

Cross-Learning (CL) Cross-Domain (XD) using LCR-Rot-hop++ with Domain Adversarial Training for Aspect-Based Sentiment Analysis (ABSA).

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
    
## How to Use

1. **Set Up Paths**: Create the necessary paths in your virtual environment as specified in `config.py`, `main_test.py`, and `main_hyper.py`.

2. **Generate Raw Data**: Run `raw_data.py` to obtain the raw data for your required domains (restaurant, laptop, and book).

3. **Get BERT embeddings**: Run `bert_prepare.py` to obtain the raw data for your required domains (restaurant, laptop, and book). Place the files in the embedding directories, and rename them if necessary.

4. **Tune hyperparameters**: Run `main_hyper.py` to  find the optimal hyperparameter settings. `main_test.py` already contains the optimal hyperparameters for the base model as found by the authors.

5. **Adjust additional settings**: Changing settings in `config.py`, `main_test.py` or `main_hyper.py` allows for running the model with other settings for e.g. epochs, adding or leaving out neutral sentiment, etc..

5. **Adjust discriminator structure**: `nn_layer.py` can be used to change the structure of the discriminator.

6. **Run the model**: Fill `main_test.py` with the hyperparameters of choice and run the model for a given amount of epochs. Results will be stored in Result_Files, including runtime, accuracy per sentiment polarity, train accuracy and general (maximum) test accuracy.


## References.

This code is adapted from Knoester, Frasincar and Truşca. (2022).

https://github.com/jorisknoester/DAT-LCR-Rot-hop-PLUS-PLUS/

Knoester, J., Frasincar, F., and Truşca, M. M. (2022). Domain adversarial training for aspect-
based sentiment analysis. In 22nd International Conference on Web Information Systems
Engineering (WISE 2022), volume 13724 of LNCS, pages 21–37. Springer.



The work of Knoester et al. is an extension on the work of Trusca, Wassenberg, Frasincar and Dekker (2020).

https://github.com/mtrusca/HAABSA_PLUS_PLUS

Truşcǎ M.M., Wassenberg D., Frasincar F., Dekker R. (2020) A Hybrid Approach for Aspect-Based Sentiment Analysis Using
Deep Contextual Word Embeddings and Hierarchical Attention. In: 20th International Conference on Web
Engineering. (ICWE 2020). LNCS, vol 12128, pp. 365-380. Springer, Cham.
https://doi.org/10.1007/978-3-030-50578-3_25

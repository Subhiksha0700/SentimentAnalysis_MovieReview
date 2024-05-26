# NLP Project: Sentiment Analysis on Movie Reviews

## Table of Contents

- [Project Overview](#project-overview)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Feature Engineering and Vectorization Techniques](#feature-engineering-and-vectorization-techniques)
    - [Classical Model with Vectorization](#classical-model-with-vectorization)
    - [Classical Model by Creating Feature Sets](#classical-model-by-creating-feature-sets)
    - [Deep Learning Model](#deep-learning-model)
- [Python Environment Setup](#python-environment-setup)
- [How to Access Notebooks](#how-to-access-notebooks)
- [Contributors](#contributors)

## Project Overview

Sentiment analysis is a critical area of natural language processing that involves understanding the emotional tone behind a body of text. This is particularly useful in analyzing movie reviews, where understanding sentiment can help gauge the overall reception of a film.

This project details the execution of sentiment analysis on a dataset of movie reviews through three distinct methodologies: classical models with vectorization, classical models with manually created feature sets, and deep learning models.

## Methodology

### Data Preprocessing

Data preprocessing is crucial to prepare raw text for further analysis and model training. The preprocessing steps implemented across all models include:

- **Tokenization**: Splitting text into individual words or tokens.
- **Removal of Noise**: Stripping away unnecessary characters such as punctuation and numbers.
- **Case Normalization**: Converting all tokens to lower case.
- **Stopword Removal**: Eliminating common words that might dilute the predictive power of important words.
- **Lemmatization**: Reducing words to their base or root form.
- **Addressing Class Imbalance**: Applying manual balancing techniques to ensure equal representation of sentiment classes.

### Feature Engineering and Vectorization Techniques

#### Classical Model with Vectorization

- **Vectorization**:
  - **Bag-of-Words (BoW)**: Transforms text into a fixed-length set of features, with each feature representing the count of a specific word in the text. Both unigrams and bigrams are used.
  - **Implementation**: Utilized `CountVectorizer` from scikit-learn.
  - **GloVe (Global Vectors for Word Representation)**: Converts words into dense vectors of fixed size where semantically similar words are mapped to similar points in the vector space. This captures more information per word than one-hot encoding or BoW. 
  - **Implementation**: Pre-trained GloVe embeddings are used to represent words in the dataset.
- **Model Implementation with Logistic Regression**:
  - Two variations: default settings and hyperparameter tuning using `GridSearchCV`.
- **Evaluation via Cross-Validation**: Ensures the model's effectiveness and generalizability.

#### Classical Model by Creating Feature Sets

- **Custom Feature Sets**:
  - **Subjectivity Lexicon**: Words labeled with their polarity and strength.
  - **Presence of Words**: Binary feature for each word.
  - **Part-of-Speech Tags**: Categorized using NLTK's POS tagger.
- **Model Implementation with Random Forest Classifier**:
  - Handles sparse and high-dimensional data.
  - Evaluated with custom cross-validation methods.

#### Deep Learning Model

- **Neural Network Configuration with Embeddings and LSTM**:
  - **Embedding Layer**: Converts words into dense vectors.
  - **Bidirectional LSTM Layers**: Captures forward and backward information.
  - **Regularization Techniques**: Dropout and batch normalization.
  - **Training and Optimization**: Trained with the Adam optimizer, dynamically adjusting the learning rate.

## Python Environment Setup

To set up the Python environment for this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/MovieReview_SentimentAnalysis.git
   ```

2. **Navigate to the project directory**:
    ```bash
    cd MovieReview_SentimentAnalysis
    ```
3. **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    ```
   
4. **Activate the virtual environment**:
   - On Windows:
       ```bash
       venv\Scripts\activate
       ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

5. **Install the required packages**:

    ```bash
     pip install -r requirements.txt
     ```

## How to Access Notebooks

The project consists of multiple Jupyter notebooks. To access and run these notebooks:

1. **Start Jupyter Notebook**:
    ```bash
     jupyter-notebook
     ```

2. **Open the notebooks**:

    - Final_NLP_Project.ipynb
    - NLP_Project_Vectorization.ipynb
    - NLP_Project_Feature_Set.ipynb
    - NLP_Project_Glove.ipynb
    - Nlp_movie_review.ipynb

## Contributors
- Subhiksha Murugesan
- Nithish Kumar Senthil Kumar
- Nagul Pandian
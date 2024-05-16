# Web Mining Project - Team 1
Hate speech and offensive language detection on the Twitter dataset.

**Dataset source:** https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset

## About the project
1. **Data Preprocessing:**
Utilize Python's Pandas library to preprocess and clean the dataset, including handling missing values, removing duplicates, and any other necessary preprocessing steps.
 
2. **Data Visualization:**
Use Matplotlib library to visualize various aspects of the preprocessed dataset. This includes but is not limited to:
* Distribution of classes or labels.
* Word frequency distribution.
* Visualization of any relevant patterns or trends.
 
3. **Text Tokenization, Stemming, and Lemmatization:**
* Utilize the Natural Language Toolkit (NLTK) library for tokenization, stemming, and lemmatization of the text data.
* Tokenization: Breaking down the text into individual words or tokens.
* Stemming: Reducing words to their root form to normalize variations.
* Lemmatization: Converting words to their base or dictionary form to further normalize the text.
 
4. **Feature Extraction:**
* Use TF-IDF (Term Frequency-Inverse Document Frequency) and Count Vectorizer techniques to convert the preprocessed text data into numerical features.
* TF-IDF is used to reflect the importance of a term in a document relative to a collection of documents.
* Count Vectorizer converts text documents to a matrix of token counts.
 
5. **Model Training for Sentiment Analysis:**
* Train multiple Naive Bayes models for sentiment analysis using the extracted features.
* Experiment with different variations of Naive Bayes models to optimize performance.
* Perform classification for sentiment analysis across three different classes/categories present in the dataset.
 
6. **Deep Learning and LLM Experiments:**
* Conduct experiments with various deep learning architectures, such as recurrent neural networks (RNNs), or transformer-based models (e.g., DistilBERT).
* Train these models on the preprocessed text data to achieve better accuracy scores compared to the Naive Bayes models.
* Fine-tune hyperparameters and architectures based on performance metrics.

# Fake-News-Detection
To predict whether the given news is fake or real

### Abstract

The rapid growth of online news platforms and social media has made it easier than ever to access information. However, it has also increased the spread of fake news, which can mislead the public, influence opinions, and create social and political issues. Detecting fake news manually is time-consuming and often impractical due to the large volume of information shared every day. Similarly, analyzing customer reviews to determine public opinion is another important text classification task that helps businesses understand customer satisfaction and improve their products and services. These challenges highlight the need for intelligent machine learning models capable of automatically classifying textual information with high accuracy.

The motivation behind this project is to develop an automated system that can accurately identify fake news while also comparing its performance with customer review sentiment prediction. Such systems can assist media organizations, researchers, and businesses in making informed decisions by reducing misinformation and extracting meaningful insights from textual data. By evaluating multiple machine learning algorithms on both tasks, the study aims to identify the most effective classification techniques for different types of text analysis.

This project proposes a machine learning-based approach for fake news prediction using five classification algorithms: **K-Nearest Neighbors (KNN), Naïve Bayes, Decision Tree, Random Forest, and Support Vector Machine (SVM)**. Before training, the textual data undergoes preprocessing, including text cleaning, tokenization, stop-word removal, and feature extraction. The performance of each algorithm is evaluated and compared using prediction accuracy. For fake news classification, **Naïve Bayes, Decision Tree, and Random Forest achieved the highest accuracy of 100%**, followed by **SVM with 95%** and **KNN with 85%**. To further evaluate the effectiveness of these algorithms, the same models were applied to customer review sentiment prediction. The comparison showed that sentiment classification achieved lower accuracies, with **Random Forest obtaining the highest accuracy of 74%**, followed by **SVM (73%)**, **Naïve Bayes (70%)**, **KNN (63%)**, and **Decision Tree (62%)**. The results indicate that fake news classification is more effectively learned by these models than customer review sentiment prediction, likely because sentiment analysis involves more subjective and context-dependent language.

The main objectives of this project are to develop an effective fake news prediction system using machine learning algorithms, compare the performance of different classifiers, evaluate their effectiveness on both fake news detection and customer review sentiment prediction, and identify the algorithm that provides the best overall performance. The project demonstrates that machine learning techniques can accurately detect fake news while also highlighting the differences in classification performance across related natural language processing tasks.

# Module Description

### Module 1: Dataset Import

Imports the fake news and customer review datasets into the system for training and evaluation.

### Module 2: Data Preprocessing

Prepares the textual data by removing unwanted characters, converting text to lowercase, removing stop words, and performing stemming to improve data quality.

### Module 3: Feature Extraction

Converts the cleaned text into numerical features using text vectorization techniques, allowing machine learning algorithms to process the textual data.

### Module 4: Train-Test Split

Divides the dataset into training and testing sets to train the models and evaluate their performance on unseen data.

### Module 5: Machine Learning Model Development

Develops five machine learning models for text classification:

* K-Nearest Neighbors (KNN)
* Naïve Bayes
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)

### Module 6: Model Training

Trains each machine learning algorithm using the training dataset to learn patterns that distinguish fake news from real news.

### Module 7: Fake News Prediction

Uses the trained models to classify news articles as either **Fake** or **Real** and compares the prediction results of all five algorithms.

### Module 8: Customer Review Sentiment Prediction

Applies the same machine learning algorithms to classify customer reviews based on sentiment and compares their performance with fake news prediction.

### Module 9: Performance Evaluation

Evaluates the performance of each algorithm using classification accuracy. The comparison shows that **Naïve Bayes, Decision Tree, and Random Forest achieved 100% accuracy for fake news prediction**, while **Random Forest achieved the highest sentiment prediction accuracy of 74%**.

### Module 10: Comparative Analysis

Compares the performance of all five machine learning algorithms on both fake news detection and customer review sentiment prediction, identifying the strengths and limitations of each model for different text classification tasks.

Algorithm 	Accuracy Fake News	Accuracy Sentiment	Accuracy Heart Disease


<img width="668" height="153" alt="Screenshot 2026-07-09 at 9 18 24 PM" src="https://github.com/user-attachments/assets/30dfbbf1-d6d6-4b1f-80a0-79bba9d7d418" />


<img width="416" height="97" alt="image" src="https://github.com/user-attachments/assets/4339266e-9327-4170-bcb8-dee5bd9dbdad" />

<img width="361" height="218" alt="image" src="https://github.com/user-attachments/assets/27496008-a105-4216-9eb1-8ae1392d2826" />
<img width="2940" height="1912" alt="image" src="https://github.com/user-attachments/assets/7a0a9cd1-b540-4f70-91af-bfa7b7c03355" />

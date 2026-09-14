# Naive-Bayes-Text-Classification-MLE-vs-MAP

Overview

This project implements a Naive Bayes text classifier from scratch using NumPy and evaluates the difference between Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP) estimation.

The model is trained and tested on the 20 Newsgroups dataset, using four categories:

alt.atheism
soc.religion.christian
comp.graphics
sci.med

The project also compares different Dirichlet priors and smoothing techniques to understand their effect on classification accuracy.

Technologies Used
Python
NumPy
Scikit-learn
20 Newsgroups Dataset
Methodology
1. Data Loading

The 20 Newsgroups dataset is loaded using Scikit-learn. Headers, footers, and quoted text are removed to reduce unwanted information.

2. Text Vectorization

CountVectorizer converts text documents into numerical word-count vectors.

English stop words are removed.
A maximum of 5,000 features is used.
3. Class Priors

The prior probability of each class is calculated from the training data:

𝑃
(
𝑐
)
=
documents in class 
𝑐
total documents

4. MLE

The word probability for each class is calculated using Maximum Likelihood Estimation:

𝑃
(
𝑤
∣
𝑐
)
=
𝑁
𝑐
,
𝑤
∑
𝑤
𝑁
𝑐
,
𝑤

5. MAP Estimation

MAP estimation adds prior information to the observed word counts:

𝑃
(
𝑤
∣
𝑐
)
=
𝑁
𝑐
,
𝑤
+
𝛼
𝑤
−
1
∑
𝑤
(
𝑁
𝑐
,
𝑤
+
𝛼
𝑤
−
1
)

The following priors are compared:

Lidstone smoothing: α = 1.01
Laplace smoothing: α = 2.0
Strong Dirichlet prior: α = 10.0
Non-uniform empirical prior
6. Prediction

Log probabilities are used instead of direct probabilities to avoid numerical underflow:

log
⁡
𝑃
(
𝑐
∣
𝑥
)
∝
log
⁡
𝑃
(
𝑐
)
+
∑
𝑤
𝑥
𝑤
log
⁡
𝑃
(
𝑤
∣
𝑐
)

The class with the highest score is selected as the prediction.

7. Evaluation

The predictions are evaluated using classification accuracy.

Project Structure
naive-bayes-mle-map/
│
├── naive_bayes.py
├── README.md
└── requirements.txt

Example Output
[MLE] Test Accuracy: 0.xxxx

Evaluating MAP Predictions:
-------------------------------------------------------
Lidstone Smoothing (Alpha = 1.01) | Test Accuracy: 0.xxxx
Laplace Smoothing (Alpha = 2.0)   | Test Accuracy: 0.xxxx
Strong Dirichlet Prior (Alpha = 10.0) | Test Accuracy: 0.xxxx
Non-Uniform Prior (Empirical Basis) | Test Accuracy: 0.xxxx

Key Learning

This project demonstrates how smoothing and prior distributions affect Naive Bayes text classification.

MLE relies entirely on observed training data, while MAP incorporates prior information to handle zero or unreliable word probabilities.

How to Run

Install the required libraries:

pip install numpy scikit-learn


Then run:

python naive_bayes.py

Conclusion

The project provides a practical implementation of MLE and MAP estimation for Multinomial Naive Bayes, showing how different prior assumptions can influence model performance on real-world text classification data.

# IMDb Movie Review Sentiment Analysis

This project uses the **IMDb Movie Review Dataset** (50,000 reviews, evenly split into positive and negative labels) to build and evaluate multiple **traditional machine learning models** for **sentiment classification**.

---

## Approach

### 1. Preprocessing
- Converted text to lowercase  
- Removed HTML tags, punctuation, and stopwords  
- Tokenized words  
- Converted text into **TF-IDF features**

### 2. Models Tried
We experimented with **five traditional ML models**:  
1. Logistic Regression  
2. Linear SVM (Support Vector Machine)  
3. Multinomial Naive Bayes  
4. Random Forest Classifier  
5. Gradient Boosting Classifier  

### 3. Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix Visualization  
- Learning Curves

---

## Results

| Model                    | Accuracy | Precision | Recall | F1-score |
|---------------------------|----------|-----------|--------|----------|
| Logistic Regression       | **0.8896** | 0.8784    | 0.9063 | **0.8922** |
| Linear SVM                | 0.8771   | 0.8696    | 0.8895 | 0.8794   |
| Multinomial Naive Bayes   | 0.8525   | 0.8522    | 0.8557 | 0.8539   |
| Random Forest             | 0.8509   | 0.8623    | 0.8379 | 0.8499   |
| Gradient Boosting         | 0.8086   | 0.7789    | 0.8660 | 0.8201   |

---

## Visualizations

### 1. Confusion Matrices
Each model’s confusion matrix was plotted to see misclassification patterns.  
- Logistic Regression & Linear SVM → fewer false predictions  
- Naive Bayes → balanced errors  
- Tree-based models → slightly worse generalization  

### 2. Learning Curves
- Logistic Regression: Strong performance, small gap → slight overfitting but stable  
- Linear SVM: Similar trend to Logistic Regression  
- Naive Bayes: Curves converge quickly, less prone to overfitting  
- Random Forest / Gradient Boosting: Training accuracy high but validation lower → some overfitting  

---

## Key Takeaways
- **Best Model:** Logistic Regression (Accuracy ≈ 89%, F1-score ≈ 0.89)  
- Linear models (Logistic Regression, SVM) outperform tree-based models for sparse TF-IDF features.  
- Naive Bayes is lightweight and performs well but slightly worse than linear models.  
- Adding deep learning models (LSTMs, CNNs, Transformers) could further improve performance, but traditional ML already gives strong results.


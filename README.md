# Breast Cancer morality prediction project
A machine learning project for predicting breast cancer mortality status and survival months using clinical data.

## Project Overview
This project develops predictive models to assess breast cancer prognosis through two main tasks:

- **Classification:** Predicting mortality status (survived/deceased)

- **Regression:** Estimating survival months for patients

## DataSet & Preprocessing
- **Dataset:** Clinical breast cancer patient data

- **Preprocessing Steps:**
     - Handled missing values and outliers
     - Feature scaling and normalization
     - Data splitting (train/test: 80/20)
 
## Models Implemented
### Classification Models (Morality Prediction)

|Model | Accuracy	| Notes
|-----|:--------:|------:|
|K-Nearest Neighbors|	0.84 → 0.90 |	Optimized with GridSearchCV
|Logistic Regression|	0.82|	Baseline performance
|Naive Bayes|	0.79 |	Comparative model
|Ensemble (Soft Voting) |	0.90 |	KNN + Logistic Regression + NB

### Regression Models (Survival Months)
|Model | MSE	| Notes
|-----|:--------:|------:|
|Decision Tree (pruned)	| 1.57 | Better generalization
|Decision Tree (Unpruned)| 2.21| Higher variance

## Model Training & Optimization
### Hyperparameter Tuning
- KNN: Optimized n_neighbors, weights, and metric using GridSearchCV
- Performance Boost: Accuracy improved from 0.84 to 0.90 (7% increase)
### Ensemble Learning
- Voting Classifier: Soft voting mechanism
- Combined Models: Best-performing KNN + Logistic Regression + Naive Bayes
- Result: Maintained 0.90 accuracy with improved robustness
## Key Results
### Classification Performance
- Best Individual Model: Hyperparameter-tuned KNN (90% accuracy)
- Ensemble Model: 90% accuracy with enhanced stability
- Significant improvement over baseline models
### Regression Performance
- Pruned Decision Tree: MSE = 1.57 (better for deployment)
- Unpruned Decision Tree: MSE = 2.21 (higher accuracy but overfit risk)
- Trade-off: Pruning reduces overfitting while maintaining reasonable error
## Technical Stack
- Programming Language: Python 3.x
- Libraries: Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
- Tools: Jupyter Notebook, GridSearchCV, VotingClassifier

## Key Insights
1.Hyperparameter tuning significantly improves model performance (KNN: +7% accuracy)

2.Ensemble methods provide robust predictions without sacrificing accuracy

3.Model pruning offers better generalization for regression tasks

4.Medical prognosis can be effectively predicted using machine learning approaches

## 👨‍💻 Author
*Developed as part of college coursework in Machine Learning.*






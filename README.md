# CS362-Race-Prediction
# Race Time Prediction Using Machine Learning
**CS 362 – Artificial Intelligence and Machine Learning | Spring 2026**  
**Group Project – Supporting Document**

---

## Overview

This project explores how machine learning can be applied to athletic performance data to predict future race times. Specifically, we use a **supervised learning** approach — training a model on historical running training data so it can estimate how fast a runner will compete in an upcoming race. The topic sits at the intersection of sports science and artificial intelligence, making it both practically relevant and approachable for demonstrating core ML concepts.

Our lesson teaches the class how a machine learning pipeline works end-to-end: from collecting and structuring data, to training a model, to evaluating and interpreting predictions. Rather than treating ML as a black box, we break down exactly what the algorithm is doing at each step.

---

## Key Concepts and Terminology

**Supervised Learning**  
Supervised learning is a category of machine learning in which a model is trained on labeled data — meaning each input example comes paired with a known correct output. The model learns to map inputs to outputs so it can later predict outputs for new, unseen inputs. In our project, the inputs are training metrics (weekly mileage, average pace, previous race time) and the output is the runner's next race time.

**Features (Input Variables)**  
Features are the measurable properties used as inputs to a machine learning model. We use three features: weekly mileage (how many miles the runner logged), average training pace (minutes per mile), and most recent race time. These were selected because they are commonly tracked by runners and are known to correlate with race performance.

**Label (Target Variable)**  
The label is what the model is trying to predict — in this case, the runner's next 5K race time in minutes. During training, the model uses known labels (actual past race results) to learn the relationship between features and outcomes.

**Linear Regression**  
Linear regression is a foundational machine learning algorithm that models the relationship between input features and a continuous output variable as a linear equation. The model learns a set of weights (coefficients) — one for each feature — and a bias term (intercept). Once trained, it combines the feature values with those weights to produce a prediction. It is one of the most interpretable ML algorithms because the equation it produces can be read and understood directly.

**Training**  
Training is the process by which the model adjusts its internal parameters (weights) to minimize prediction error on the training dataset. The algorithm iterates through the data, compares its predictions to the actual labels, and updates the weights to reduce the difference. In scikit-learn's `LinearRegression`, this is solved analytically in a single step using the ordinary least squares method.

**Prediction**  
After training, the model can take new input data it has never seen and produce an estimated output. This is the core use case: given a runner's current training metrics, the model outputs an estimated race time.

**Model Evaluation**  
A model is only useful if its predictions are accurate enough for the intended application. Common evaluation metrics for regression models include Mean Absolute Error (MAE), which measures the average size of prediction errors in the same units as the output, and R² (coefficient of determination), which measures how well the model explains the variance in the target variable. A perfect model would have MAE of 0 and R² of 1.0.

**Overfitting and Data Limitations**  
Because our dataset is intentionally small (four rows), the model will fit the training data very closely but may not generalize well to new runners with very different profiles. This is a useful teaching moment: more data, more features, and proper train/test splitting all improve real-world model reliability.

---

## How Our Lesson Teaches the Topic

Our 20–30 minute lesson is structured to move students from intuition to implementation to reflection.

We open with a brief warm-up discussion asking students to predict what factors affect race performance. This activates prior knowledge and mirrors what data scientists do when selecting features — starting with domain understanding before touching any data. It also surfaces any misconceptions early.

From there, we introduce the concept of supervised learning with a simple analogy: a coach who reviews an athlete's training log and, based on past patterns, estimates what they will run next weekend. The model does the same thing, but mathematically. We show students the dataset — four rows of training data with three input columns and one output column — and ask them to spot the pattern before the algorithm does. Most students will correctly notice that more mileage and faster training pace correspond to faster race times.

We then walk through the Python code step by step, explaining what each line does conceptually rather than syntactically. The `model.fit()` call is the moment the model "learns." The `model.predict()` call is the moment it applies what it learned to new data. We explain the linear equation the model builds internally — how it assigns a weight to each feature based on how predictive that feature is — without requiring students to understand the underlying linear algebra.

The individual activity reinforces this by having students run the code themselves, record their predicted output, and answer analytical questions about the model's behavior. An extension challenge asks them to modify one input variable and explain how and why the prediction changes, which directly tests conceptual understanding.

We close with a discussion of real-world applications and limitations: professional sports teams using ML for injury prevention, pace planning in marathon racing, and the ethical considerations of using algorithmic predictions to make decisions about athletes.

---

## Real-World Applications

Machine learning for athletic performance prediction is an active and growing field. Several professional and semi-professional sports organizations now use ML models to guide training decisions, predict injury risk, and optimize race strategies. Running apps such as Garmin Coach and Nike Run Club use personalized prediction models trained on large user datasets to generate race time estimates and training recommendations. The 2023 Boston Marathon used pacing algorithms informed by historical data to help elite runners hit target splits. Beyond running, similar regression-based approaches are used in cycling power modeling, swimming stroke efficiency analysis, and load management in team sports.

---

## References

Bishop, C. M. (2006). *Pattern recognition and machine learning*. Springer.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

Vickers, A. J., & Vertosick, E. A. (2016). An empirical study of race times in recreational endurance runners. *BMC Sports Science, Medicine and Rehabilitation, 8*(1), 26. https://doi.org/10.1186/s13102-016-0052-y

Géron, A. (2022). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.

---

## Repository Structure

```
/
├── README.md              ← This file (supporting document)
├── /slides                ← PowerPoint presentation
├── /activity              ← Student handout (Word doc)
├── /code                  ← Python prediction script
└── /docs                  ← Additional reference materials
```

## Running the Code

Requirements: Python 3.8+, pandas, scikit-learn

```bash
pip install pandas scikit-learn
cd code
python predict.py
```

The script loads `data.csv`, trains a linear regression model, and outputs a predicted race time for a sample runner.

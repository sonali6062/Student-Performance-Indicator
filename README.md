## END TO END STUDENT PERFORMANCE INDICATOR PROJECT

## The Student Performance Indicator predicts student exam scores based on demographic, social, and academic features.This project demonstrates data preprocessing, feature engineering, model training, hyperparameter optimization, and deployment readiness.


🎓 Student Performance Indicator
📊 Predicting academic performance using Machine Learning

This project aims to predict student performance scores based on various demographic, social, and academic features using advanced regression models and hyperparameter tuning.

✅ Final Model R² Score: 87.77%

🧠 Project Overview

The goal of this project is to build a complete end-to-end Machine Learning pipeline — from data ingestion and transformation to model training, evaluation, and deployment — for predicting student performance.

🌟 Project Highlights

🔹 Built a complete end-to-end Machine Learning pipeline

🔹 Achieved R² Score of 87.77% using Gradient Boosting Regressor

🔹 Implemented 7 regression algorithms with GridSearchCV for hyperparameter tuning

🔹 Designed a Flask-based API for predictions

🔹 Fully modularized project with robust logging, exception handling, and artifact management


---

## 📁 Repository Structure

```
Student-Performance-Indicator/
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py         # Loads and splits the dataset
│   │   ├── data_transformation.py    # Preprocessing and feature scaling
│   │   ├── model_trainer.py          # Model training and evaluation
│   ├── pipeline/
│   │   ├── predict_pipeline.py       # For new predictions
│   ├── utils.py                      # Helper functions (save/load objects)
│   ├── exception.py                  # Custom exception handling
│   ├── logger.py                     # Logging setup
│
├── artifacts/                        # Saved models and preprocessors
├── reports/                          # Model performance reports
├── notebooks/                        # EDA and experimentation
├── requirements.txt
├── app.py                            # Flask app for prediction
└── README.md
```

---

## ⚙️ Workflow

1. Data Ingestion → Read dataset and split into train/test
2. Data Transformation → Encode, scale, and clean data
3. Model Training → Train multiple regression models
4. Hyperparameter Tuning → Optimize with `GridSearchCV`
5. Evaluation → Compare based on **R²**, **MAE**, **RMSE**
6. Model Saving → Save the best model & preprocessor in `artifacts/`
7. Prediction Pipeline → Load trained model to predict unseen data

---

## 🤖 Models Implemented

| Algorithm                      | Description                             | Tuned | Library      |
| ------------------------------ | --------------------------------------- | ----- | ------------ |
| 🌲 Random Forest Regressor     | Ensemble of Decision Trees              | ✅     | scikit-learn |
| 🌳 Decision Tree Regressor     | Simple tree-based model                 | ✅     | scikit-learn |
| 🚀 Gradient Boosting Regressor | Boosting with residual learning         | ✅     | scikit-learn |
| 📈 Linear Regression           | Baseline linear model                   | ❌     | scikit-learn |
| ⚡ XGBoost Regressor            | Gradient boosting with regularization   | ✅     | xgboost      |
| 🐱 CatBoost Regressor          | Boosting optimized for categorical data | ✅     | catboost     |
| 🔺 AdaBoost Regressor          | Ensemble boosting using weak learners   | ✅     | scikit-learn |

---

## 🧮 Hyperparameter Tuning

Each model was tuned with `GridSearchCV (cv=5)`:

```python

  models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
```

---

## 🏆 Model Performance

| Model                           | R² Score   | Status   |
| ------------------------------- | ---------- | -------- |
| Gradient Boosting Regressor     | 0.8777     | 🏆 Best  |
| Random Forest Regressor         | 0.8612     | ✅        |
| XGBoost Regressor               | 0.8564     | ✅        |
| CatBoost Regressor              | 0.8501     | ✅        |
| Decision Tree Regressor         | 0.8213     | ✅        |
| AdaBoost Regressor              | 0.8085     | ✅        |
| Linear Regression               | 0.7430     | Baseline |

> ✅ **Best Model Selected:** Gradient Boosting Regressor
> 📊 **Final R² Score:** 87.77%

---

## 📈 Metrics Used

| Metric       | Description                              |
| ------------ | ---------------------------------------- |
| **R² Score** | Measures variance explained by the model |
| **MAE**      | Mean Absolute Error                      |
| **MSE**      | Mean Squared Error                       |
| **RMSE**     | Root Mean Squared Error                  |

---

## 🧰 Tech Stack

| Category            | Tools                                          |
| ------------------- | ---------------------------------------------- |
| **Language**        | Python 3.8+                                    |
| **Libraries**       | Pandas, NumPy, Scikit-learn, XGBoost, CatBoost |
| **Deployment**      | Flask                                          |
| **Version Control** | Git & GitHub                                   |


---

## 🚀 Run Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/sonali6062/Student-Performance-Indicator.git
cd Student-Performance-Indicator
```

### 2️⃣ Create a Virtual Environment

```bash
conda create -p venv python==3.8 -y
conda activate venv/
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Train the Model

```bash
python src/components/model_trainer.py
```

### 5️⃣ Run the Flask App

```bash
python app.py
```

Then open your browser and navigate to:
👉 **[http://localhost:5000](http://localhost:5500)**

---

## 📦 Artifacts Generated

| File                             | Description                    |
| -------------------------------- | ------------------------------ |
| `artifacts/model.pkl`            | Final trained regression model |
| `artifacts/preprocessor.pkl`     | Transformation pipeline        |
| `reports/final_model_report.csv` | Model performance summary      |
| `logs/`                          | System & training logs         |

---

## 🧩 Key Learnings

* Building modular ML pipelines for reusability
  
* Performing hyperparameter tuning effectively using GridSearchCV
  
* Understanding bias-variance tradeoff in regression models
  
* Saving and loading ML artifacts for deployment

---

## 🛠️ Future Enhancements

* 📊 Add **feature importance** visualization
* 
* 🧾 Integrate **MLflow** for experiment tracking
* 
* 🌐 Build **Streamlit UI** for user-friendly predictions
* 
* ⚙️ Deploy on **Render / AWS EC2**

---

## 👩‍💻 Author

**👤 Sonali**
💡 Machine Learning Enthusiast | Data Science Learner
🔗 [GitHub Profile](https://github.com/sonali6062)

---

---

## 🌠 Summary 

🔹 Developed a complete **regression-based ML pipeline** from scratch

🔹 Achieved **87.77% R² Score** with **Gradient Boosting**

🔹 Implemented **hyperparameter tuning** for 7 models

🔹 Built with **Flask**, **scikit-learn**, **XGBoost**, and **CatBoost**

🔹 Demonstrates **real-world ML engineering workflow** (data → model → deployment)



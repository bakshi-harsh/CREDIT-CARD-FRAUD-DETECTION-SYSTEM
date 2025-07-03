# CREDIT-CARD-FRAUD-DETECTION-SYSTEM
# 💳 Credit Card Fraud Detection System

This project is a comprehensive **Credit Card Fraud Detection System** built using Python and Machine Learning. It aims to detect fraudulent transactions based on historical patterns and is equipped with an interactive web interface for real-time transaction review.

## 📌 Project Overview

Credit card fraud continues to be a major threat in the financial sector. This project uses machine learning to build an intelligent system that classifies transactions as **fraudulent** or **legitimate**, helping financial institutions reduce financial losses.

### 🎯 Key Features

- ⚙️ **Machine Learning Models**: Logistic Regression, Decision Tree (Gini & Entropy), and Random Forest.
- 📄 **Transaction PDF Input**: Supports input via transaction list PDFs for batch fraud analysis.
- 🖥️ **Flask Web Interface**: User-friendly interface for checking and reviewing flagged transactions.
- 📊 **Performance Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC.
- 🔍 **Explainability**: Model interpretability using key features like `Amount`, `Time`, and PCA-transformed attributes `V1–V28`.
---

### 📂 Dataset

You can access the dataset used in this project via the following link:

🔗 Download Dataset from Google Drive:

https://drive.google.com/file/d/1GD3cMcFb1w4pF2AMsYZUZHcMpweNricx/view?usp=sharing


### 🛠 Preprocessing Steps

- Handled missing values (none present)
- Feature scaling using `StandardScaler` for `Time` and `Amount`
- Addressed class imbalance using `SMOTE`

### 🧪 Models and Evaluation

| Model                     | Accuracy | Precision | Recall | ROC AUC |
|--------------------------|----------|-----------|--------|---------|
| Logistic Regression      | 99.86%   | 0.58      | 0.56   | 0.90    |
| Decision Tree (Gini)     | 99.89%   | 0.81      | 0.52   | 0.96    |
| Decision Tree (Entropy)  | 92.89%   | 0.93      | 0.93   | 0.93    |
| Random Forest            | 93.91%   | 0.99      | 0.89   | 0.94    |

---

## 🖥️ Web Interface

The system includes a Flask-based web interface for:

- Uploading PDF lists of transactions
- Viewing flagged transactions in real-time
- Manual verification for critical cases


---

## 📚 Tools and Libraries Used

- `Python 3.8+`
- `Pandas`, `NumPy` – data processing
- `Scikit-learn` – modeling and evaluation
- `Seaborn`, `Matplotlib` – data visualization
- `Imbalanced-learn` – SMOTE for handling imbalance
- `Flask` – web deployment
- `pdfplumber` – reading transaction lists from PDFs
- `joblib` – model serialization

---

## 🚀 Setup Instructions

1. **Clone the repository**

   bash
   git clone https://github.com/your-username/fraud-detection.git
   cd fraud-detection
2. Create a virtual environment

   bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies

   bash
   Copy code
   pip install -r requirements.txt

4. Run the app

  bash
  Copy code
  python app/routes.py


🔐 Future Enhancements

✅ Integrate SHAP or LIME for model explainability

✅ Real-time alerts via email/SMS

✅ Role-based authentication for auditors

✅ Additional visual analytics dashboard using Plotly/Dash

🙌 Contributors
Harsh Kumar – LinkedIn : https://www.linkedin.com/in/harsh-kumar-a120b8328/
Vansh Pratap Gautam – LinkedIn : https://www.linkedin.com/in/vansh-pratap-gautam-9375511a2/
Sonu Kumar – LinkedIn : https://www.linkedin.com/in/sonukumar102/

📄 License
This project is open-source and available under the MIT License.


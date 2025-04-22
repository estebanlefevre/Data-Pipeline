import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from flask import Flask, request, render_template_string

# Chargement des données
df = pd.read_csv("iris.csv", sep=";")
print(df.columns.tolist())
X = df[["sepal_width"]]
y = df["sepal_length"]

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle
with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("r2_score", score)
    mlflow.sklearn.log_model(model, "model")

# App Flask
app = Flask(__name__)

HTML = """
<!doctype html>
<title>Prédiction de longueur de sépale 🌸</title>
<h2>Entrez la largeur de sépale :</h2>
<form method=post>
  <input type=number step=0.01 name=sepal_width>
  <input type=submit value=Prédire>
</form>
{% if prediction %}
  <h3>📏 Longueur prédite : {{ prediction }} cm</h3>
{% endif %}
"""

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            val = float(request.form['sepal_width'])
            prediction = round(model.predict(np.array([[val]]))[0], 2)
        except:
            prediction = "Erreur de saisie"
    return render_template_string(HTML, prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

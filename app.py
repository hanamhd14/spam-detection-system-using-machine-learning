from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load Logistic Regression and Naive Bayes models and the TF-IDF Vectorizer
models = {
    'naive_bayes': pickle.load(open('naive_bayes_model.pkl', 'rb')),
    'logistic_regression': pickle.load(open('logistic_regression_model.pkl', 'rb'))
}
tfidf_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    result = None
    error_message = ""

    if request.method == 'POST':
        message = request.form.get('message', '')
        selected_model = request.form.get('model', '')

        if not selected_model:
            error_message = "Please choose a model."
        elif message and selected_model in models:
            # Transform the message for prediction
            transformed_message = tfidf_vectorizer.transform([message])

            # Predict using the selected model
            model = models[selected_model]
            prediction = model.predict(transformed_message)[0]
            result = 'Spam' if prediction == 1 else 'Ham'

    return render_template("predict.html", result=result, error_message=error_message)

if __name__ == "__main__":
    app.run(debug=True)

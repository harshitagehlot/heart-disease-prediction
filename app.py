from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return """
    <h1>Heart Disease Prediction Project</h1>
    <p>ML model Docker + Jenkins CI/CD successfully running!</p>
    """

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
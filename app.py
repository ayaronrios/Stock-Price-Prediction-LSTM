

from flask import Flask, render_template, request
from train_model import train_predict

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    stock = request.form["stock"]

    try:
        result = train_predict(stock)

        return render_template(
            "result.html",
            stock=stock,
            company=result["company"],
            rmse=result["rmse"],
            mae=result["mae"],
            mape=result["mape"],
            r2=result["r2"],
            prediction=result["predicted"],
            actual=result["actual"],
            image=f"{stock.upper()}_prediction.png",
            loss=f"{stock.upper()}_loss.png",
            history=f"{stock.upper()}_history.png"
        )

    except Exception as e:
        return render_template(
            "error.html",
            error=str(e)
        )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
from flask import Flask, render_template,request, jsonify

from chat import response

app = Flask(__name__)
context =dict()
@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    #print(text)
    res = response(text)
    #print(res)
    message = {"answer": res}
    #print(message)
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)
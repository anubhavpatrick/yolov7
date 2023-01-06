from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, NVIDIA GRIL Students!</p>"
    
    
@app.route("/bms")
def hello_bms():
    return "<p>Hello, NVIDIA BMS Trainees!</p>"

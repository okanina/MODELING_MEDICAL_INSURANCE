from flask import Flask, render_template, request
from src.pipeline.prediction_pipeline import CustomData, predictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/home.html', methods=["GET", "POST"])
def predict_datapoint():
    if request.method =='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            age = request.form.get("age"),
            sex = request.form.get("sex"),
            bmi = request.form.get("bmi"),
            smoker = request.form.get("smoker"),
            weight = request.form.get("weight"),
            diabetes = request.form.get("diabetes"),
            city = request.form.get("city"),
            regular_ex = request.form.get("regular_ex"),
            no_of_dependents = request.form.get("no_of_dependents"),
            bloodpressure = request.form.get("bloodpressure"),
            hereditary_diseases = request.form.get("hereditary_diseases"),
            job_title = request.form.get("job_title"),
            )
        df = data.get_dataframe()

        pred = predictPipeline()
        result =pred.get_predictions(df)

        
    return render_template('home.html', result = result[0]) 

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)   
    
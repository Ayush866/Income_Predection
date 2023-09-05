from flask import render_template,Flask,request
import numpy as np
import pandas as pd
from income.exception import CustomException
from income.logger import logging
from income.predictor import ModelResolver
import pandas as pd
from income.utils import load_object
import os,sys
from income.pipeline.prediction import CustomData,PredictPipeline

logging.info("starting Flask")
app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")
logging.info("started")

@app.route('/predictincome', methods=['GET', 'POST'])
def predict_datapoint():
    try:
        if request.method == 'GET':
            return render_template('index.html')
        else:
            data = CustomData(
                age=float(request.form.get('Age')),
                workclass=request.form.get('workclass'),
                education_num=int(request.form.get('Education_num')),
            marital_status=request.form.get('marital-status'),
                occupation=request.form.get('occupation'),
            relationship=request.form.get('relationship'),
            race=request.form.get('race'),
            sex=request.form.get('sex'),
            capital_gain = int(request.form.get('capital-gain')),
            capital_loss = int(request.form.get('capital-loss')),
            hours_per_week=int(request.form.get('hours-per-week')),
            country=request.form.get('country')


        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        if results[0] == 1:
            results = "Salary greater Than 50k"
        else:
            results = "Salary not greater than 50k"
        return render_template('index.html', results=results)
    except Exception as e:
        raise CustomException(error_message=e, error_detail=sys)


if __name__=="__main__":
    app.run(host="0.0.0.0")
# Project-4-Team-1
Machine Learning project

[Diabetes Dataset](https://www.kaggle.com/code/tumpanjawat/diabetes-eda-random-forest-hp)

We are predicting the likelihood of diabetes based on various demographic, health, and lifestyle factors. The dataset includes information such as gender, age, presence of hypertension and heart disease, smoking history, BMI, HbA1c level, and blood glucose level. Our goal is to build a predictive model that accurately classifies individuals as either having or not having diabetes, which can help in early identification and management of the condition.

Explanation of Data columns:
- Gender - refers to the biological sex of the individual, which can have an impact on their susceptibility to diabetes. There are three categories in it male ,female and other.
- Age - is an important factor as diabetes is more commonly diagnosed in older adults.Age ranges from 0-80 in our dataset.
- Hypertension - is a medical condition in which the blood pressure in the arteries is persistently elevated. It has values a 0 or 1 where 0 indicates they don’t have hypertension and for 1 it means they have hypertension.
- Heart disease - is another medical condition that is associated with an increased risk of developing diabetes. It has values a 0 or 1 where 0 indicates they don’t have heart disease and for 1 it means they have heart disease.
- Smoking history - is also considered a risk factor for diabetes and can exacerbate the complications associated with diabetes.In our dataset we have 5 categories i.e not current,former,No Info,current,never and ever.
- BMI (Body Mass Index) - is a measure of body fat based on weight and height. Higher BMI values are linked to a higher risk of diabetes. The range of BMI in the dataset is from 10.16 to 71.55. BMI less than 18.5 is underweight, 18.5-24.9 is normal, 25-29.9 is overweight, and 30 or more is obese.
- HbA1c (Hemoglobin A1c) level - is a measure of a person's average blood sugar level over the past 2-3 months. Higher levels indicate a greater risk of developing diabetes. Mostly more than 6.5% of HbA1c Level indicates diabetes.
- Blood glucose level - refers to the amount of glucose in the bloodstream at a given time. High blood glucose levels are a key indicator of diabetes.
- Diabetes - is the target variable being predicted, with values of 1 indicating the presence of diabetes and 0 indicating the absence of diabetes.


#### Divya

### Model 1: Logistic Regression
I used logistic regression because the target variable had values of 0s and 1s. Initially, the model showed 95.78% accuracy. But when I checked the predictions against the actual values, I found that it often labeled positive cases as negative, which isn't right. This misclassification needs fixing to make the predictions more accurate.

In the confusion matrix, the following values were found.
[[    0 22875]
 [    0  2125]]

This confusion matrix suggests that the model has correctly classified all negative instances (TN = 0) but incorrectly classified all positive instances as negative (FP = 22875). It's possible that there may be issues with the data imbalance. In this case, it appears that the model is not effectively distinguishing between the two classes, resulting in a high number of false positives and no true negatives.

Classification report:
Accuracy: 0.96
Precision (Class 0): 0.96
Recall (Class 0): 0.99
F1-score (Class 0): 0.98

Precision (Class 1): 0.87
Recall (Class 1): 0.60
F1-score (Class 1): 0.71

The model performs well in identifying instances without diabetes (class 0), achieving high precision (96%) and recall (99%).
However, its performance in identifying instances with diabetes (class 1) is less satisfactory, with lower recall (60%) despite reasonable precision (87%).
Overall, the model demonstrates high accuracy (96%), but there's room for improvement in correctly identifying positive cases (class 1).

Hence I tried another model.

### Model 2: Random Forest Classifier

The confusion matrix of Random Forest classification looked much better. 
	        Predicted 0	Predicted 1
Actual 0	    22722	    153
Actual 1	    740	        1385

The model correctly identified:
22,722 instances as negative (class 0) and labeled them correctly (True Negatives).
1,385 instances as positive (class 1) and labeled them correctly (True Positives).
The model incorrectly classified:
153 instances as positive (class 1) when they were actually negative (False Positives).
740 instances as negative (class 0) when they were actually positive (False Negatives).

The accuracy and classification report looked like this.

Accuracy Score : 0.96428
Classification Report
              precision    recall  f1-score   support

           0       0.97      0.99      0.98     22875
           1       0.90      0.65      0.76      2125

In summary, the model performs well for class 0, with high precision, recall, and F1-score, indicating accurate classification. However, for class 1, while precision is still relatively high, recall is lower, suggesting that the model struggles to capture all instances of class 1.

The dataset is highly imbalanced which is affecting the prediction, hence tried to set the class weights.

Adjusting the class weights did not help. In some cases class 1 performed a tiny bit better, but class 0 declined a lot. In some cases both declined. So the model without the class weights is the one good for now.

#### Kajal

#### Jessamyn

#### Donal

#### Ayo

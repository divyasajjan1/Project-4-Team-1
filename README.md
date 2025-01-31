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


#### [Divya](diabetes_prediction.ipynb)

First, found the distribution of the target variable. 91.5% (91500) were 0 (No diabetes) and 8.5% (8500) were 1 (Diabetes) in the target variable. Understood that the dataset is pretty imbalanced.

Converted the categorical columns of 'gender' and 'smoking history' into numerical data using pd.get_dummies().

Scaled the training and testing data using StandardScaler().

Continued to model building using the scaled data.

### Model 1: Logistic Regression
I used logistic regression because the target variable had binary values. Initially, the model showed 95.78% accuracy. But when I checked the predictions against the actual values, I found that it often labeled positive cases as negative, which isn't right. In health, spotting problems early is key. Missing a diagnosis means people won't get the treatment they need, which can make things worse. This misclassification needs fixing to make the predictions more accurate.

By looking at the co-efficients to identify the most important features
- Features like 'HbA1c_level', 'blood_glucose_level', and 'age' have the largest absolute coefficients, indicating significant influence on the predicted probability of the target class.
- Positive coefficients for features like 'hypertension', 'heart_disease', and certain smoking history categories suggest that their presence increases the probability of the target class.
- Negative coefficients for features like 'smoking_history_No Info', 'gender_Other', and 'gender_Female' suggest that their presence decreases the probability of the target class.

In the confusion matrix, the following values were found.
[[22679   196]
 [  858  1267]]

- This confusion matrix suggests that the model has correctly classified the majority of instances as either positive or negative. It has a high number of true positives and true negatives, indicating effective classification performance.
- However, there are also notable numbers of false positives and false negatives. The presence of false positives (FP = 196) indicates instances incorrectly classified as positive, while false negatives (FN = 858) indicate instances incorrectly classified as negative.
- The relatively small number of false positives compared to true positives suggests that the model's precision (the proportion of true positive predictions out of all positive predictions) is reasonably high. However, the presence of false negatives indicates that the model may have lower recall (the proportion of true positive predictions out of all actual positives).

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

Tuning the model:

Tried improving the false negatives and false positives by using regularization techniques.
Implemented Lasso and Ridge regularizations. The values did not improve.
Hence I tried another model.

### Model 2: Random Forest Classifier

In our Random Forest model, we analyzed feature importance to identify the most influential predictors. The feature importance scores indicate the relative importance of each feature in predicting the target variable.

Top 3 features contributing to the target variable diabetes - 
- blood_glucose_level: This feature has the highest importance score, indicating that it contributes the most to predicting diabetes in the Random Forest model. Elevated blood glucose levels are a key indicator of diabetes and play a central role in diagnosing and managing the condition.
- HbA1c_level: Hemoglobin A1c (HbA1c) level is another crucial predictor of diabetes. It reflects an individual's average blood sugar levels over the past 2-3 months and is commonly used to diagnose and monitor diabetes. Higher HbA1c levels are associated with poorly controlled diabetes and an increased risk of complications.
- age: Age is a significant predictor of diabetes risk, with the likelihood of developing the condition increasing with age. Aging is associated with changes in metabolism, hormone levels, and lifestyle factors, all of which can influence diabetes risk.

Understanding their importance can help healthcare professionals prioritize screening, diagnosis, and treatment strategies for individuals at risk of diabetes.

The confusion matrix of Random Forest classification looked much better. 

[[22722   153]
 [  740  1385]]	    	        

The model correctly identified:
22,722 instances as negative (class 0) and labeled them correctly (True Negatives).
1,385 instances as positive (class 1) and labeled them correctly (True Positives).
The model incorrectly classified:
153 instances as positive (class 1) when they were actually negative (False Positives).
740 instances as negative (class 0) when they were actually positive (False Negatives).

The accuracy and classification report looked like this.

Accuracy Score : 0.96428

Precision (Class 0): 0.97
Recall (Class 0): 0.99
F1-score (Class 0): 0.98

Precision (Class 1): 0.90
Recall (Class 1): 0.65
F1-score (Class 1): 0.76

In summary, the model performs well for class 0, with high precision, recall, and F1-score, indicating accurate classification. However, for class 1, while precision is still relatively high, recall is lower, suggesting that the model struggles to capture all instances of class 1.

Tuning the model:
Tried doing a hyperparameter tuning with grid search and randomized search. The search did not produce any output maybe because of the heavy system requirements needed for them. 

The dataset is highly imbalanced which is affecting the prediction, hence tried to set the class weights.

Tried classweights {0:1,1:10}, {0:1,1:9.5}, {0:1.2,1:1}, {0:1,1:8} and so on.
Adjusting the class weights did not help. In some cases class 1 performed a tiny bit better, but class 0 declined a lot. In some cases both declined. So the model without the class weights is the one good for now.

([Kajal]([JupyterNotebooks/Videogame_comparisions_Analysis_KM.ipynb](https://github.com/divyasajjan1/Project-4-Team-1/blob/main/KM_diabetes_prediction.ipynb)))
  ### Model 3: Neural Network
-Dataset Preprocessing: 

1.	Categorical Variable Encoding:
 Transform categorical variables, such as gender and smoking history, into numerical representations. This process enables us to incorporate these variables into our analyses effectively.( I 
 USED LABEL-ENCODER TO ENCODE THE CATEGORICAL VARIABLE )
2.	Standardization:
  Standardize numerical data, ensuring that all variables are on a consistent scale using sklearn's StandardScaler.
3.	Train-Test-Split:
  Partition dataset into separate training and testing sets using sklearn's train_test_split function.

-Neural Network Architecture:

Model:Sequential

* Input Layer: Dense layer with 64 neurons, ReLU activation.
* Hidden Layer 1: Dense layer with 32 neurons, ReLU activation.
* Output Layer: Dense layer with 1 neuron, Sigmoid activation for binary classification.

-Confusion Matrix:

[[18284   8]
 [  551  1157]]


-Metrics used to Evaluate the Model:

Accuracy: 0.9716
Precision: 0.9830508474576272
Recall: 0.6791569086651054
F1 Score: 0.8033240997229918

-Hyperparameter tuning of a neural network classifier using Keras and scikit-learn:

Ultimately finding the optimal configuration for maximizing classification accuracy

Best Parameters: {'dropout_rate': 0.3, 'optimizer': 'adam'}
Best Accuracy: 0.9715874932722871

-A neural network model with a deep architecture:

* Input layer: Dense layer with 128 neurons and ReLU activation function, accepting input data with the shape determined by the number of features in the training dataset.
* Dropout layer: Applied dropout regularization with a rate of 0.2 to prevent overfitting by randomly dropping 20% of the neurons during training.
* Hidden layers: Dense layers with 64, 32, and 16 neurons, respectively, each utilizing the ReLU activation function.
* Output layer: Dense layer with 1 neuron and a sigmoid activation function, suitable for binary classification tasks like predicting diabetes.
Following model construction, the model was compiled with the Adam optimizer and binary cross-entropy loss function, optimized for accuracy. Subsequently, the model was trained using the training dataset for 20 epochs with a batch size of 32, with 20% of the data reserved for validation.

Test Accuracy: 0.971750020980835


#### Jessamyn
Created visualization to showcase the importance of each variable within the dataset and how it correlates to the target variable, diabetes. Visualizing our data not only makes our findings more comprehensible but also offers a clearer understanding of the relationships between variables and their impact on diabetes risk. We've employed various visualization techniques to bring our analysis to life, enabling us to spot trends and patterns, compare variables and enhance interpretation.

Key visualizations to expect:
1.	Gender Distribution: A pie chart showcasing the distribution of diabetes cases across genders, highlighting any disparities.
2.	Age vs. Diabetes: A line graph depicting the correlation between age and diabetes risk, with age on the x-axis and diabetes prevalence on the y-axis.
3.	BMI Distribution: A histogram or box plot illustrating the BMI distribution among individuals with and without diabetes.
4.	Smoking History Impact: A pie chart or bar graph comparing the diabetes prevalence between smokers and non-smokers.
5.	Hypertension and Heart Disease: A series of stacked bar charts or heatmaps showcasing the prevalence of diabetes among individuals with varying levels of hypertension and heart disease.
6.	Blood Glucose and HbA1c Levels: displaying the relationship between blood glucose levels, HbA1c levels, and diabetes risk.

#### Donal

To visualize the relationship between gender and diabetes status (presence or absence of diabetes), we can use a bar chart to show the distribution of diabetic and non-diabetic individuals across different genders. This visualization will help us understand how diabetes prevalence varies between different gender categories
To create a visualization of **"Gender vs. Diabetes"** using Python and matplotlib, we can plot a bar chart to show the distribution of diabetes cases among different gender categories. This will help us understand how diabetes prevalence varies between male, female, and other genders in the dataset.

Here's a step-by-step guide to creating this visualization:
Load the Dataset:
Start by loading your dataset that contains information about gender and diabetes status.
Data Preparation:
Prepare the data by grouping it based on gender and diabetes status to count the occurrences.
Visualization:
Use matplotlib to plot a bar chart showing the count of diabetic and non-diabetic individuals by gender.

To visualize the relationship between **age and diabetes** status, we can create a box plot to compare the distribution of ages among diabetic and non-diabetic individuals. This type of visualization helps us understand how the age varies between the two groups and identify any age-related patterns associated with diabetes.



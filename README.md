

![churn](https://github.com/TariTech28/Customer-Churn-Project/assets/140518602/281cc025-b4cd-4a6c-8854-591cb94403d8)

# Project Steps
### 1. Problem Definition
ConnectTel, a leading telecommunications company, is facing a significant challenge in customer churn. To address this issue, the company has initiated a project to develop a robust customer churn prediction system. 
As a Data Scientist, my role in this project is to leverage advanced analytics and machine learning techniques on available customer data to accurately forecast customer churn. 
The goal is to implement targeted retention initiatives, reduce customer attrition, enhance customer loyalty, and maintain a competitive edge in the dynamic telecommunications industry.



### 2. Data Collection
Obtain the dataset containing relevant information about telecom customers. The dataset includes features such as customer demographics, contract details, and service usage.

* CustomerID: A unique identifier assigned to each telecom customer, enabling tracking and identification of individual customers.
* Gender: The gender of the customer, which can be categorized as male, or female. This information helps in analyzing gender based trends in customer churn.
* SeniorCitizen: A binary indicator that identifies whether the customer is a senior citizen or not. This attribute helps in understanding if there are any specific churn patterns among senior customers.
* Partner: Indicates whether the customer has a partner or not. This attribute helps in evaluating the impact of having a partner on churn behavior.
* Dependents: Indicates whether the customer has dependents or not. This attribute helps in assessing the influence of having dependents on customer churn.
* Tenure: The duration for which the customer has been subscribed to the telecom service. It represents the loyalty or longevity of the customer’s relationship with the company and is a significant predictor of churn.
* PhoneService: Indicates whether the customer has a phone service or not. This attribute helps in understanding the impact of phone service on churn.
* MultipleLines: Indicates whether the customer has multiple lines or not. This attribute helps in analyzing the effect of having multiple lines on customer churn.
* InternetService: Indicates the type of internet service subscribed by the customer, such as DSL, fiber optic, or no internet service. It helps in evaluating the relationship between internet service and churn.
* OnlineSecurity: Indicates whether the customer has online security services or not. This attribute helps in analyzing the impact of online security on customer churn.
* OnlineBackup: Indicates whether the customer has online backup services or not. This attribute helps in evaluating the impact of online backup on churn behavior.
* DeviceProtection: Indicates whether the customer has device protection services or not. This attribute helps in understanding the influence of device protection on churn.
* TechSupport: Indicates whether the customer has technical support services or not. This attribute helps in assessing the impact of tech support on churn behavior.
* StreamingTV: Indicates whether the customer has streaming TV services or not. This attribute helps in evaluating the impact of streaming TV on customer churn.
* StreamingMovies: Indicates whether the customer has streaming movie services or not. This attribute helps in understanding the influence of streaming movies on churn behavior.
* Contract: Indicates the type of contract the customer has, such as a month to month, one year, or two year contract. It is a crucial factor in predicting churn as different contract lengths may have varying impacts on customer loyalty.
* PaperlessBilling: Indicates whether the customer has opted for paperless billing or not. This attribute helps in analyzing the effect of paperless billing on customer churn.
* PaymentMethod: Indicates the method of payment used by the customer, such as electronic checks, mailed checks, bank transfers, or credit cards. This attribute helps in evaluating the impact of payment methods on churn.
* MonthlyCharges: The amount charged to the customer on a monthly basis. It helps in understanding the relationship between monthly charges and churn behavior.
* TotalCharges: The total amount charged to the customer over the entire tenure. It represents the cumulative revenue generated from the customer and may have an impact on churn.
* Churn: The target variable indicates whether the customer has churned (canceled the service) or not. It is the main variable to predict in telecom customer churn analysis.

### 3. Exploratory Data Analysis (EDA)
Performed exploratory data analysis to understand the dataset. Key steps include:

#### Data Inspection:

* The dataset contains various features related to telecom customer information.
* Senior Citizen indicates that around 16.2% of the customers are senior citizens.
* Tenure has a range from 0 to 72 months.
* Monthly Charges range from 18.25 to 118.75.
* I Performed a comprehensive check for data quality issues, including missing values and duplicates, to ensure the integrity of the dataset:
* The result showed zero missing values across all features, indicating that the dataset is complete with no information gaps.
* The output revealed no duplicated rows, confirming the uniqueness of each record.



#### Descriptive Statistics:

Summarize key statistics for numerical and categorical features.
* SeniorCitizen:
Count: Indicates that there are 7,043 records in the dataset.
Mean: The mean of 0.162 suggests that around 16.2% of the customers are senior citizens.
Standard Deviation: With a standard deviation of 0.369, there is some variability in the distribution of senior citizens.
min-max Provides the range of values: 0 (non-senior) and 1 (senior). 25% of customers are senior citizens, as indicated by the 75th percentile.
Tenure: Indicates the range of tenure from 0 to 72 months.

* Majority of customers are loyal patrons and veteran customers

* The dataset is clean, containing no missing values or duplicate records, ensuring the reliability of the subsequent analysis and modeling steps.

#### Visualizations:
Visualized distributions, relationships, and correlations.
Explored the impact of various features on customer churn using visualizations.
Examined univariate, bivariate, and multivariate relationships.

* Majority of customers are loyal patrons and veteran customers (as seen in the tenure bracket visualization).
* Senior citizens have a lower count, and there is some variability in the distribution.
* Veteran customers (51-72yearss) have the least churn, while Loyal patrons (<=50years) have the highest churn.
* Customers without online security services have the highest churn rate.
* Gender distribution is slightly skewed, with slightly more males.
* The distribution of tenure exhibited a right-skewed pattern, indicating that a significant portion of customers has relatively lower tenure durations. This right-skewed distribution implies that a substantial number of customers are relatively new or have shorter relationships with the telecom service.
* Indicates that Customers with fibre optic service have the highest churn rate.
* Indicates that customers with no internet service have the lowest Churn rate.
* Customers with the month-to-month contract have the highest churn rate.
* Indicates that Customers who use the  electronic check payment method have the highest churn rate while customers who use the credit card have the lowest Churn rate.
* Customers using fibre optic have the highest churn rate.


### 4. Feature Engineering
Enhanced the dataset by encoding categorical variables and creating new features based on insights from EDA. Transformed the data to ensure it is suitable for machine learning model training.

### 5. Model Selection, Training, and Validation
Selected and trained at least three supervised learning models, including Logistic Regression and Random Forest Classifier. 
 Steps include:

* Data Splitting:
Split the dataset into training and testing sets.


* Model Training:
Train each selected model on the training data and validated their performance.

* Model Evaluation:
Analyzed the results of the trained models, considering important metrics such as accuracy, precision, recall, F1-score, and AUC-ROC. Assessed the trade-offs and identified the model(s) that best address the customer churn prediction problem.


# 6. Model Interpretation
Interpreted the results of the chosen model, including confusion matrices and important model parameters.
The Logistic Regression model has the highest ROC score at 70.38%. This means that the model is 70.38% accurate at distinguishing between churners and non-churners.

* Logistic Regression: 
* Accuracy: The model correctly predicted approximately 79.63% of the instances.
* Precision: The model achieved a precision of approximately 64.73%, indicating that when it predicts a positive outcome (churn), it is correct about 64.73% of the time.
* Recall: The model captured approximately 50.67% of the actual positive cases.
* F1-score: The F1-score is the harmonic mean of precision and recall. It provides a balance between precision and recall. In this case, the F1-score is approximately 56.84%.
* An AUC-ROC value of 0.7038 suggests a moderate level of discrimination ability by the model.

# BUSINESS RECOMMENDATIONS
The company should be more concerned with reducing false negatives. This is because false negatives can lead to lost revenue and customer loyalty. When a customer churns, the company loses the recurring revenue that the customer was generating. Additionally, churned customers are more likely to leave negative reviews and spread negative word-of-mouth, which can damage the company's reputation.
To reduce false negatives, the company should focus on improving the recall of their churn prediction model.
The company should identify the customers who are at risk of churning based on the model's predictions. The company can then intervene early and try to prevent these customers from churning. For example, the company could offer these customers a discount, a special promotion, or personalized customer service.




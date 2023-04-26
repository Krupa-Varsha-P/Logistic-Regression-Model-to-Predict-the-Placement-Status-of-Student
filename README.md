# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```txt
1. Import dataset
2. Check for null and duplicate values
3. Assign x and y values
4. Split data into train and test data
5. Import logistic regression and fit the training data
6. Predict y value
7. Calculate accuracy and confusion matrix
```

## Program:
```txt
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Krupa Varsha P 
RegisterNumber: 212220220022
```
```python3
import pandas as pd 
data=pd.read_csv('/content/Placement_Data.csv') 
data.head()
```

```python3
data1=data.copy() 
data1=data1.drop(["sl_no","salary"],axis=1) 
data1.head()
```

```python3
data1.isnull().sum()
```

```python3
data1.duplicated().sum()
```

```python3
from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder() 
data1["gender"]=le.fit_transform(data1["gender"]) 
data1["ssc_b"]=le.fit_transform(data1["ssc_b"]) 
data1["hsc_b"]=le.fit_transform(data1["hsc_b"]) 
data1["hsc_s"]=le.fit_transform(data1["hsc_s"]) 
data1["degree_t"]=le.fit_transform(data1["degree_t"]) 
data1["workex"]=le.fit_transform(data1["workex"]) 
data1["specialisation"]=le.fit_transform(data1["specialisation"]) 
data1["status"]=le.fit_transform(data1["status"]) 
data1
```

```python3
x=data1.iloc[:,:-1] 
x
y=data1["status"] 
y
```

```python3
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0) 
from sklearn.linear_model import LogisticRegression 
lr=LogisticRegression(solver="liblinear") 
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test) 
y_pred
```

```python3
from sklearn.metrics import accuracy_score 
accuracy=accuracy_score(y_test,y_pred) 
accuracy
```

```python3
from sklearn.metrics import confusion_matrix 
confusion=(y_test,y_pred) 
confusion
```

```python3
from sklearn.metrics import classification_report 
classification_report1=classification_report(y_test,y_pred) 
print(classification_report1)
```

```python3
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![238cac5c-1c68-4a85-ab04-1685abc8aa9d](https://user-images.githubusercontent.com/100466625/234622529-40c32d6c-b219-4461-9e95-9e0181c8ddf1.jpg)

![d8d10370-2c85-4be2-be4a-1f6d5c955385](https://user-images.githubusercontent.com/100466625/234622580-5eb82550-7ee7-417d-8ba3-a828cc4a8635.jpg)

![07bfcb5c-3eee-4b8c-ad66-8c4a4f80c4c2](https://user-images.githubusercontent.com/100466625/234622611-43f71053-3b37-4b9d-a402-4aa954f8fb7f.jpg)

![a8c66e52-3c2b-46f6-a505-e11665046346](https://user-images.githubusercontent.com/100466625/234622643-bf07c08a-ca8f-4cc4-8232-267a2e5d87ca.jpg)

![965c4220-d42f-4041-8db1-ad7429b46fd9](https://user-images.githubusercontent.com/100466625/234622673-5769f09d-9488-4b84-8a9a-5881744e74a9.jpg)

![d2a72f51-adb0-4549-8024-7e01d8ef3cc3](https://user-images.githubusercontent.com/100466625/234622720-8bbd112a-58b7-4264-8670-79b1822446ec.jpg)

![658b3d41-f436-4dc6-826a-3ba7f73fd48d](https://user-images.githubusercontent.com/100466625/234622754-83627cec-21ab-4cfc-ac1d-c31fc02b100b.jpg)

![0d2312ab-cb28-4550-9365-df0b2c352d58](https://user-images.githubusercontent.com/100466625/234622804-d3fdf56b-4a39-4229-842b-5d06e57fcb1d.jpg)

![a43a655e-0597-4b01-bcda-6ab981bebe9b](https://user-images.githubusercontent.com/100466625/234622848-1b103247-9dde-4506-9f1a-c73aa5a0c831.jpg)

![235911df-39ff-4226-9c04-447840999ff1](https://user-images.githubusercontent.com/100466625/234622870-4232c9fb-c904-4d49-9762-695ea516595e.jpg)

![d9b7766d-33e2-4d98-8729-d1b72190f59e](https://user-images.githubusercontent.com/100466625/234622898-791b75c9-d89b-4d95-852f-0fbee6ed3a6b.jpg)

![ef876898-62ec-483d-a62f-1b3a39f4a881](https://user-images.githubusercontent.com/100466625/234622928-3e4aa0b7-abbb-4b6c-89e4-5884d805764d.jpg)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

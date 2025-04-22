## Name:Manimaran V
## Reg.no:212224220060
## EXNO-3-Feature Encoding and Transformation

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
      ```
      import pandas as pd
      df=pd.read_csv("Encoding Data.csv")
      df
      ```
  ![image](https://github.com/user-attachments/assets/d012aabc-3563-45ec-994e-82be0dde1221)
  ```
      from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
      pm=['Hot','Warm','Cold']
      e1=OrdinalEncoder(categories=[pm])
      e1.fit_transform(df[["ord_2"]])
  ```
![image](https://github.com/user-attachments/assets/9488b366-f636-4109-993f-b8ac2cd306b7)
  ``` 
     df['bo2']=e1.fit_transform(df[["ord_2"]])
     df
  ```
![image](https://github.com/user-attachments/assets/384c38b4-ad7d-49bf-840d-00759637c3cc)
 ```
     le=LabelEncoder()
     dfc=df.copy()
     dfc['ord_2']=le.fit_transform(dfc['ord_2'])
     dfc
 ```
![image](https://github.com/user-attachments/assets/e37cde6e-2b27-4d24-87cf-e34ad067fd20)
 ```
     from sklearn.preprocessing import OneHotEncoder
     ohe=OneHotEncoder(sparse_output=False)
     df2=df.copy()
     enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
     df2=pd.concat([df2,enc],axis=1)
     df2
```
![image](https://github.com/user-attachments/assets/1f53d115-10db-4487-b81f-4d197a13dd4e)
```
     pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/5c44df34-bc94-47bd-8da6-039c904a2a1d)
```
     pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/844dd393-5e27-4790-8338-3b283ec96724)
```
     from category_encoders import BinaryEncoder
     df=pd.read_csv("data.csv")
     df
     be=BinaryEncoder()
     nd=be.fit_transform(df['Ord_2'])
     df
     dfb=pd.concat([df,nd],axis=1)
     dfb
```
![image](https://github.com/user-attachments/assets/d8a68bb6-2f45-40cb-ae47-27581c17a438)
```
     from category_encoders import TargetEncoder
     te=TargetEncoder()
     CC=df.copy()
     new=te.fit_transform(X=CC["City"],y=CC["Target"])
     CC=pd.concat([CC,new],axis=1)
     CC
```
![image](https://github.com/user-attachments/assets/543e396b-29bd-4716-81ff-333b301751e0)
```
     import pandas as pd
     from scipy import stats
     import numpy as np
     df=pd.read_csv("Data_to_Transform.csv")
     df
```
![image](https://github.com/user-attachments/assets/ec979cba-573b-4900-bf03-01bb0c79781a)
```
     df.skew()
```
![image](https://github.com/user-attachments/assets/33a0b424-df5a-4bfb-84a0-4005c1c8e0d6)
```
     np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/a9d7b53a-3346-48ca-9e29-d09763b20410)
```
     np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/0b9f2e54-c935-41c6-8cab-f88b9c9cf957)
```
     np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/7d607107-eb77-4b0e-822f-154156933ff8)
```
     np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/3309ea8f-eee8-453e-9598-28d08a1ac525)
```
     df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
     df
```
![image](https://github.com/user-attachments/assets/b9390829-72f4-4d76-8025-65f5465f4865)
```
     df.skew()
```
![image](https://github.com/user-attachments/assets/d8936f8a-2543-4f53-ac4e-b95528a41349)
```
     df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
     df.skew()
```
![image](https://github.com/user-attachments/assets/0cdb95ba-ab21-4403-a5c7-82d0953d186c)
```
     from sklearn.preprocessing import QuantileTransformer
     qt=QuantileTransformer(output_distribution='normal')
     df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
     df
```
![image](https://github.com/user-attachments/assets/dda78aea-6234-4205-91cd-cd62670eb70b)
```
     import seaborn as sns
     import statsmodels.api as sm
     import matplotlib.pyplot as plt
     sm.qqplot(df["Moderate Negative Skew"],line='45')
     plt.show()
```
![image](https://github.com/user-attachments/assets/0edf21eb-118c-4f41-b929-bab2d4faf88b)
```
     sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
     plt.show()
```
![image](https://github.com/user-attachments/assets/cde4131f-8cf0-4212-a718-8fd7a7ba366a)
```
     from sklearn.preprocessing import QuantileTransformer
     qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
     df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
     sm.qqplot(df["Moderate Negative Skew"],line='45')
     plt.show()
```
![image](https://github.com/user-attachments/assets/fae8bbcb-f85b-480f-981d-899fe6f3660b)
```
     df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
     sm.qqplot(df["Highly Negative Skew"],line='45')
     plt.show()
```
![image](https://github.com/user-attachments/assets/28de6c58-93c7-4347-a60b-a0da7a52c0e6)
```
     dt=pd.read_csv("titanic_dataset.csv")
     dt
```
![image](https://github.com/user-attachments/assets/c3ff7935-adeb-408c-8c90-4f55b672556c)
```
     from sklearn.preprocessing import QuantileTransformer
     qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
     dt["Age_1"]=qt.fit_transform(dt[["Age"]])
     sm.qqplot(dt['Age'],line='45') 
     plt.show()
```
![image](https://github.com/user-attachments/assets/bb4e4bf8-c7a8-40e9-a236-f1e20029d50a)
```
     sm.qqplot(df["Highly Negative Skew_1"],line='45')
     plt.show()
```
![image](https://github.com/user-attachments/assets/e6c0a1f4-c299-424b-9e61-c8146594fca4)

# RESULT:
  Thus the given data, Feature Encoding, Transformation process and save the data to a file
was performed successfully.

       

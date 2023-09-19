#!/usr/bin/env python
# coding: utf-8

# # ML-Insurance Cost Prediction Project

# In[53]:


import pandas as pd


# In[54]:


data = pd.read_csv(r"C:\Users\mdmumtaz\Downloads\archive (4)\insurance.csv")


# # 1. Display Top 5 Rows of The Dataset

# In[55]:


data.head()


# # 2. Check Last 5 Rows of The Dataset

# In[56]:


data.tail()


# # 3. Find Shape of Our Dataset (Number of Rows And Number of Columns)

# In[57]:


data.shape


# In[58]:


print("Number of Rows",data.shape[0])
print("Number of Columns",data.shape[1])


# # 4. Get Information About Our Dataset Like Total Number Rows, Total Number of Columns, Datatypes of Each Column And Memory Requirement

# In[59]:


data.info()


# # 5.Check Null Values In The Dataset

# In[60]:


data.isnull().sum()


# # 6. Get Overall Statistics About The Dataset

# In[61]:


data.describe(include='all')


# # 7. Covert Columns From String ['sex' ,'smoker','region' ] To Numerical Values

# In[62]:


data.head()


# In[63]:


data['sex'].unique()


# In[64]:


data['sex']=data['sex'].map({'female':0,'male':1})


# In[20]:


data.head()


# In[65]:


data['smoker']=data['smoker'].map({'yes':1,'no':0})


# In[66]:


data.head()


# In[67]:


data['region'].unique()


# In[68]:


data['region']=data['region'].map({'southwest':1,'southeast':2,
                   'northwest':3,'northeast':4})


# In[69]:


data.head()


# # 8. Store Feature Matrix In X and Response(Target) In Vector y

# In[70]:


data.columns


# In[71]:


X = data.drop(['charges'],axis=1)


# In[72]:


y = data['charges']


# # 9. Train/Test split
# 1. Split data into two part : a training set and a testing set
# 2. Train the model(s) on training set
# 3. Test the Model(s) on Testing set

# In[73]:


from sklearn.model_selection import train_test_split


# In[74]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[75]:


y_train


# # 10. Import the models

# In[76]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# # 11. Model Training

# In[77]:


lr = LinearRegression()
lr.fit(X_train,y_train)
svm = SVR()
svm.fit(X_train,y_train)
rf = RandomForestRegressor()
rf.fit(X_train,y_train)
gr = GradientBoostingRegressor()
gr.fit(X_train,y_train)


# # 12. Prediction on Test Data

# In[78]:


y_pred1 = lr.predict(X_test)
y_pred2 = svm.predict(X_test)
y_pred3 = rf.predict(X_test)
y_pred4 = gr.predict(X_test)

df1 = pd.DataFrame({'Actual':y_test,'Lr':y_pred1,
                  'svm':y_pred2,'rf':y_pred3,'gr':y_pred4})


# In[36]:


df1


# # 13. Compare Performance Visually

# In[79]:


import matplotlib.pyplot as plt


# In[80]:


plt.subplot(221)
plt.plot(df1['Actual'].iloc[0:11],label='Actual')
plt.plot(df1['Lr'].iloc[0:11],label="Lr")
plt.legend()

plt.subplot(222)
plt.plot(df1['Actual'].iloc[0:11],label='Actual')
plt.plot(df1['svm'].iloc[0:11],label="svr")
plt.legend()

plt.subplot(223)
plt.plot(df1['Actual'].iloc[0:11],label='Actual')
plt.plot(df1['rf'].iloc[0:11],label="rf")
plt.legend()

plt.subplot(224)
plt.plot(df1['Actual'].iloc[0:11],label='Actual')
plt.plot(df1['gr'].iloc[0:11],label="gr")

plt.tight_layout()

plt.legend()


# # 14. Evaluating the Algorithm

# In[81]:


from sklearn import metrics


# In[82]:


score1 = metrics.r2_score(y_test,y_pred1)
score2 = metrics.r2_score(y_test,y_pred2)
score3 = metrics.r2_score(y_test,y_pred3)
score4 = metrics.r2_score(y_test,y_pred4)


# In[83]:


print(score1,score2,score3,score4)


# In[84]:


s1 = metrics.mean_absolute_error(y_test,y_pred1)
s2 = metrics.mean_absolute_error(y_test,y_pred2)
s3 = metrics.mean_absolute_error(y_test,y_pred3)
s4 = metrics.mean_absolute_error(y_test,y_pred4)


# In[85]:


print(s1,s2,s3,s4)


# # 15. Predict Charges For New Customer

# In[86]:


data = {'age' : 40,
        'sex' : 1,
        'bmi' : 40.30,
        'children' : 4,
        'smoker' : 1,
        'region' : 2}

df = pd.DataFrame(data,index=[0])
df


# In[87]:


new_pred = gr.predict(df)
print("Medical Insurance cost for new : ",new_pred)


# # 16. Save Model Usign Joblib

# In[88]:


import joblib


# In[89]:


joblib.dump(gr,'model_joblib_test')


# In[90]:


model = joblib.load('model_joblib_test')


# In[91]:


model.predict([[40,1,40.3,4,1,2]])


# # GUI

# In[92]:


from tkinter import *


# In[93]:


import joblib


# In[ ]:


def show_entry():
    
    p1 = float(e1.get())
    p2 = float(e2.get())
    p3 = float(e3.get())
    p4 = float(e4.get())
    p5 = float(e5.get())
    p6 = float(e6.get())

    model = joblib.load('model_joblib_gr')
    result = model.predict([[p1,p2,p3,p4,p5,p6]])
    
    Label(master, text = "Insurance Cost").grid(row=7)
    Label(master, text=result).grid(row=8)


master =Tk()
master.title("Insurance Cost Prediction")
label = Label(master,text = "Insurance Cost Prediction",bg = "black",
               fg = "white").grid(row=0,columnspan=2)

Label(master,text = "Enter Your Age").grid(row=1)
Label(master,text = "Male Or Female [1/0]").grid(row=2)
Label(master,text = "Enter Your BMI Value").grid(row=3)
Label(master,text = "Enter Number of Children").grid(row=4)
Label(master,text = "Smoker Yes/No [1/0]").grid(row=5)
Label(master,text = "Region [1-4]").grid(row=6)


e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)



e1.grid(row=1,column=1)
e2.grid(row=2,column=1)
e3.grid(row=3,column=1)
e4.grid(row=4,column=1)
e5.grid(row=5,column=1)
e6.grid(row=6,column=1)


Button(master,text="Predict",command=show_entry).grid()

mainloop()


# In[ ]:





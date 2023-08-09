#pip install streamlit
#pip install pandas
#pip install sklearn


# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns



df = pd.read_csv(r'C:\Users\kolis\Desktop\diabetes/diabetes.csv')

# HEADINGS
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())


# X AND Y DATA
x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)
# When you change test_size = 0.4, it means that 40% of the data will be allocated to the testing set, and the remaining 60% will be used for training.
#  The effect of this change is that you will have more data in the testing set for evaluating the model's performance, while the training set will have less data for model training.

# Here's what will be present in x_train, x_test, y_train, and y_test when test_size = 0.4:

# x_train: This will contain 60% of the input features (x) randomly selected for training the machine learning model. 
# This will be a larger subset of the data compared to when test_size = 0.2.

# x_test: This will contain 40% of the input features (x) randomly selected for evaluating the model's performance. 
# This will be a smaller subset of the data compared to when test_size = 0.2.

# y_train: This will contain the corresponding target variable values (diabetes outcomes) for the 60% of the data present in x_train. It will be used for training the model.

# y_test: This will contain the corresponding target variable values (diabetes outcomes) for the 40% of the data present in x_test.
#  It will be used for evaluating the model's performance.


# FUNCTION
def user_report():
  pregnancies = st.sidebar.slider('Pregnancies', 0,17, 3 )
  glucose = st.sidebar.slider('Glucose', 0,200, 120 )
  bp = st.sidebar.slider('Blood Pressure', 0,122, 70 )
  skinthickness = st.sidebar.slider('Skin Thickness', 0,100, 20 )
  insulin = st.sidebar.slider('Insulin', 0,846, 79 )
  bmi = st.sidebar.slider('BMI', 0,67, 20 )
  dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0,2.4, 0.47 )
  age = st.sidebar.slider('Age', 21,88, 33 )

  user_report_data = {
      'Pregnancies': pregnancies,
      'Glucose': glucose,
      'BloodPressure': bp,
      'SkinThickness': skinthickness,
      'Insulin': insulin,
      'BMI': bmi,
      'DiabetesPedigreeFunction': dpf,
      'Age': age
     
  }
  report_data = pd.DataFrame(user_report_data, index=[0])

  return report_data




# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)




# MODEL
rf  = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)



#  LISATIONS
st.title('Visualised Patient Report')



# COLOR FUNCTION
if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'


# Age vs Pregnancies
st.header('Pregnancy count Graph (Others vs Yours)')
fig_preg = plt.figure() #This line creates a new empty figure (fig_preg) using the plt.figure() function from the matplotlib.pyplot module. A figure in Matplotlib represents the entire visualization window or canvas.
ax1 = sns.scatterplot(x = 'Age', y = 'Pregnancies', data = df, hue = 'Outcome', palette = 'Greens')#This line creates a scatter plot using Seaborn's scatterplot function. It uses the DataFrame df as the data source, where the x-axis represents the 'Age', the y-axis represents the 'Pregnancies', and the points are colored based on the 'Outcome' variable. The 'Outcome' variable is used as a hue (color) to distinguish between healthy (0) and unhealthy (1) individuals. The palette 'Greens' specifies the color scheme.
ax2 = sns.scatterplot(x = user_data['Age'], y = user_data['Pregnancies'], s = 150, color = color) #This line adds a scatter plot point to the existing plot (ax1) to visualize the user's input data. It plots a single point at the x-coordinate equal to the user's 'Age' and the y-coordinate equal to the user's 'Pregnancies'. The point's size is set to 150, and the color is determined by the color variable, which depends on the model's prediction (healthy or unhealthy).
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_preg)  #This line displays the scatter plot in the Streamlit app using the st.pyplot() function. The plot includes points representing both the original dataset and the user's input data.



# Age vs Glucose
st.header('Glucose Value Graph (Others vs Yours)')
fig_glucose = plt.figure()
ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = df, hue = 'Outcome' , palette='magma')
ax4 = sns.scatterplot(x = user_data['Age'], y = user_data['Glucose'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,220,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_glucose)



# Age vs Bp
st.header('Blood Pressure Value Graph (Others vs Yours)')
fig_bp = plt.figure()
ax5 = sns.scatterplot(x = 'Age', y = 'BloodPressure', data = df, hue = 'Outcome', palette='Reds')
ax6 = sns.scatterplot(x = user_data['Age'], y = user_data['BloodPressure'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,130,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bp)


# Age vs St
st.header('Skin Thickness Value Graph (Others vs Yours)')
fig_st = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'SkinThickness', data = df, hue = 'Outcome', palette='Blues')
ax8 = sns.scatterplot(x = user_data['Age'], y = user_data['SkinThickness'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,110,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_st)


# Age vs Insulin
st.header('Insulin Value Graph (Others vs Yours)')
fig_i = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = df, hue = 'Outcome', palette='rocket')
ax10 = sns.scatterplot(x = user_data['Age'], y = user_data['Insulin'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,900,50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_i)


# Age vs BMI
st.header('BMI Value Graph (Others vs Yours)')
fig_bmi = plt.figure()
ax11 = sns.scatterplot(x = 'Age', y = 'BMI', data = df, hue = 'Outcome', palette='rainbow')
ax12 = sns.scatterplot(x = user_data['Age'], y = user_data['BMI'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,70,5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bmi)


# Age vs Dpf
st.header('DPF Value Graph (Others vs Yours)')
fig_dpf = plt.figure()
ax13 = sns.scatterplot(x = 'Age', y = 'DiabetesPedigreeFunction', data = df, hue = 'Outcome', palette='YlOrBr')
ax14 = sns.scatterplot(x = user_data['Age'], y = user_data['DiabetesPedigreeFunction'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,3,0.2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_dpf)



# OUTPUT
st.subheader('Your Report: ')
output=''
if user_result[0]==0:
  output = 'You are not Diabetic'
else:
  output = 'You are Diabetic'
st.title(output)
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')
#accuracy_score(...): The accuracy_score function from the scikit-learn library is then used to calculate the accuracy of the model's predictions. 
# It compares the predicted outcomes (from rf.predict(x_test)) with the actual target variable values (y_test) and calculates the accuracy score.

#In summary, accuracy_score(y_test, rf.predict(x_test)) is a function call that computes the accuracy of the trained RandomForestClassifier model on the testing data.
#  The accuracy is a metric that measures the proportion of correctly predicted outcomes (samples where the predicted outcome matches the true outcome)
#  out of the total number of testing samples. It indicates how well the model performs on unseen data and is often used to evaluate the performance of classification models. 
# The accuracy score ranges from 0 to 1, where 1 represents a perfect prediction and 0 represents no correct predictions.
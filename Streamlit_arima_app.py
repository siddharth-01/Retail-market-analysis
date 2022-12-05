
from statsmodels.tsa.statespace import sarimax
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np # linear algebra
import seaborn as sns
from itertools import cycle

# Title
st.header("Streamlit Machine Learning Webpage")

# Input bar 1
category= st.selectbox("Select Category", ("Food", "Hobbies","Household"))
#days = st.number_input("Enter Category (1:Food, 2:Hobbies, 3:Household")
start = st.number_input("Enter start day")
end = st.number_input("Enter end day")


# If button is pressed
if st.button("Submit"):
    total_sales_OverCalendar=pd.read_csv('predictions/total_sales_OverCalendar.csv',index_col=0)

    # Unpickle classifier
    if (str(category)=="Food"):
        variable="FOODS"
        shfited=pd.read_csv('predictions/shfited_food.csv',index_col=0)
    elif(str(category)=="Hobbies"):
        variable="HOBBIES"
        shfited=pd.read_csv('predictions/shfited_hobbies.csv',index_col=0)
    elif(str(category)=="Household"):   
        variable="HOUSEHOLD"
        shfited=pd.read_csv('predictions/shfited_household.csv',index_col=0)

    Type_Series = total_sales_OverCalendar[variable]
    Type_Series.fillna(Type_Series.mean(),inplace=True)
    movingAverage = Type_Series.rolling(window=30).mean()
    TypeSeriesDiff = Type_Series-movingAverage
  
    predictions_ARIMA_final = pd.Series(Type_Series.at['d_2'],index=Type_Series.index)
    shfited.loc['d_1'] = 0
    shfited.loc['d_2'] = 0
    movingAverage.fillna(0)
    predictVsActual = pd.DataFrame({'actual':Type_Series,'diffMean':TypeSeriesDiff,
                                    'predictDiffOri':shfited['predicShfited2'],
                                    'predictDiff':shfited['predicShfited2'],
                                    'base':movingAverage})
    predictVsActual['predict'] = predictVsActual.loc[:,['predictDiff','base']].sum(axis=1)
    predictVsActual['error'] = predictVsActual['actual'] - predictVsActual['predict']
    st.set_option('deprecation.showPyplotGlobalUse', False)

    plt.figure(figsize=(20,5))
    plt.plot(predictVsActual['actual'].iloc[int(start):int(end)],label='Actual')
    plt.plot(predictVsActual['predict'].iloc[int(start):int(end)],label='Predicted')
    plt.legend(loc='best')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    plt.show()
    st.pyplot()



        
    st.text(f"This instance is a prediction")
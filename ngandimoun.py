
import streamlit as st
import streamlit as st
import plotly.express as px
import os
import openai
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np # linear algebra
import pandas as pd # for data preparation
import nltk
import PIL
import plotly.express as px # for data visualization
from textblob import TextBlob # for sentiment analysis
from plotly.subplots import make_subplots
import yfinance as yf
import datetime
import numpy as np
import os 
import pandas as pd
import matplotlib.pyplot as plt
import glob
st.set_option('deprecation.showPyplotGlobalUse', False)
import seaborn as sns
#%matplotlib inline

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf
from datetime import date, timedelta
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
from scipy import stats
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import plotly.offline as pyo
sns.set_style('darkgrid')
pyo.init_notebook_mode()
nltk.download('vader_lexicon')
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from nltk.util import ngrams
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import random
from itertools import count
from nltk.util import pr
import re
import string
import logging



ima= PIL.Image.open('C:/Users/PC/Downloads/rosaline/assets/thumb.webp')
bon= PIL.Image.open('C:/Users/PC/Downloads/rosaline/assets/big.jpg')

def page_one():

    st.title('**:violet[NGANDIMOUN ANALYTICS]**' ":100:")
    st.subheader('Analysis with a Single Value' ":blue_heart:")
    st.write('You can upload your Dataset(CSV file) and Perfom Analysis' ":dart:")
        
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        dff = pd.read_csv(uploaded_file)
        
            
        
        dff.isna().sum()

        # For Numerical Type
        dff.select_dtypes(include=(['int64', 'float64'])).isna().sum()

        dff.select_dtypes(include=(['int64', 'float64'])).isna().sum()
        dff_num_col = dff.select_dtypes(include=(['int64', 'float64'])).columns
        for c in dff_num_col:
            dff[c].fillna(dff[c].median(), inplace=True)

        dff.select_dtypes(include=(['int64', 'float64'])).isna().sum()

        # For Categorical type
        dff.select_dtypes(include=('object')).isna().sum()

        dff_cat_col = dff.select_dtypes(include=('object')).columns
        for c in dff_cat_col:
            dff[c].fillna(dff[c].mode().values[0], inplace=True)
            
        dff.select_dtypes(include=('object')).isna().sum()
        
        st.write(dff)
        
        st.write ("**:blue[**:blue[Your Dataset is CLEAN and READY to use]**]**" ":blossom:")
        
        jose0 = st.selectbox('Select column value', dff.columns)
        jose2 = st.number_input("Enter number of ranking", 1, 200)
        
        jose1 = st.text_input("Enter your Title")
        
        
        
        
        
        if st.button('Confirm Selection for entries'):


            #jose0= 'director'

            dff[jose0]=dff[jose0].fillna('No Director Specified')
            filtered_directors=pd.DataFrame()
            filtered_directors=dff[jose0].str.split(',',expand=True).stack()
            filtered_directors=filtered_directors.to_frame()
            filtered_directors.columns=[jose0]
            directors=filtered_directors.groupby([jose0]).size().reset_index(name='Total Content')
            #directors=directors[directors.Director !='No Director Specified']
            directors=directors.sort_values(by=['Total Content'],ascending=False)
            directorsTop5=directors.head(jose2)
            directorsTop5=directorsTop5.sort_values(by=['Total Content'])
            fig1=px.bar(directorsTop5,x=jose0,y='Total Content', color=jose0 ,title=jose1)
            #fig1.show()

            fig1.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="left",
                        active=0,
                        x=0,
                        y=1.5,
                        buttons=list([
                            dict(
                                args=["type", "bar"],
                                label="Bar Plot",
                                method="restyle"
                            ),
                            dict(
                                args=["type", "histogram"],
                                label="Histogram",
                                method="restyle"
                            )
                        ]),
                    ),
                ]
            )


            st.plotly_chart(fig1)
            
            
            fig = px.scatter(directorsTop5, x=jose0,
                        y='Total Content', color=jose0,
                        trendline="ols",
                        title=jose1)




            st.plotly_chart(fig)
            
            
            z = dff.groupby([jose0]).size().reset_index(name='counts')
            directorsTop5=z.head(jose2)
            directorsTop5=directorsTop5.sort_values(by=['counts'])
            pieChart = px.pie(directorsTop5, values='counts', names=jose0, 
                              title=jose1,
                              color_discrete_sequence=px.colors.qualitative.Set3)        
                              
            st.plotly_chart(pieChart)
            



     

#plot two value column x y y must numerical
def page_two():

    
    st.title('**:violet[NGANDIMOUN ANALYTICS]**' ":100:")
    st.subheader('Analysis with Two Values' ":blue_heart:")
    st.write('You can upload your Dataset(CSV file) and Perfom Analysis' ":dart:")
    st.write('Perfom an Analysis from Two Values **:red[but the Second Value must be Numerical]**')
        
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv") 
    
    
    
    if uploaded_file is not None:
        dff1 = pd.read_csv(uploaded_file)
        
        dff1.isna().sum()

        # For Numerical Type
        dff1.select_dtypes(include=(['int64', 'float64'])).isna().sum()

        dff1.select_dtypes(include=(['int64', 'float64'])).isna().sum()
        dff1_num_col = dff1.select_dtypes(include=(['int64', 'float64'])).columns
        for c in dff1_num_col:
            dff1[c].fillna(dff1[c].median(), inplace=True)

        dff1.select_dtypes(include=(['int64', 'float64'])).isna().sum()

        # For Categorical type
        dff1.select_dtypes(include=('object')).isna().sum()

        dff1_cat_col = dff1.select_dtypes(include=('object')).columns
        for c in dff1_cat_col:
            dff1[c].fillna(dff1[c].mode().values[0], inplace=True)
            
        dff1.select_dtypes(include=('object')).isna().sum()
        
        st.write(dff1)
        
        st.write ("**:blue[Your Dataset is CLEAN and READY to use]**" ":blossom:")
        
        jose1 = st.selectbox('Select the ONE column for the values', dff1.columns)
        jose2 = st.selectbox('Select the SECOND column for the values', dff1.columns)
        jose22 = st.text_input("Enter your Title")
        if st.button('Confirm Selection for two entries'):
        
            figure21 = px.bar(dff1, x=jose1, 
                y = jose2, 
                color = jose2,
            title=jose22)
            
            figure21.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1,
                                 step="day",
                                 stepmode="backward"),
                        ])
                    ),
                    rangeslider=dict(
                        visible=True
                    ),
                )
            )
            st.plotly_chart(figure21)
            
            
            st.write("##")
            
            
            figure22 = px.box(dff1, x=jose1, 
             y=jose2, 
             color=jose2)
             
            figure22.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1,
                                 step="day",
                                 stepmode="backward"),
                        ])
                    ),
                    rangeslider=dict(
                        visible=True
                    ),
                )
            )
             
            st.plotly_chart(figure22)
             
        
    
#REL  TIONSHIP BETWEEN TWO VALUES COLUMNS //// TWO NULERICAL VALUE
def page_three():


    
    st.title('**:violet[NGANDIMOUN ANALYTICS]**' ":100:")
    st.subheader('Analysis two Values' ":blue_heart:")
    st.write('You can upload your Dataset(CSV file) and Perfom Analysis' ":dart:")
    st.write('Perfom an Analysis **:red[from Two Numerical Values]**')
        
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    
    if uploaded_file is not None:
        dff3 = pd.read_csv(uploaded_file)
        
        dff3.isna().sum()

        # For Numerical Type
        dff3.select_dtypes(include=(['int64', 'float64'])).isna().sum()

        dff3.select_dtypes(include=(['int64', 'float64'])).isna().sum()
        dff3_num_col = dff3.select_dtypes(include=(['int64', 'float64'])).columns
        for c in dff3_num_col:
            dff3[c].fillna(dff3[c].median(), inplace=True)

        dff3.select_dtypes(include=(['int64', 'float64'])).isna().sum()

        # For Categorical type
        dff3.select_dtypes(include=('object')).isna().sum()

        dff3_cat_col = dff3.select_dtypes(include=('object')).columns
        for c in dff3_cat_col:
            dff3[c].fillna(dff3[c].mode().values[0], inplace=True)
            
        dff3.select_dtypes(include=('object')).isna().sum()
        
        st.write(dff3)
        
        
        
        st.write ("**:blue[Your Dataset is CLEAN and READY to use]**" ":blossom:")
        
        
        jose3 = st.selectbox('Select the ONE Numerical column', dff3.columns)
        jose4 = st.selectbox('Select the SECOND Numerical column ', dff3.columns)
        jose5 = st.text_input("Enter your Title")
        
        if st.button('Confirm Selection for two entries'):
        
            figure31 = px.scatter(data_frame = dff3, x=jose3, color= jose4,
                        y=jose4, size=jose4, trendline="ols", title=jose5 )
            
            figure31.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1,
                                 step="day",
                                 stepmode="backward"),
                        ])
                    ),
                    rangeslider=dict(
                        visible=True
                    ),
                )
            )
            st.plotly_chart(figure31)
            
            
            figure32 = px.line(data, x=jose3, y=jose4, color= jose4,
                 title=jose5)
            figure32.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1,
                                 step="day",
                                 stepmode="backward"),
                        ])
                    ),
                    rangeslider=dict(
                        visible=True
                    ),
                )
            )
            st.plotly_chart(figure32)
            
            figure33 = px.bar(data, x=jose3, y=jose4, color= jose4,
                 title=jose5)
            figure33.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1,
                                 step="day",
                                 stepmode="backward"),
                        ])
                    ),
                    rangeslider=dict(
                        visible=True
                    ),
                )
            )
                 
            st.plotly_chart(figure33)



#NUMBER of oocurence of one value
def page_four():


    
    st.title('**:violet[NGANDIMOUN ANALYTICS]**' ":100:")
    st.subheader('Analysis with a Single Value' ":blue_heart:")
    st.write('You can upload your Dataset(CSV file) and Perfom Analysis' ":dart:")
    st.write('Perfom an Analysis on the number of Occurence')
        
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")


    if uploaded_file is not None:
        dff4 = pd.read_csv(uploaded_file)
        
        dff4.isna().sum()

        # For Numerical Type
        dff4.select_dtypes(include=(['int64', 'float64'])).isna().sum()

        dff4.select_dtypes(include=(['int64', 'float64'])).isna().sum()
        dff4_num_col = dff4.select_dtypes(include=(['int64', 'float64'])).columns
        for c in dff4_num_col:
            dff4[c].fillna(dff4[c].median(), inplace=True)

        dff4.select_dtypes(include=(['int64', 'float64'])).isna().sum()

        # For Categorical type
        dff4.select_dtypes(include=('object')).isna().sum()

        dff4_cat_col = dff4.select_dtypes(include=('object')).columns
        for c in dff4_cat_col:
            dff4[c].fillna(dff4[c].mode().values[0], inplace=True)
            
        dff4.select_dtypes(include=('object')).isna().sum()
        
        st.write(dff4)
        
        
        st.write ("**:blue[Your Dataset is CLEAN and READY to use]**" ":blossom:")
        
        
        jose41 = st.selectbox('Select column value', dff4.columns)
        
        
        jose42 = st.text_input("Enter your Title")
        
        if st.button('Confirm Selection for two entries'):
        
            figure41 = px.histogram(dff4, x = jose41, 
                      color = jose41, 
                      title= jose42)
                      
            figure41.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1,
                                 step="day",
                                 stepmode="backward"),
                        ])
                    ),
                    rangeslider=dict(
                        visible=True
                    ),
                )
            )
            st.plotly_chart(figure41)
            
            figure42 = px.bar(dff4, x = jose41,
                title=jose42)
            figure42.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1,
                                 step="day",
                                 stepmode="backward"),
                        ])
                    ),
                    rangeslider=dict(
                        visible=True
                    ),
                )
            )
            st.plotly_chart(figure42)
        
        
#plot 3 values
def page_five():


    
    st.title('**:violet[NGANDIMOUN ANALYTICS]**' ":100:")
    st.subheader('Analysis with Three Values' ":blue_heart:")
    st.write('You can upload your Dataset(CSV file) and Perfom Analysis' ":dart:")
    
        
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        dff5 = pd.read_csv(uploaded_file)
        
        dff5.isna().sum()

        # For Numerical Type
        dff5.select_dtypes(include=(['int64', 'float64'])).isna().sum()

        dff5.select_dtypes(include=(['int64', 'float64'])).isna().sum()
        dff5_num_col = dff5.select_dtypes(include=(['int64', 'float64'])).columns
        for c in dff5_num_col:
            dff5[c].fillna(dff5[c].median(), inplace=True)

        dff5.select_dtypes(include=(['int64', 'float64'])).isna().sum()

        # For Categorical type
        dff5.select_dtypes(include=('object')).isna().sum()

        dff5_cat_col = dff5.select_dtypes(include=('object')).columns
        for c in dff5_cat_col:
            dff5[c].fillna(dff5[c].mode().values[0], inplace=True)
            
        dff5.select_dtypes(include=('object')).isna().sum()
        
        st.write(dff5)
        
        
        st.write ("**:blue[Your Dataset is CLEAN and READY to use]**" ":blossom:")
        

        jose51 = st.selectbox('Select the ONE column values', dff5.columns)
        jose52 = st.selectbox('Select the SECOND column values', dff5.columns)
        jose53 = st.selectbox('Select the Third column values', dff5.columns)

        if st.button('Confirm Selection for Three entries'):
        
            dff5=dff5.groupby([jose51,jose52,jose53]).size().reset_index(name='Total Content')
            fig5 = px.scatter_3d(dff5, x=jose51, y=jose52, z=jose53,
                                size=None, symbol=None)
                                
            st.plotly_chart(fig5)

#y axis must to be description or text on the dataset
def page_six():


    
    st.title('**:violet[NGANDIMOUN ANALYTICS]**' ":100:")
    st.subheader('Perform a Sentiment Analysis' ":blue_heart:")
    st.write('You can upload your Dataset(CSV file) and Perfom Analysis' ":dart:")
    
        
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        dff6 = pd.read_csv(uploaded_file)
        
        dff6.isna().sum()

        # For Numerical Type
        dff6.select_dtypes(include=(['int64', 'float64'])).isna().sum()

        dff6.select_dtypes(include=(['int64', 'float64'])).isna().sum()
        dff6_num_col = dff6.select_dtypes(include=(['int64', 'float64'])).columns
        for c in dff6_num_col:
            dff6[c].fillna(dff6[c].median(), inplace=True)

        dff6.select_dtypes(include=(['int64', 'float64'])).isna().sum()

        # For Categorical type
        dff6.select_dtypes(include=('object')).isna().sum()

        dff6_cat_col = dff6.select_dtypes(include=('object')).columns
        for c in dff6_cat_col:
            dff6[c].fillna(dff6[c].mode().values[0], inplace=True)
            
        dff6.select_dtypes(include=('object')).isna().sum()
        
        #text cleaning

        st.write(dff6)
        
        
        
        f_data = dff6
        
        st.write ("**:blue[Your Dataset is CLEAN and READY to use]**" ":blossom:")
        
        
        jose61 = st.selectbox('Select the ONE column ', dff6.columns)
        st.write("The Second column must refer to Description or Descriptive Text")
        jose62 = st.selectbox('Select the SECOND column', dff6.columns)


        
        if st.button('Confirm Selection for two entries'):
        
            f_data = dff6
            f_data.text =f_data[jose62].str.lower()

            #Remove twitter handlers
            f_data.text = f_data.text.apply(lambda x:re.sub('@[^\s]+','',x))

            #remove hashtags
            f_data.text = f_data.text.apply(lambda x:re.sub(r'\B#\S+','',x))


            # Remove URLS
            f_data.text = f_data.text.apply(lambda x:re.sub(r"http\S+", "", x))

            # Remove all the special characters
            f_data.text = f_data.text.apply(lambda x:' '.join(re.findall(r'\w+', x)))

            #remove all single characters
            f_data.text = f_data.text.apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', '', x))

            # Substituting multiple spaces with single space
            f_data.text = f_data.text.apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))
        
        
            sid = SIA()
            f_data['sentiments']           = f_data[jose62].apply(lambda x: sid.polarity_scores(' '.join(re.findall(r'\w+',x.lower()))))
            f_data['Positive Sentiment']   = f_data['sentiments'].apply(lambda x: x['pos']+1*(10**-6)) 
            f_data['Neutral Sentiment']    = f_data['sentiments'].apply(lambda x: x['neu']+1*(10**-6))
            f_data['Negative Sentiment']   = f_data['sentiments'].apply(lambda x: x['neg']+1*(10**-6))

            f_data.drop(columns=['sentiments'],inplace=True)
            
            
            fig62, ax = plt.subplots(2, 1)

            ax[0].set_title('Distribution Of Sentiments', fontsize=19, fontweight='bold')
            sns.kdeplot(f_data['Negative Sentiment'], bw=0.1, ax=ax[0])
            sns.kdeplot(f_data['Positive Sentiment'], bw=0.1, ax=ax[0])
            sns.kdeplot(f_data['Neutral Sentiment'], bw=0.1, ax=ax[0])
            ax[0].set_xlabel('Sentiment Value', fontsize=19)

            ax[1].set_title('Cummulative Distribution Function Of Sentiments', fontsize=19, fontweight='bold')
            sns.kdeplot(f_data['Negative Sentiment'], bw=0.1, cumulative=True, ax=ax[1])
            sns.kdeplot(f_data['Positive Sentiment'], bw=0.1, cumulative=True, ax=ax[1])
            sns.kdeplot(f_data['Neutral Sentiment'], bw=0.1, cumulative=True, ax=ax[1])
            ax[1].set_xlabel('Sentiment Value', fontsize=19)

            plt.tight_layout()
            st.pyplot(fig62)
        
            #dfx=dff6[['release_year','description']]
            dfx=dff6[[jose61, jose62]]
            #dfx=dfx.rename(columns={'release_year':'Release Year'})
            for index,row in dfx.iterrows():
                z=row[jose62]
                testimonial=TextBlob(z)
                p=testimonial.sentiment.polarity
                if p==0:
                    sent='Neutral'
                elif p>0:
                    sent='Positive'
                else:
                    sent='Negative'
                dfx.loc[[index,2],'Sentiment']=sent


            dfx=dfx.groupby([jose61,'Sentiment']).size().reset_index(name='Total Content')

            #dfx=dfx[dfx['Release Year']>=2010]
            fig61 = px.bar(dfx, x=jose61, y="Total Content", color="Sentiment", title="Sentiment of Content")
            
            fig61.update_layout(
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1,
                                     step="day",
                                     stepmode="backward"),
                            ])
                        ),
                        rangeslider=dict(
                            visible=True
                        ),
                    )
                )
            st.plotly_chart(fig61)
            
            Most_Positive = f_data[f_data['Positive Sentiment'].between(0.4,1)]
            Most_Negative = f_data[f_data['Negative Sentiment'].between(0.25,1)]

            Most_Positive_text = ' '.join(Most_Positive[jose62])
            Most_Negative_text = ' '.join(Most_Negative[jose62])


            pwc = WordCloud(width=600,height=400,collocations = False).generate(Most_Positive_text)
            nwc = WordCloud(width=600,height=400,collocations = False).generate(Most_Negative_text)

            fig63, ax = plt.subplots(1, 2)

            ax[0].set_title('Words Most Positive', fontsize=16, fontweight='bold')
            ax[0].imshow(pwc)
            ax[0].axis('off')

            ax[1].set_title('Words Most Negative', fontsize=16, fontweight='bold')
            ax[1].imshow(nwc)
            ax[1].axis('off')

            plt.subplots_adjust(wspace=0.5)  # Increase the horizontal space between the subplots

            st.pyplot(fig63)
            
            
def page_seven():


    
    st.title('**:violet[NGANDIMOUN ANALYTICS]**' ":100:")
    st.subheader('Generative AI applied to Data Science' ":blue_heart:")
    
    
    st.write('You can upload your Dataset(CSV file) and Perfom Analysis' ":dart:")
       
    
    NGANDIMOUN = st.text_area("**:violet[Question:]**")
    button = st.button("**NGANDIMOUN:**")


        
    if button and NGANDIMOUN:
        st.spinner("Almost Ready...")

        data = pd.read_csv('TSLA.csv')
        
        data.isna().sum()

        # For Numerical Type
        data.select_dtypes(include=(['int64', 'float64'])).isna().sum()

        data.select_dtypes(include=(['int64', 'float64'])).isna().sum()
        data_num_col = data.select_dtypes(include=(['int64', 'float64'])).columns
        for c in data_num_col:
            data[c].fillna(data[c].median(), inplace=True)

        data.select_dtypes(include=(['int64', 'float64'])).isna().sum()

        # For Categorical type
        data.select_dtypes(include=('object')).isna().sum()

        data_cat_col = data.select_dtypes(include=('object')).columns
        for c in data_cat_col:
            data[c].fillna(data[c].mode().values[0], inplace=True)
            
        data.select_dtypes(include=('object')).isna().sum()
        
        st.write(data)
        
        st.write ("**:blue[Your Dataset is CLEAN and READY to use]**" ":blossom:")
        
        today = date.today()

        d1 = today.strftime("%Y-%m-%d")
        end_date = d1
        d2 = date.today () - timedelta(days=365)
        d2 = d2.strftime("%Y-%m-%d")

        start_date = d2
        
        figure71 = go.Figure(data=[go.Candlestick(x=data["Date"],
                                                open=data["Open"], high=data["High"],
                                                low=data["Low"], close=data["Close"])])
        figure71.update_layout(title = "Stock Price Analysis", xaxis_rangeslider_visible=False)
        st.plotly_chart(figure71)

        figure72 = px.bar(data, x = "Date", y= "Close")
        
        st.plotly_chart(figure72)

        figure73 = px.line(data, x='Date', y='Close', 
                         title='Stock Market Analysis with Rangeslider')
        figure73.update_xaxes(rangeslider_visible=True)
        
        st.plotly_chart(figure73)

        figure74 = px.line(data, x='Date', y='Close', 
                         title='Stock Market Analysis with Time Period Selectors')

        figure74.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
       
        st.plotly_chart(figure74)
        
def page_huit():

    st.title('**:violet[NGANDIMOUN ANALYTICS]**' ":100:")
    st.subheader('Generative AI applied to Data Science' ":blue_heart:")
    
    
    st.write('You can upload your Dataset(CSV file) and Perfom Analysis' ":dart:")
    
    NGANDIMOUN = st.text_area("**:violet[Question:]**")
    button = st.button("**NGANDIMOUN:**")


        
    if button and NGANDIMOUN:
        st.spinner("Almost Ready...")
        
        files = os.path.join(r'C:\Users\PC\Downloads\rosaline\Sales_Data', "*.csv")

        files = glob.glob(files)
        df = pd.concat(map(pd.read_csv, files), ignore_index=True)
        df.to_csv('all_months_data',index= False )
        
        data8 = pd.read_csv('all_months_data')


        #read final data

        final_data = pd.read_csv('all_months_data')
        #final_data.head()

        final_data.isnull().sum()

        remove_nan = final_data.dropna(how= 'all')

        final_data= remove_nan.copy()
        #final_data.head()

        # removing or from Month column

        final_data= final_data[final_data['Order Date'].str[0:2] != 'or']

        final_data= final_data[final_data['Quantity Ordered'] != 'Quantity Ordered']

        #final_data.head()

        #convert columns to the correct type

        final_data['Quantity Ordered'] = pd.to_numeric(final_data['Quantity Ordered'])

        final_data['Price Each'] = pd.to_numeric(final_data['Quantity Ordered'])


        #task 2 adding Month column

        final_data['Month'] = final_data['Order Date'].str[1:2]

        #final_data.head()

        #Add a sales column

        final_data['Sales'] = final_data['Quantity Ordered']*final_data['Price Each']
        
        st.write ("**:red[Question I:]** What was the best month for Sales? How much was earned that month?")
        grouped_data = final_data.groupby('Month').sum()
        st.write(grouped_data)
        
        final_data.groupby('Month').sum().plot(kind = 'bar')
        st.pyplot()

        
        
        
        
        
        st.write("**:red[Question II:]** Which city has highest number of sales?")
        final_data['city']= final_data['Purchase Address'].apply(lambda x: x.split(',')[1])
        #final_data.head()
        
        final_data.groupby('city').sum().plot(kind = 'bar')
        st.pyplot()
        
        
        
        st.write("**:red[Question III:]** What time should we display advertisements to increase customer likelihood of buying product?")
        final_data['Order Date'] = pd.to_datetime(final_data['Order Date'])

        final_data['Hour'] = final_data['Order Date'].dt.hour

        final_data['Minute'] = final_data['Order Date'].dt.minute

        
        
        
        

        hours = [hour for hour, df in final_data.groupby('Hour')]

        fig83, ax = plt.subplots()


        ax.plot(hours, final_data.groupby(['Hour']).count())

        ax.set_xticks(hours)
        ax.set_xlabel('Hour')
        ax.set_ylabel('Number of Orders')
        ax.grid()

        st.pyplot(fig83)
    
    



#descriptive stat describe dataset , describe users pattern users behavior 

st.sidebar.title('NGANDIMOUN ANALYTICS' ":bar_chart:")

st.sidebar.title('AI, ML, LLM' ":heart:")
selected_page = st.sidebar.radio('Select your Analysis:', ["Analysis with a Single Value", 'Analysis Two Values', 'Analysis TWO Values', 'Analysis of Occurence', 'Analysis with Three Values', 'Sentiment Analysis', 'Stock Analytics','Generative AI'])


# Create some space between the top and bottom images
st.sidebar.empty()

# Open and resize the bottom image

bon = bon.resize((350, 150))

# Display the bottom image in the sidebar
st.sidebar.image(bon, width=300)

# Use a conditional statement to determine which page to display
if selected_page == 'Analysis with a Single Value':
    page_one()
elif selected_page == 'Analysis Two Values':
    page_two()
elif selected_page == 'Analysis TWO Values':
    page_three()
elif selected_page == 'Analysis of Occurence':
    page_four()
elif selected_page == 'Analysis with Three Values':
    page_five()
elif selected_page == 'Sentiment Analysis':
    page_six()
    
elif selected_page == 'Stock Analytics':
    page_seven()
elif selected_page == 'Generative AI':
    page_huit()

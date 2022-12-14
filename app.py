#%%writefile app.py

import pickle
import numpy as np
import streamlit as st
import os
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans 
import pandas as pd
from pathlib import Path
import streamlit_authenticator as stauth


names = ["Mobius DA"]
usernames = ["Mobius_Data_Analytics"]

file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

credentials = {"usernames":{}}
for un, name, pw in zip(usernames, names, hashed_passwords):
    user_dict = {"name":name,"password":pw}
    credentials["usernames"].update({un:user_dict}) 
authenticator = stauth.Authenticate(credentials,"RE","abc123",cookie_expiry_days=0)

name,authetication_status,username = authenticator.login("LOGIN","main")
hide_streamlit_style = """<style> #MainMenu {visibility: hidden;}footer {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

if authetication_status == False:
    st.error("Username/Password is incorrect")
if authetication_status == None:
    st.warning("Please enter your Username and Password")
    
#------ IF USER AUTHENTICATION STATUS IS TRUE  -----------    
    
if authetication_status:    
    dealer = pd.read_csv("Recommendation Dealer data.csv") 
    dealer = dealer.dropna()
    product = pd.read_csv("Recommendation Product data.csv",usecols=['Product','Quantity','Dealer Name']) 
    Kmeans_Model = KMeans(n_clusters=3,random_state=1)
    dealer_df1 = dealer.copy()
    dealer_df1['Dealer Zone'] = pd.Categorical(dealer_df1['Dealer Zone']).codes
    dealer_df1['Dealer Location'] = pd.Categorical(dealer_df1['Dealer Location']).codes
    dealer_df1['Dealer Investment Capacity (lakhs)'] = pd.Categorical(dealer_df1['Dealer Investment Capacity (lakhs)']).codes
    dealer_df1['Dealer type'] = pd.Categorical(dealer_df1['Dealer type']).codes
    pred = Kmeans_Model.fit_predict(dealer_df1.iloc[:,1:])
    dealer['Clusters'] = pred
    full_df = pd.merge(product,dealer,how='left',on='Dealer Name')
    Product_list = list(full_df['Product'].unique())


    @st.cache()
    
    def cluster_prediction(zone,Location,Investment,Experience,dealer_type):
        if zone == "North":
            zone = 0
        elif zone == "South":
            zone = 1
        if Location == "Bangalore":
            Location = 0
        elif Location == "Chennai":
            Location = 1
        elif Location == "Delhi":
            Location = 2
        elif Location == "Lucknow":
            Location = 3
        elif Location == "Srinagar":
            Location = 4
        if Investment  == "10-15 lacs":
            Investment = 0
        elif Investment  == "5-10 lacs":
            Investment = 1
        elif Investment  == "2-5 lacs":
            Investment = 2
        if dealer_type == "Existing":
            dealer_type = 0
        elif dealer_type == "New":
            dealer_type = 1
        cluster = Kmeans_Model.predict([[zone,Location,Investment,Experience,dealer_type]])
        identified_cluster = cluster[0]
        return identified_cluster
        
        
    def main():
        authenticator.logout("Logout",'sidebar')
        
        tab1, tab2 = st.tabs(['File Upload','Predict Recommendations'])
        with tab1:
            st.image("""https://www.packaginginnovation.com/wp-content/uploads/2015/03/food-logo-groot.png""")
            st.markdown('<p style="font-family:sans-serif; color:black;text-align:left; font-size: 20px;"><b>Upload a file to Predict the output!</b></p>',unsafe_allow_html = True)
            uploaded_file = st.file_uploader(" ")
            if uploaded_file is not None:
            # To predict a test dataframe!!!
                dataframe = pd.read_csv(uploaded_file)                
                test = dataframe.copy()
                test['Dealer Zone'] = test['Dealer Zone'].replace({'North':0,'South':1})
                test['Dealer Location'] = test['Dealer Location'].replace({'Bangalore':0,'Chennai':1,'Delhi':2,'Lucknow':3,'Srinagar':4})
                test['Dealer Investment Capacity (lakhs)'] = test['Dealer Investment Capacity (lakhs)'].replace({'10-15 lacs':0,'2-5 lacs':1,'5-10 lacs':2})
                test['Dealer type'] = test['Dealer type'].replace({'Existing':0,'New':1})
                output = pd.DataFrame()
                for idx in range(len(test)):  
                    values = test.loc[idx][:5].values
                    pdt = test.loc[idx]['Product']
                    identified_cluster = Kmeans_Model.predict([values])[0]
                    data =full_df[full_df['Clusters']==identified_cluster]
                    data_pivot = pd.pivot_table(data, values='Quantity', index='Product', columns='Dealer Name').fillna(0)
                    data_matrix = csr_matrix(data_pivot.values)
                    Model_knn = NearestNeighbors(metric='cosine',algorithm='brute')
                    Model_knn.fit(data_matrix)
                    #test['Recommendations'] = np.NaN
                    if not data_pivot.loc[pdt].empty:
                        recom = list(data_pivot.iloc[Model_knn.kneighbors(data_pivot.loc[pdt].values.ravel().reshape(1,-1),n_neighbors=6)[1].ravel()].index)[1:]
                        recom = set(recom) - set(pdt)
                        recom = " | ".join([str(item) for item in recom])
                        temp = dataframe.loc[idx]
                        temp['Recommendations'] = recom
                        output = output.append(temp)                                                
                dataframe_org = output.copy()
                dataframe_org = dataframe_org.drop('Dealer type',axis=1)
                dataframe_org = dataframe_org[['Dealer Name','Dealer Zone','Dealer Location','Dealer Investment Capacity (lakhs)','Dealer Experience (years)','Product','Recommendations']]
                st.markdown('<p style="font-family:sans-serif; color:black;text-align:left; font-size: 12px;"><b>Table: List of Recommendations from Similar Dealers for each Dealer</b></p>',unsafe_allow_html = True)
                st.write(dataframe_org)
                dataframe_org =dataframe_org.to_csv(index=False).encode('utf-8')
                st.download_button(label='Download recommendations',data=dataframe_org,mime='text/csv',file_name='Download.csv')   
        with tab2:
            st.markdown('<p style="font-family:sans-serif; color:black;text-align:left; font-size: 14px;"><b>Enter the Dealer information</b></p>',unsafe_allow_html = True)
            name = st.text_input("Dealer Name")
            #st.sidebar.write('Entered Dealer Name is', name)
            zone = st.selectbox('Dealer zone',("North","South"))
            if zone == "North":
                Location = st.selectbox('Dealer Location',("Delhi","Lucknow","Srinagar"))
            else:
                Location = st.selectbox('Dealer Location',("Bangalore","Chennai",))
            Investment = st.selectbox('Dealer Investment Capacity (lakhs)',("10-15 lacs","5-10 lacs","2-5 lacs"))
            Experience = st.number_input('Dealer Experience',step=1)
            dealer_type = st.selectbox('Dealer Type',("Existing","New"))
            product_ordered = st.multiselect('Products already ordered',(Product_list))
            if zone == "North":
                zone_en = 0
            elif zone == "South":
                zone_en = 1
            if Location == "Bangalore":
                Location_en = 0
            elif Location == "Chennai":
                Location_en = 1
            elif Location == "Delhi":
                Location_en = 2
            elif Location == "Lucknow":
                Location_en = 3
            elif Location == "Srinagar":
                Location_en = 4
            if Investment  == "10-15 lacs":
                Investment_en = 0
            elif Investment  == "5-10 lacs":
                Investment_en = 1
            elif Investment  == "2-5 lacs":
                Investment_en = 2
            if dealer_type == "Existing":
                dealer_type_en = 0
            elif dealer_type == "New":
                dealer_type_en = 1
            values = [zone_en,Location_en,Investment_en,Experience,dealer_type_en]
            identified_cluster = Kmeans_Model.predict([values])[0]
            data =full_df[full_df['Clusters']==identified_cluster]
            data_pivot = pd.pivot_table(data, values='Quantity', index='Product', columns='Dealer Name').fillna(0)
            data_matrix = csr_matrix(data_pivot.values)
            Model_knn = NearestNeighbors(metric='cosine',algorithm='brute')
            Model_knn.fit(data_matrix)
            output = pd.DataFrame()
            recom_list = []
            recom_output = pd.DataFrame()    
            for pdt in product_ordered:
                try:      ## run this if the entered product is available within clusters  
                    
                    recom = list(data_pivot.iloc[Model_knn.kneighbors(data_pivot.loc[pdt].values.ravel().reshape(1,-1),n_neighbors=5)[1].ravel()].index)[1:]
                    distances = Model_knn.kneighbors(data_pivot.loc[pdt].values.ravel().reshape(1,-1),n_neighbors=5)[0].ravel()[1:]                
                    dictt = {'Product':pdt, 'Recom_products':recom,'Similarity':distances}               
                    #recom_comb = " | ".join([str(item) for item in recom])
                    temp_output = pd.DataFrame({"Dealer Name":name,"Dealer zone":zone,"Dealer Location":Location,"Dealer Investment Capacity (lakhs)":Investment,"Dealer Experience":Experience,"Dealer type":dealer_type,"Products ordered":pdt,"Recommendations":[recom]},index=[0])
                    output = output.append(temp_output)                
                    x = pd.DataFrame({'Product already ordered': pdt, 'Recommendations based on similar dealers':[recom]})
                    recom_output = recom_output.append(x)
                    
                except: ## run this if the entered product is not available within clusters  
                    data_pivot_1 = pd.pivot_table(full_df, values='Quantity', index='Product', columns='Dealer Name').fillna(0)
                    print(data_pivot_1)
                    data_matrix_1 = csr_matrix(data_pivot_1.values)
                    Model_knn_1 = NearestNeighbors(metric='cosine',algorithm='brute')
                    Model_knn_1.fit(data_matrix_1)           
                    recom = list(data_pivot_1.iloc[Model_knn_1.kneighbors(data_pivot_1.loc[pdt].values.ravel().reshape(1,-1),n_neighbors=5)[1].ravel()].index)[1:]
                    #recom = set(recom) - set(pdt)
                    #recom_comb = " | ".join([str(item) for item in recom])
                    distances = Model_knn_1.kneighbors(data_pivot_1.loc[pdt].values.ravel().reshape(1,-1),n_neighbors=5)[0].ravel()[1:]
                    dictt = {'Product':pdt, 'Recom_products':recom,'Similarity':distances}
                    temp_output = pd.DataFrame({"Dealer Name":name,"Dealer zone":zone,"Dealer Location":Location,"Dealer Investment Capacity (lakhs)":Investment,"Dealer Experience":Experience,"Dealer type":dealer_type,"Products ordered":pdt,"Recommendations":[recom]},index=[0])
                    output = output.append(temp_output)
                    #recom_list.append(recom)
                    y = pd.DataFrame({'Product already ordered': pdt,'Recommendations based on similar dealers':[recom]})
                    recom_output =recom_output.append(y)
                    
            if len(product_ordered)==0:
                recom = list(full_df[full_df['Clusters']==identified_cluster]['Product'].sample(5))
                #recom = list(data_pivot.index[:5])
                recom_comb = " | ".join([str(item) for item in recom])            
                recom_list.append(recom)
                temp_output = pd.DataFrame({"Dealer Name":name,"Dealer zone":zone,"Dealer Location":Location,"Dealer Investment Capacity (lakhs)":Investment,"Dealer Experience":Experience,"Dealer type":dealer_type,"Products ordered":0,"Recommendations":recom_list},index=[1])
                output = output.append(temp_output)
                recom_output = pd.DataFrame({'Product already ordered': 'NA', 'Recommendations based on similar dealers':recom_list},index=[1]) 

            if st.button("RECOMENDATIONS"): 
                st.subheader(" ")
                st.markdown("Product Recommendations for the New Dealer based on similar dealers".format(', '.join(product_ordered)))
                recom_output = recom_output.drop_duplicates(subset='Product already ordered')
                st.dataframe(recom_output)  
                st.subheader("")
                st.markdown("Download the recommendations for the New Dealer")
                st.dataframe(output)
                output =output.to_csv(index=True).encode('utf-8')
                st.download_button(label='Download CSV',data=output,mime='text/csv',file_name='Download.csv')     
               
                
                   
            
    if __name__=='__main__': 
        main()

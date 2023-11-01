#import streamlit_authenticator as stauth
#import database as db
#from deta import Deta
import streamlit as st
import time as t
import requests
from streamlit_lottie import st_lottie
import json
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import random
import plotly.express as px
import plotly.graph_objects as go



#loading the diabetes data saved model
loaded_model_2 = pickle.load(open("D_trained_model.sav","rb"))


def predict_diabetes(input_data_D):
    # Making user-based predictions
    #input_data = (1,113,98,32,87,34.4,0.617,63)
    # For ease, we will convert this data to a numpy array
    input_data_array = np.asarray(input_data_D)
    # Reshaping the array to give only one data instance
    reshaped_input = input_data_array.reshape(1, -1)

    #standardizing the input data
    #std_data = scaler.transform(reshaped_input)
    #print(std_data)

    prediction = loaded_model_2.predict(reshaped_input)


    if prediction[0] == 0:
        st.success('DOES NOT HAVE DIABETES')
        st.snow()
    elif prediction[0] == 1:
        st.warning('MAY HAVE DIABETES')








def main():
    def load_lottie(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    st.set_page_config(page_title="MACHINE LEARNING MODEL FOR DIABETES",page_icon=" 💉",layout="wide")
    lottie_diabetes_animate = load_lottie("animation_lkfl0fus.json")

    
    #title
    st.title(" :violet[WELCOME TO DIABETES PREDICTION MODEL :stethoscope:]  " )
    #image addtion
    #st.image("DiabetesPredictioninMachineLearningusingPython20220505114933 - Copy.jpg")
    
    t1 , t2 ,t3= st.tabs(["DIABETES PREDICTOR","DATA VISUALIZATIONS CORNER","VIEW THE HYPERPLANE OF THE MODEL"])
    with t1:

            #header file
            st.header(" :blue[Get your diabetes data monitored here ] ")
            st_lottie(lottie_diabetes_animate,height=300,width=300,speed=1,reverse=True)


            st.caption(" #### :violet[*WITHOUT SUGAR, YOU ARE STILL SWEET*] ")



            

                


            st.subheader(":green[ENTER YOUR HEALTH DETAILS HERE]")
            #WARNING MESSAGE
            st.warning("""  ----- NOTE  -----  \n ENTER DETAILS CORRECTLY FOR ACCURATE EVALUATION OF THE RESULTS, OTHERWISE THE MODEL WILL GENERATE FALSE RESULTS WHICH MAY BE INAPPROPRIATE """)

            

            a = st.radio("GENDER",["MALE","FEMALE"])

                
                    

            #taking user input
            if a == "MALE":
                Pregnancies = 0
            elif a == "FEMALE":
                Pregnancies = st.slider("Number of pregnancies",0,10)

                    
                
            Glucose = st.number_input("Your Fasting Blood Sugar Level(mg/dl):  ")
            with st.expander("TAP TO KNOW MORE"):
                st.write(":green[_A normal fasting blood sugar level is usually considered to be between 70 and 99 milligrams per deciliter (mg/dL)_]")
                
            BloodPressure = st.number_input("Your Resting Blood Pressure(mm Hg): ")
            with st.expander("TAP TO LEARN MORE"):
                st.write("""The general range for resting blood pressure is typically measured in millimeters of mercury (mm Hg) and is categorized as follows:
                                                    \n
        - Normal : Systolic (top number) less than 120 mm Hg and diastolic (bottom number) less than 80 mm Hg.
        - Elevated : Systolic between 120-129 mm Hg and diastolic less than 80 mm Hg.
        - Hypertension Stage 1 : Systolic between 130-139 mm Hg or diastolic between 80-89 mm Hg.
        - Hypertension Stage 2 : Systolic 140 mm Hg or higher or diastolic 90 mm Hg or higher.
        - Hypertensive Crisis :  Systolic higher than 180 mm Hg and/or diastolic higher than 120 mm Hg.""")
                    
            #SkinThickness = st.number_input(" Your Skinfold Thickness (mm): ")
            #with st.expander(" WHAT IS SKIN FOLD THICKNESS? "):
                #st.write(""" :orange[Skinfold thickness refers to the amount of fat under the skin.
                            #It's often measured at specific body sites to estimate body fat.]\n
                            
                                                        #\t__NOTE__\t
                           # \n
        #:red[It is advisable to go to medical professional who will measure it using a skinfold caliper]
                #\n
                            
        # :blue[To measure at home, gently pinch skin between thumb and forefinger, then measure the pinched area's width with a ruler or caliper.
        # Keep in mind, professional assessments are more accurate]""")
                
            Insulin = st.number_input("Your Insulin Level after 2 Hours (mu U/ml): ")
            with st.expander("TAP TO KNOW MORE ABOUT INSULIN LEVEL AFTER 2 HOURS"):
                st.write(""" :green[_Insulin Level after 2 Hours" refers to the measurement of insulin concentration
                            in the blood two hours after a particular event,such as consuming a meal or undergoing a test.
                            This measurement helps assess the body's response to glucose metabolism over time and
                            can provide insights into insulin sensitivity and diabetes risk._]
                            """)

                st.write("""    *HOW IT IS DIFFERENT FROM POST MEAL BLOOD SUGAR LEVEL*? """)
                st.write("""  :blue[_Insulin Level after 2 Hours" checks how insulin reacts to sugar in the blood after 2 hours.
            "Post-Meal Blood Sugar Level" measures sugar concentration in the blood after eating.
            Both help evaluate glucose control and diabetes risk_]""")  
                st.write(":red[_for simplicity you can add your post meal blood sugar level_]")
                
            weight = st.number_input("Enter your weight in kgs: ",2,200)
                
            height = st.number_input("Enter your height in centimetres: ",20,300)
            Height = height/100
            BMI = (weight)/(Height*Height)
            with st.expander(" CHECK YOUR BMI "):
                st.write("YOUR BMI IS",BMI)

            diabetes_history = st.radio("DOES YOUR FAMILY HAVE A HISTORY OF DIABETES ?",["YES","NO"])
            if diabetes_history == "YES":
                DiabetesPedigreeFunction = random.uniform(0.5,0.9)
            elif diabetes_history == "NO":
                DiabetesPedigreeFunction = 0
                                                                                         
                
            #DiabetesPedigreeFunction = st.slider("Your Family Diabetes History Score:  ",0.0,1.0,0.001)
            #with st.expander(" KNOW MORE "):
           #     st.write(""":orange[_Think of the this as a score that shows how much your family's history of diabetes might affect your own risk.
        #If the score is higher, it means there's a stronger chance that diabetes runs in your family, which could increase your own risk.
        #It's a way to see how family history might play a role in your health_]""")
                
            Age = st.number_input("ENTER YOUR AGE: ",0,101)

            SkinThickness = 10
            if a == "MALE":  # Male
                if Age >= 1 and Age <= 9:
                    SkinThickness = random.uniform(5.0,12.1)
                elif Age >= 10 and Age <= 13:
                    SkinThickness = random.uniform(6.0,15.1)
                elif Age >= 14 and Age <= 17:
                    SkinThickness = random.uniform(8.0,18.1)
                elif Age >= 18 and Age <= 29:
                    SkinThickness = random.uniform(5.5, 10.6)
                elif Age >= 30 and Age <= 59:
                    SkinThickness = random.uniform(7.0, 12.6)
                elif Age >= 60:
                    SkinThickness = random.uniform(9.0, 14.2)


            
            elif a == "FEMALE":  # Female
                if Age >= 1 and Age <= 9:
                    SkinThickness = random.uniform(5.0,13.1)
                elif Age >= 10 and Age <= 13:
                    SkinThickness = random.uniform(7.0,16.1)
                elif Age >= 14 and Age <= 17:
                    SkinThickness = random.uniform(9.0,20.1)
                elif Age >= 18 and Age <= 29:
                    SkinThickness = random.uniform(11.0, 18.6)
                elif Age >= 30 and Age <= 59:
                    SkinThickness = random.uniform(12.5, 21.2)
                elif Age >= 60:
                    SkinThickness = random.uniform(14.5, 23.3)
                

                

            #code for prediction
            #diagnosis = " "

            #prediction button
            if st.button('DIABETES DATA Test Result'):
                diagnosis = predict_diabetes([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
            with st.spinner(" WAIT FOR A WHILE "):
                t.sleep(3)


        #st.success(diagnosis)
        #calculate your A1C percentage
    tab1,tab2 = st.tabs(["A1C % CALCULATOR","AVERAGE BLOOD GLUCOSE LEVEL"])
    with tab1:
        st.header(" :blue[KNOW YOUR A1C PERCENTAGE] ")
        b = st.radio(":green[TAKE YOUR AVERAGE BLOOD SUGAR LEVEL SAME AS ABOVE GIVEN FASTING BLOOD SUGAR LEVEL ?]",["NO","YES"])
        if b == "NO":
            average_blood_sugar = st.number_input("Enter your average blood sugar (mg/dL) for past 2-3 months: ")
        elif b == "YES":
            average_blood_sugar = Glucose
            
        a1c_percentage = (average_blood_sugar + 46.7) / 28.7
        number = a1c_percentage
        decimal_places = 3
        if average_blood_sugar == 0:
            formatted_number = 0
        elif average_blood_sugar != 0.00:           
            formatted_number = format(number, f".{decimal_places}f")
        with st.spinner(" WAIT "):
            t.sleep(1)
            st.write(" :orange[YOUR A1C percentage is: ] ",formatted_number)#getting only specified number of decimal places
            if st.button(" CHECK IF YOU HAVE DIABETES OR NOT "):
                    
                if a1c_percentage < 5.7:
                    st.success("*NORMAL!YOU DON'T HAVE DIABETES*")
                elif a1c_percentage < 6.4 and a1c_percentage > 5.7:
                    st.write("*YOU MAY BE PREDIABETEIC*")
                elif a1c_percentage > 6.5:
                    st.warning(":red[*YOU MAY BE DIABETIC*]")
    with tab2:
        st.header(" :violet[KNOW YOUR AVERAGE BLOOD GLUCOSE USING A1C PERCENTAGE] ")
        a1c_percentage = st.number_input("Enter your A1C percentage: ")
        average_blood_sugar = (a1c_percentage * 28.7) - 46.7
        number = average_blood_sugar
        decimal_places = 3
        if a1c_percentage == 0.00:
            formatted_number = 0
        elif a1c_percentage != 0.00:
            formatted_number = format(number, f".{decimal_places}f")
        with st.spinner(" WAIT "):
            t.sleep(1)
            st.write(":green[YOUR YOUR AVERAGE BLOOD GLUCOSE LEVEL IN mg/dl IS:  ] ",formatted_number)

        #pass
        #error message
        #st.error(" ONE SHOULD NOT PLAY WITH THIS WEBSITE ")

        #SUCCESS MESSAGE
    st.file_uploader("UPLOAD SUPPORTING MEDICAL DATA FOR YOUR ABOVE GIVEN DETAILS")
        




    with t2:

        df = pd.read_csv("diabetes.csv")
        features = df.columns[:-1]
        st.header(" :green[_DATA VISUALIZATIONS SECTION_]")
        with st.expander(" ## :orange[*__CHECK THE DATASET HERE__*]"):
            #df = pd.read_csv("diabetes.csv")
            st.dataframe(df)
            st.write("[learn more about this dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)")

        ch = st.radio(":violet[SELECT YOUR VISUALIZATION PLOT]",["DISTRIBUTION PLOTS","HEATMAP","SCATTER PLOTS","LINE GRAPHS","BAR GRAPHS","PIE CHARTS","HISTOGRAMS"])
        
        #data visualizations corner

        #correlation heatmap
        if ch == "HEATMAP":
        
            st.subheader(":blue[Correlation Heatmap]")
            corr_matrix = df.corr()
            fig_heatmap = px.imshow(corr_matrix)
            st.plotly_chart(fig_heatmap)

        
        
        #distribution plots
        elif ch == "DISTRIBUTION PLOTS":

            st.subheader(":violet[DISTRIBUTION PLOTS]")
            features = df.columns[:-1]
            for feature in features:
                fig_dist = px.histogram(df, x=feature, color='Outcome', title=f'Distribution of {feature}')
                st.plotly_chart(fig_dist)



        #scatter plots
        elif ch == "SCATTER PLOTS":
            st.subheader("Scatter Plots")
            
            for feature_x in features:
                for feature_y in features:
                    if feature_x != feature_y:
                        fig_scatter = px.scatter(df, x=feature_x, y=feature_y, color='Outcome', 
                                     title=f'Scatter Plot of {feature_x} vs {feature_y}')
                        st.plotly_chart(fig_scatter)

        elif ch == "LINE GRAPHS":
            st.subheader("Line Graphs")
    
            features = df.columns[:-1]  # Exclude the 'Outcome' column
    
            selected_feature = st.selectbox("Select a feature for the x-axis", features)
    
            for feature_y in features:
                if feature_y != selected_feature:
                    fig_line = px.line(df, x=selected_feature, y=feature_y, color='Outcome', 
                               title=f'Line Graph: {feature_y} vs {selected_feature}')
                    st.plotly_chart(fig_line)


        #bar graphs
        elif ch == "BAR GRAPHS":
            st.subheader("Bar Graphs")
    
            features = df.columns[:-1]  # Excluding the 'Outcome' column
    
            selected_feature = st.selectbox("Select a feature for the x-axis", features)
    
            for feature_y in features:
                if feature_y != selected_feature:
                    fig_bar = px.bar(df, x=selected_feature, y=feature_y, color='Outcome', 
                             title=f'Bar Graph: {feature_y} vs {selected_feature}')
                    st.plotly_chart(fig_bar)


        # adding pie charts
        elif ch == "PIE CHARTS":
            st.subheader("Pie Charts")
    
            for feature in features:
                if feature != 'Outcome':
                    fig_pie = px.pie(df, names='Outcome', title=f'Pie Chart: {feature}')
                    st.plotly_chart(fig_pie)



        #adding histograms
        elif ch == "HISTOGRAMS":
            st.subheader("Histograms")
    
            for feature in features:
                if feature != 'Outcome':
                    fig_hist = px.histogram(df, x=feature, color='Outcome', 
                                    title=f'Histogram: {feature}')
                    st.plotly_chart(fig_hist)

    with t3:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn import svm
        from sklearn.preprocessing import StandardScaler
        import warnings
        import plotly.graph_objs as go

        # Ignore scikit-learn warnings
        warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")
        warnings.filterwarnings("ignore", message="X does not have valid feature names, but SVC was fitted with feature names")

        # Load your dataset and preprocess it
        # For demonstration purposes, let's assume you have a CSV file named "diabetes.csv" with appropriate columns

        # Load the CSV file
        d_data = pd.read_csv("diabetes.csv")

        # Grouping the data
        d_data['Outcome'].value_counts()

        # Separating the useful labels from the original data
        X = d_data.drop(columns="Outcome", axis=1)
        Y = d_data["Outcome"]

        # Splitting train and test data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, stratify=Y, random_state=2)

        # Standardizing the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Implementing machine learning algorithm
        classifier = svm.SVC(kernel="linear")
        # Training the SVM model
        classifier.fit(X_train_scaled, Y_train)

        # Calculating the accuracy of the model
        X_train_prediction = classifier.predict(X_train_scaled)
        training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

        print("Accuracy on train data prediction is:", training_data_accuracy)

        X_test_prediction = classifier.predict(X_test_scaled)
        test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

        print("Accuracy on test data prediction is:", test_data_accuracy)

        # Getting the coefficients and intercept of the hyperplane
        coef = classifier.coef_[0]
        intercept = classifier.intercept_[0]

        # Creating a mesh grid for visualization
        xx, yy = np.meshgrid(np.linspace(X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1, 100),
                             np.linspace(X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1, 100))

        zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]

        # Creating a 3D surface plot using Plotly
        surface_trace = go.Surface(x=xx, y=yy, z=zz, colorscale='Viridis', showscale=False)

        # Create scatter plots for the two classes (optional)
        class_0_trace = go.Scatter3d(x=X_train_scaled[Y_train == 0, 0],
                                     y=X_train_scaled[Y_train == 0, 1],
                                     z=X_train_scaled[Y_train == 0, 2],
                                     mode='markers', name='Class 0')

        class_1_trace = go.Scatter3d(x=X_train_scaled[Y_train == 1, 0],
                                     y=X_train_scaled[Y_train == 1, 1],
                                     z=X_train_scaled[Y_train == 1, 2],
                                     mode='markers', name='Class 1')

        layout = go.Layout(title='SVM Hyperplane',
                           scene=dict(xaxis_title='Feature 1',
                                      yaxis_title='Feature 2',
                                      zaxis_title='Feature 3'))
        fig = go.Figure(data=[surface_trace, class_0_trace, class_1_trace], layout=layout)
        st.title("Diabetes Prediction with SVM Hyperplane Visualization")
        st.write("SVM hyperplane visualization for diabetes prediction.")
        st.plotly_chart(fig)







        
    
    #about the developer
    with st.expander("CONTACT THE DEVELOPER"):
        st.write(" [LEARN MORE ABOUT THE DEVELOPER >](https://sourabh-dey-resume.streamlit.app/)")
        st.image("profile-pic (5).png",width = 250)
        contact = """ <form action="https://formsubmit.co/uhddey@gmail.com" method="POST">
                 <input type="text" name="name" placeholder = "YOUR NAME" required>
                 <input type="email" name="email" placeholder = "YOUR EMAIL" required>
                 <textarea name="message" placeholder="Tell us your problem" required></textarea>
                 <button type="submit">Send</button>
            </form>  """

        
        st.markdown(contact, unsafe_allow_html=True)
     







if __name__ == "__main__":
    main()




    

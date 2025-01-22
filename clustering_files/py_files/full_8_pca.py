# Train/Valid/Test = 60/20/20
# PCA = True
# GMM = 'full', 8 components, Continuous+Discrete+PCA - Continuous[PCA], Scaling:True
import os
import sys
path = "../statistical-models-for-improvide-insurance-risk-assessment-using-telematics"
sys.path.insert(0, path)
from gmmglm import gmmglm
import pandas as pd
import statsmodels.api as sm
import joblib
# Set parameters
file_name = 'full_8_pca'  #### Edit
scaling = True
n_components = 8
covariance_type = "full"


# Load Data
df = pd.read_csv(path+"/data/telematics_clean.csv")
# Model
mod = gmmglm(df=df, 
             target_variable=["Claim"],
             discrete_variables=[
                 "Duration", "Insured_age", "Car_age", "Credit_score",
                 "Years_noclaims", "Accel_06miles", "Accel_08miles",
                 "Accel_09miles", "Accel_11miles", "Accel_12miles", "Accel_14miles",
                 "Brake_06miles", "Brake_08miles", "Brake_09miles", "Brake_11miles",
                 "Brake_12miles", "Brake_14miles", "Left_turn_intensity08",
                 "Left_turn_intensity09", "Left_turn_intensity10",
                 "Left_turn_intensity11", "Left_turn_intensity12",
                 "Right_turn_intensity08", "Right_turn_intensity09",
                 "Right_turn_intensity10", "Right_turn_intensity11",
                 "Right_turn_intensity12", 
                 "Total_days_driven"], 
                 continuous_variables=[
                     "Annual_miles_drive", "Annual_pct_driven", "Total_miles_driven",
                     "Pct_drive_mon", "Pct_drive_tue", "Pct_drive_wed", "Pct_drive_thr",
                     "Pct_drive_fri", "Pct_drive_sat", "Pct_drive_sun",
                     "Pct_drive_2hrs", "Pct_drive_3hrs", "Pct_drive_4hrs",
                     "Pct_drive_wkday", "Pct_drive_wkend", "Pct_drive_rush_am",
                     "Pct_drive_rush_pm", "Pct_drive_rush", "Avgdays_week"], 
                 nominal_variables=[
                     "Territory_nominal", "Insured_sex_nominal", "Marital_nominal",
                     "Car_use_nominal", "Region_nominal"], 
             drop_variables=[
                 "NB_Claim", "AMT_Claim", "Pct_drive_wkend", "Pct_drive_mon",
                 "Pct_drive_tue", "Pct_drive_wed", "Pct_drive_thr",
                 "Pct_drive_fri", "Pct_drive_sat", "Pct_drive_sun",
                 "Pct_drive_rush_am", "Pct_drive_rush_pm", "Annual_pct_driven",
                 "Annual_miles_drive", "Avgdays_week", "Territory_nominal"],
             output_to_file = True,
             file_path = path+'/clustering_files/clustering_results/txt/'+file_name+'.txt', 
             random_state=16430792)
mod.train_test_split(valid_size=0.25, test_size=0.2, drop=False, random_state=None)
mod.calculate_mean_std() # Calculate mean and standard deviation
mod.fit_pca(p=3, 
            pca_vars=[
                "Pct_drive_2hrs", "Pct_drive_3hrs", "Pct_drive_4hrs","Accel_06miles", "Accel_08miles", 
                "Accel_09miles","Accel_11miles", "Accel_12miles", "Accel_14miles","Brake_06miles", 
                "Brake_08miles", "Brake_09miles","Brake_11miles", "Brake_12miles", "Brake_14miles",
                "Left_turn_intensity08", "Left_turn_intensity09","Left_turn_intensity10", 
                "Left_turn_intensity11","Left_turn_intensity12", "Right_turn_intensity08",
                "Right_turn_intensity09", "Right_turn_intensity10","Right_turn_intensity11",
                "Right_turn_intensity12"])
mod.calculate_mean_std() # Calculate mean and standard deviation after PCA
mod.fit_gmm(n_components=n_components, 
            covariance_type=covariance_type, 
            cluster_variables=list(set(mod.continuous_variables + mod.discrete_variables + mod.pca_variables)-set(mod.pca_vars)),
            gmm_scaling = scaling,
            gmm_kwargs={'tol':0.001, 'reg_covar':1e-06, 'max_iter':1000, 'n_init':10, 'init_params':'kmeans', 
                        'weights_init':None, 'means_init':None, 'precisions_init':None, 
                        'warm_start':False, 'verbose':1, 'verbose_interval':10}) 
# Pickle files
file = open(path+'/clustering_files/clustering_results/files/'+file_name, 'wb') 
joblib.dump(mod, file)
file.close()
# This one accounts for number of trials
# Train/Valid/Test = 60/20/20
# PCA = True
# GMM = 'full', 3 components, Continuous+Discrete, Scaling:True
# GLM = Binomial(Loglog), Problem:Classification, Metric:MAVG, Scaling:True, Poly:3, OptCutoff:True, Offset:False, Exposure:False, FreqWeights:False, VarWeights:False, Intercept:True, Sampling:False, Weighted:False
# Load packages
import os
import sys
path = "../statistical-models-for-improvide-insurance-risk-assessment-using-telematics"
sys.path.insert(0, path)
from gmmglm import gmmglm
import pandas as pd
import statsmodels.api as sm
import joblib 
# Set parameters
file_name = 'best_model'  #### Edit
scaling = True
glm_family=sm.families.Binomial(link=sm.families.links.LogLog())  #### Edit
problem_type='classification'
use_opt_cutoff=True
use_sampling=False
sample_size=0 
offset=None
exposure=None
freq_weights=None
var_weights=None
glm_intercept=True
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
             file_path = path+'/regression_files/regression_results/'+file_name+'.txt',
             random_state=16430792)
mod.train_test_split(valid_size=0.25, test_size=0.2, drop=False, random_state=None)
mod.calculate_mean_std() # Calculate mean and standard deviation
mod.fit_gmm(n_components=3, 
            covariance_type="full", 
            cluster_variables=mod.continuous_variables + mod.discrete_variables,
            gmm_scaling = scaling,
            gmm_kwargs={'tol':0.001, 'reg_covar':1e-06, 'max_iter':1000, 'n_init':10, 'init_params':'kmeans', 
                        'weights_init':None, 'means_init':None, 'precisions_init':None, 
                        'warm_start':False, 'verbose':1, 'verbose_interval':10}) 
mod.stepwise_glm(glm_family=glm_family,
                 problem_type=problem_type,
                 metric= 'mavg', 
                 poly_order=3, 
                 use_sampling=use_sampling, 
                 sample_size=sample_size, 
                 use_opt_cutoff=use_opt_cutoff,
                 glm_scaling=scaling,
                 offset=offset, 
                 exposure=exposure, 
                 freq_weights=freq_weights, 
                 var_weights=var_weights, 
                 missing='raise',
                 glm_intercept=glm_intercept,
                 glm_kwargs={},
                 fit_kwargs={'start_params':None, 'maxiter':10000, 'method':'IRLS', 'tol':1e-08, 'scale':None, 
                             'cov_type':'nonrobust', 'cov_kwds':None, 'use_t':None, 'full_output':True, 
                             'disp':False, 'max_start_irls':3}, 
                 metric_kwargs={})
mod.fit_glm(regression_variables=mod.stepwise_regression_variables, 
            glm_family=glm_family,
            problem_type=problem_type,
            use_opt_cutoff=use_opt_cutoff,
            use_sampling=use_sampling, 
            sample_size=sample_size, 
            weighted=False, 
            glm_scaling=scaling,
            offset=offset, 
            exposure=exposure, 
            freq_weights=freq_weights, 
            var_weights=var_weights, 
            missing='raise',
            glm_intercept=glm_intercept,
            glm_kwargs=None,
            fit_kwargs={'start_params':None, 'maxiter':10000, 'method':'IRLS', 'tol':1e-08, 'scale':None, 
                        'cov_type':'nonrobust', 'cov_kwds':None, 'use_t':None, 'full_output':True, 
                        'disp':False, 'max_start_irls':3})
# Pickle files
file = open(path+'/regression_files/regression_results/'+file_name, 'wb')
joblib.dump(mod, file)
file.close()
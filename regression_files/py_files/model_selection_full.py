# Train/Valid/Test = 60/20/20
# PCA = False
# GMM = 'full', 2-12 components, Continuous+Discrete, Scaling:True
# GLM = Binomial, Problem:Classification, Metric:MaVG, Scaling:True, Poly:3, OptCutoff:True, Offset:False, Exposure:False, FreqWeights:False, VarWeights:False, Intercept:True, Sampling:False, Weighted:False
# Load packages
import os
import sys
path = "../statistical-models-for-improvide-insurance-risk-assessment-using-telematics"
path_to_files = path+'/clustering_files/clustering_results/files/'
path_to_csv = path+'/regression_files/regression_results/csv/'
sys.path.insert(0, path)
from gmmglm import gmmglm
import pandas as pd
import statsmodels.api as sm
import joblib
os.makedirs(path_to_files, exist_ok=True)
os.makedirs(path_to_csv, exist_ok=True)
# Set parameters
scaling = True
glm_families={"logit":sm.families.Binomial(link=sm.families.links.Logit()), "probit":sm.families.Binomial(link=sm.families.links.Probit()), "cauchy":sm.families.Binomial(link=sm.families.links.Cauchy()), "loglog":sm.families.Binomial(link=sm.families.links.LogLog()), "cloglog":sm.families.Binomial(link=sm.families.links.CLogLog())} #### Edit
problem_type='classification'
use_opt_cutoff=True
use_sampling=False
sample_size=0 
offset=None
exposure=None
freq_weights=None 
var_weights=None
glm_intercept=True

# Iterate over gmm files and glm family
file_names = ['full_2', 'full_2_pca', 'full_3', 'full_3_pca', 'full_4', 'full_4_pca', 'full_5', 'full_5_pca', 'full_6', 'full_6_pca', 'full_7', 'full_7_pca', 'full_8', 'full_8_pca', 'full_9', 'full_9_pca', 'full_10', 'full_10_pca', 'full_11', 'full_11_pca', 'full_12', 'full_12_pca']
mavg = []

for file_name in file_names:
    file = open(path_to_files+file_name, 'rb')
    # dump information to that file
    mod = joblib.load(file)#pickle.load(file)
    # close the file
    file.close()
    for glm_name, glm_family in glm_families.items():
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
        mavg.append(mod.glm_metric(y_true=mod.df_test["Claim"].values, y_pred=mod.glm_test_pred_class, metric="mavg", problem_type="classification"))
    
df = pd.DataFrame({"model":[element for element in file_names for _ in range(len(list(glm_families.keys())))], "mavg":mavg, "glm_family":list(glm_families.keys())*len(file_names)})
df.to_csv(path_to_csv+"/model_selection_full.csv", index=False)    
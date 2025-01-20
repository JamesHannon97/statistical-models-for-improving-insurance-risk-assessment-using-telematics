# Packages
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats import stattools, descriptivestats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns 
from prettytable import PrettyTable
from scipy.stats import norm, gmean, chi2, shapiro, kstest, anderson, jarque_bera, normaltest, kurtosis
from scipy.spatial.distance import mahalanobis
import pingouin as pg
from tqdm.notebook import tqdm
from IPython.display import display
import sfrancia
import logging
import sys 

class gmmglm:
    """Gaussian Mixture Model and Generalized Linear Models.

    Parameters
    ----------
    df : Dataframe
    glm_family : family class instance from statsmodels.genmod.families
        Available familes with link function in brackets include: 
        - Binomial (Logit, Probit, Cauchy, Log, LogLog, CLogLog, Identity), 
        - Gamma(Inverse, Log, Identity), 
        - Gaussian(Identity, Log, Inverse), 
        - InverseGaussian(Inverse Squared, Inverse, Log, Identity), 
        - NegativeBinomial(Log, CLogLog, Identity, Nbinom, Power), 
        - Poisson(Log, Identity, Sqrt), 
        - Tweedie(Log, Power, any aliases of power)
        Default is Binomial(Logit).
    target_variable : str
        Name of target variable.
    discrete_variables : List
        List of discrete variables.
    continuous_variables : List
        List of continuous variables.
    nominal_variables : List 
        List of nominal variables.
    drop_variables : List
        List of variables to drop.
    offset : str or None
        Offset variable for regression.
    random_state : Int
        Seed for random number generation.
    
    Notes
    -----
    Use train_test_split() to produce train/valid/test sets
    Use calculate_mean_variance() to produce mu and sigma. This must be done after fit_pca() to ensure pca variables are included. 
    Use fit_pca() to produce pca variables
    Variables that are nominal need to have it in their name

    New Attributes
    --------------
    df
    glm_family
    offset
    target_variable
    continuous_variables
    discrete_variables
    nominal_variables
    drop_variables
    problem_type
    random_state
    train_set - boolean indicating if train set is created
    valid_set - boolean indicating if validation set is created
    test_set - boolean indicating if test set is created
    use_pca - boolean indicating if pca is used. Default is False. Only changed by calling fit_pca()
    """

    def __init__(self,
                 df, 
                 target_variable=["NB_Claim"], 
                 discrete_variables=["Duration", "Insured_age", "Car_age", "Credit_score",
                "Years_noclaims", "Accel_06miles", "Accel_08miles",
                "Accel_09miles", "Accel_11miles", "Accel_12miles", "Accel_14miles",
                "Brake_06miles", "Brake_08miles", "Brake_09miles", "Brake_11miles",
                "Brake_12miles", "Brake_14miles", "Left_turn_intensity08",
                "Left_turn_intensity09", "Left_turn_intensity10",
                "Left_turn_intensity11", "Left_turn_intensity12",
                "Right_turn_intensity08", "Right_turn_intensity09",
                "Right_turn_intensity10", "Right_turn_intensity11",
                "Right_turn_intensity12", 
                "Total_days_driven"], ###### EDIT 07/03/24: Removed Avgdays_week_discrete from discrete_variables
                 continuous_variables=["Annual_miles_drive", "Annual_pct_driven", "Total_miles_driven",
                "Pct_drive_mon", "Pct_drive_tue", "Pct_drive_wed", "Pct_drive_thr",
                "Pct_drive_fri", "Pct_drive_sat", "Pct_drive_sun",
                "Pct_drive_2hrs", "Pct_drive_3hrs", "Pct_drive_4hrs",
                "Pct_drive_wkday", "Pct_drive_wkend", "Pct_drive_rush_am",
                "Pct_drive_rush_pm", "Pct_drive_rush", "Avgdays_week"], 
                 nominal_variables=["Territory_nominal", "Insured_sex_nominal", "Marital_nominal",
                "Car_use_nominal", "Region_nominal"], 
                 drop_variables=["AMT_Claim", "Pct_drive_wkend", "Pct_drive_mon",
                                 "Pct_drive_tue", "Pct_drive_wed", "Pct_drive_thr",
                                 "Pct_drive_fri", "Pct_drive_sat", "Pct_drive_sun",
                                 "Pct_drive_rush_am", "Pct_drive_rush_pm", "Annual_pct_driven",
                                 "Annual_miles_drive", "Avgdays_week", "Territory_nominal"], 
                output_to_file=False,
                file_path=None,
                level=logging.INFO,
                random_state=16430792):
        # Store Attributes
        self.version = "1.0.0"
        if isinstance(target_variable, list) == False:
            print("Target variable should be a list.")
            sys.exit()
        else:
            self.trials=False
            if len(target_variable) > 1:
                df["trials"] = df[target_variable].sum(axis=1)
                self.trials = True
        self.df = df.copy()
        self.target_variable = target_variable # should be default of glm
        self.continuous_variables = list(set(continuous_variables) - set(drop_variables))
        self.discrete_variables = list(set(discrete_variables) - set(drop_variables))
        self.df[nominal_variables] = df[nominal_variables].astype('category') # Change dtype from int64 to category
        for var in nominal_variables:
            if "_nominal" not in var:
                self.df.rename(columns={var: var + "_nominal"}, inplace=True)
        # important drop comes before nominal
        drop_variables = [var + "_nominal" if "_nominal" not in var and var in nominal_variables else var for var in drop_variables]
        nominal_variables = [var + "_nominal" if "_nominal" not in var else var for var in nominal_variables]
        self.nominal_variables = list(set(nominal_variables) - set(drop_variables))
        if self.trials:
            drop_variables += ["trials"]
        self.drop_variables = drop_variables # Drop variables
        self.random_state = random_state # Store random state
        self.train_set = False # Train set not created
        self.valid_set = False # Validation set not created
        self.test_set = False # Test set not created
        self.use_pca = False # False by default, until fit_pca() is called
        self.output_to_file = output_to_file
        self.file_path = file_path
        if self.output_to_file:
            logging.basicConfig(filename=file_path, 
                                filemode='w', 
                                level=level, 
                                format='%(asctime)s:%(levelname)s:%(message)s')
        self.print_both("Model Initialised.", log_level="info")

    # Define a custom print function
    def print_both(self, *args, log_level="info", **kwargs):
        # Print to stdout
        print(*args, **kwargs)
        if self.output_to_file:
            if log_level == "info":
                logging.info(*args, **kwargs)
            elif log_level == "error":
                logging.error(*args, **kwargs)
            elif log_level == "warning":
                logging.warning(*args, **kwargs)

    def train_test_split(self, valid_size, test_size, drop=False, random_state=None):
        """This produces the training, validation and test sets.

        Parameters
        ----------
        valid_size : float
            Proportion of validation set
        test_size : float
            Proportion of test set
        drop : bool
            Drop index when reseting
        random_state : int
            Seed for random number generation. Default is None and uses self.random_state
    
        New Attributes
        --------------
        df_train - Training set
        df_valid - Validation set
        df_test - Test set

        Modified Attributes
        -------------------
        train_set - Boolean indicating if train set is created
        valid_set - Boolean indicating if validation set is created
        test_set - Boolean indicating if test set is created
        """
        if random_state is None:
            random_state = self.random_state
        if valid_size == 0 and test_size == 0:
            self.df_train = self.df.copy()
            self.print_both("Train set created.", log_level="info")
            self.print_both("Train set shape: {}".format(self.df_train.shape), log_level="info")
        elif valid_size == 0 and test_size != 0:
            df_train, df_test = train_test_split(self.df, test_size=test_size, random_state=random_state)
            df_test.sort_index(inplace=True) # Sort index
            df_test.reset_index(inplace=True, drop=drop) # Reset index
            df_train.sort_index(inplace=True) # Sort index
            df_train.reset_index(inplace=True, drop=drop) # Reset index
            self.df_test = df_test # Store as attribute
            self.df_train = df_train # Store as attribute
            self.train_set = True # Train set created
            self.test_set = True # Test set created
            self.print_both("Train and Test sets created.", log_level="info")
            self.print_both("Train set shape: {}".format(self.df_train.shape), log_level="info")
            self.print_both("Test set shape: {}".format(self.df_test.shape), log_level="info")
        elif valid_size != 0 and test_size == 0:
            df_train, df_valid = train_test_split(self.df, test_size=valid_size, random_state=random_state)
            df_valid.sort_index(inplace=True) # Sort index
            df_valid.reset_index(inplace=True, drop=drop) # Reset index
            df_train.sort_index(inplace=True) # Sort index
            df_train.reset_index(inplace=True, drop=drop) # Reset index
            self.df_valid = df_valid # Store as attribute
            self.df_train = df_train # Store as attribute
            self.train_set = True # Train set created
            self.valid_set = True # Valid set created
            self.print_both("Train and Validation sets created.", log_level="info")
            self.print_both("Train set shape: {}".format(self.df_train.shape), log_level="info")
            self.print_both("Validation set shape: {}".format(self.df_valid.shape), log_level="info")
        else:
            df_train_valid, df_test = train_test_split(self.df, test_size=test_size, random_state=random_state)
            df_train, df_valid = train_test_split(df_train_valid,
                                              test_size=valid_size,
                                              random_state=random_state)
            df_test.sort_index(inplace=True) # Sort index
            df_test.reset_index(inplace=True, drop=drop) # Reset index
            df_valid.sort_index(inplace=True) # Sort index
            df_valid.reset_index(inplace=True, drop=drop) # Reset index
            df_train.sort_index(inplace=True) # Sort index
            df_train.reset_index(inplace=True, drop=drop) # Reset index
            self.df_test = df_test # Store as attribute
            self.df_valid = df_valid # Store as attribute
            self.df_train = df_train # Store as attribute
            self.train_set = True # Train set created
            self.valid_set = True # Valid set created
            self.test_set = True # Test set created
            self.print_both("Train, Validation and Test sets created.", log_level="info")
            self.print_both("Train set shape: {}".format(self.df_train.shape), log_level="info")
            self.print_both("Validation set shape: {}".format(self.df_valid.shape), log_level="info")
            self.print_both("Test set shape: {}".format(self.df_test.shape), log_level="info")
        
    def describe(self, df=None, kwargs={}):
        """This describes the dataframe.

        Parameters
        ----------
        df : Dataframe
            Default is None and uses self.df

        Returns
        -------
        Dataframe

        Notes
        -----
        This returns the description of the dataframe.
        """
        if df is None:
            if self.valid_set & self.test_set:
                return descriptivestats.describe(self.df, **kwargs), descriptivestats.describe(self.df_train, **kwargs), descriptivestats.describe(self.df_valid, **kwargs), descriptivestats.describe(self.df_test, **kwargs)
            elif self.valid_set:
                return descriptivestats.describe(self.df, **kwargs), descriptivestats.describe(self.df_train, **kwargs), descriptivestats.describe(self.df_valid, **kwargs)
            elif self.test_set:
                return descriptivestats.describe(self.df, **kwargs), descriptivestats.describe(self.df_train, **kwargs), descriptivestats.describe(self.df_test, **kwargs)
            else:
                return descriptivestats.describe(self.df, **kwargs)
        else:
            return descriptivestats.describe(df, **kwargs)
        
    def exploratory_data_analysis(self, **kwargs):
        """This produces exploratory data analysis.

        Parameters
        ----------
        df : Dataframe
            Default is None and uses self.df

        Returns
        -------
        Dataframe

        Notes
        -----
        This returns the exploratory data analysis of the dataframe.
        """
        for var in self.continuous_variables:
            print("Continuous Variables:")
            fig, ax = plt.subplots(figsize=(8, 8))
            sns.histplot(ax=ax, data=self.df, x=var, kde=True, **kwargs)
            ax.set_xlabel(var)
            ax.set_ylabel("Count")
            fig.tight_layout()
            plt.show()     
        for var in self.discrete_variables:
            print("Discrete Variables:")
            fig, ax = plt.subplots(figsize=(8, 8))
            sns.histplot(ax=ax, data=self.df, x=var, kde=True, **kwargs)
            ax.set_xlabel(var)
            ax.set_ylabel("Count")
            fig.tight_layout()
            plt.show()            
        for var in self.nominal_variables:
            print("Nominal Variables:")
            fig, ax = plt.subplots(figsize=(6, 6))
            self.df[var].value_counts().plot(ax = ax, kind = 'bar', xlabel=var, ylabel = 'Frequency', rot=0, edgecolor='black')
            fig.tight_layout()
            plt.show()

    def correlation(self, list_of_vars=None, kwargs={}):
        if list_of_vars is None:
            display(self.df.select_dtypes(exclude=object).corr().style.background_gradient(**kwargs))
            return self.df.select_dtypes(exclude=object).corr().style.background_gradient(**kwargs)
        else:    
            display(self.df[list_of_vars].select_dtypes(exclude=object).corr().style.background_gradient(**kwargs))
            return self.df[list_of_vars].select_dtypes(exclude=object).corr().style.background_gradient(**kwargs)


    def pairs_plot(self, list_of_vars, df=None, kwargs={}):
        if df is None:
            sns.pairplot(data=self.df, vars=list_of_vars, **kwargs)
            plt.show()
        else:
            sns.pairplot(data=df, vars=list_of_vars, **kwargs)
            plt.show()

    def qqplot(self, list_of_vars=None, by_cluster=False, kwargs={}):
        df = self.df_train.copy()
        df = self.scale_data(df) # just use continuous and discrete variables
        for var in list_of_vars:
            sm.qqplot(df[var], line ='45', **kwargs)
            plt.title(f'QQ Plot for {var}')  # Set the title for each variable's QQ plot
            plt.show()
        if by_cluster:
            for i in range(self.n_components):
                print("Cluster ", int(i))
                clust_i_df = df[self.clust_train_preds == i].copy()
                for var in list_of_vars:
                    sm.qqplot(clust_i_df[var], line ='45', **kwargs)
                    plt.title(f'QQ Plot for {var}')  # Set the title for each variable's QQ plot
                    plt.show()

    def tests_for_normality(self, gmm_scaling, df=None, list_of_vars=None, by_cluster=False, alpha=0.05, sample_size=2000, multivariate=True):
        """Performs Univariate ("Shapiro-Wilks", "Kolmogorov-Smirnov", "Jarque-Bera", "Omnibus", "Anderson-Darling") and Multivariate tests () for normality.

        Parameters
        ----------
        list_of_vars : List
            List of variables to test
        alpha : float
            Significance level. Default is 0.05.

        Notes
        -----
        Anderson-Darling test is hard coded for alpha=0.05.
        """
        if df is None:
            df_train = self.df_train.copy()
        else:
            df_train = df.copy()
        if gmm_scaling:
            df_train = self.scale_data(df_train)
        if list_of_vars is None:
            list_of_vars = self.continuous_variables + self.discrete_variables
        for var in list_of_vars:
            print("\n",var)
            res = pd.DataFrame(columns=["Test", "Statistic", "P-Value", "Conclusion"])
            test = ["Shapiro-Wilks", "Kolmogorov-Smirnov", "Jarque-Bera", "Omnibus", "Anderson-Darling"]
            test_statistic = [shapiro(df_train[var])[0], kstest(df_train[var], norm.cdf)[0], jarque_bera(x=df_train[var], axis=None, nan_policy='propagate', keepdims=False)[0], normaltest(a=df_train[var], axis=0, nan_policy='propagate')[0], anderson(df_train[var], dist='norm')[0]]
            p_value = [shapiro(df_train[var])[1], kstest(df_train[var], norm.cdf)[1], jarque_bera(x=df_train[var], axis=None, nan_policy='propagate', keepdims=False)[1], normaltest(a=df_train[var], axis=0, nan_policy='propagate')[1], anderson(df_train[var], dist='norm')[1][4]]
            conclusion = ["Reject Null" if p < alpha else "Fail to Reject Null" for p in p_value[:-1]]
            conclusion.append("Reject Null" if test_statistic[-1] > p_value[-1] else "Fail to Reject Null")
            res["Test"] = test
            res["Statistic"] = test_statistic
            res["P-Value"] = p_value
            res["Conclusion"] = conclusion
            display(res)
        if multivariate:
            # Add multivariate tests using pingouin
            res = pd.DataFrame(columns=["Test", "Statistic", "P-Value", "Conclusion"])
            test = ["Henze-Zirkler", "Royston", "Mardia Skew", "Mardia Kurtosis", "Mardia"]
            test_statistic = []
            p_value = []
            if df_train.shape[0]>sample_size:
                print(f"Data is too large for multivariate tests. Performing on a sample of {sample_size}.")
                sample_index = np.random.choice(df_train.index.values, size=sample_size, replace=False, p=None)
                df_sample = df_train.loc[sample_index].copy() 
                test_statistic.append(pg.multivariate_normality(df_sample[list_of_vars], alpha=alpha)[0])
                p_value.append(pg.multivariate_normality(df_sample[list_of_vars], alpha=alpha)[1])
                test_statistic.append(self.Royston_H_test(df=df_sample[list_of_vars], scale=False)[0])
                p_value.append(self.Royston_H_test(df=df_sample[list_of_vars], scale=False)[1])
                test_statistic.append(self.mardia_test(df=df_sample[list_of_vars], scale=False)[0])
                p_value.append(self.mardia_test(df=df_sample[list_of_vars], scale=False)[1])
                test_statistic.append(self.mardia_test(df=df_sample[list_of_vars], scale=False)[2])
                p_value.append(self.mardia_test(df=df_sample[list_of_vars], scale=False)[3])
                test_statistic.append(np.nan)
                p_value.append(np.nan)
                self.Royston_QQ_plot(df_sample[list_of_vars])
            else:
                test_statistic.append(pg.multivariate_normality(df_train[list_of_vars], alpha=alpha)[0])
                p_value.append(pg.multivariate_normality(df_train[list_of_vars], alpha=alpha)[1])
                test_statistic.append(self.Royston_H_test(df=df_train[list_of_vars])[0])
                p_value.append(self.Royston_H_test(df=df_train[list_of_vars])[1])
                test_statistic.append(self.mardia_test(df=df_train[list_of_vars], scale=False)[0])
                p_value.append(self.mardia_test(df=df_train[list_of_vars], scale=False)[1])
                test_statistic.append(self.mardia_test(df=df_train[list_of_vars], scale=False)[2])
                p_value.append(self.mardia_test(df=df_train[list_of_vars], scale=False)[3])
                test_statistic.append(np.nan)
                p_value.append(np.nan)
                self.Royston_QQ_plot(df_train[list_of_vars])
            conclusion = ["Reject Null" if p < alpha else "Fail to Reject Null" for p in p_value[:-1]]
            conclusion += ["Fail to Reject Null" if conclusion[-1]==conclusion[-2]=="Fail to Reject Null" else "Reject Null"]
            res["Test"] = test
            res["Statistic"] = test_statistic
            res["P-Value"] = p_value
            res["Conclusion"] = conclusion
            display(res)
        if by_cluster:
            if df is not None:
                clust_df_preds = self.predict_gmm(df=df_train, gmm_scaling=False) # scaling already done if true
            for i in range(self.n_components):
                print("Cluster ", i)
                if df is None:
                    clust_i_df_train = df_train[self.clust_train_preds == i].copy()
                else:
                    clust_i_df_train = df_train[clust_df_preds == i].copy()
                for var in list_of_vars:
                    print("\n",var)
                    res = pd.DataFrame(columns=["Test", "Statistic", "P-Value", "Conclusion"])
                    test = ["Shapiro-Wilks", "Kolmogorov-Smirnov", "Jarque-Bera", "Omnibus", "Anderson-Darling"]
                    test_statistic = [shapiro(clust_i_df_train[var])[0], kstest(clust_i_df_train[var], norm.cdf)[0], jarque_bera(x=clust_i_df_train[var], axis=None, nan_policy='propagate', keepdims=False)[0], normaltest(a=clust_i_df_train[var], axis=0, nan_policy='propagate')[0], anderson(clust_i_df_train[var], dist='norm')[0]]
                    p_value = [shapiro(clust_i_df_train[var])[1], kstest(clust_i_df_train[var], norm.cdf)[1], jarque_bera(x=clust_i_df_train[var], axis=None, nan_policy='propagate', keepdims=False)[1], normaltest(a=clust_i_df_train[var], axis=0, nan_policy='propagate')[1], anderson(clust_i_df_train[var], dist='norm')[1][4]]
                    conclusion = ["Reject Null" if p < alpha else "Fail to Reject Null" for p in p_value[:-1]]
                    conclusion.append("Reject Null" if test_statistic[-1] > p_value[-1] else "Fail to Reject Null")
                    res["Test"] = test
                    res["Statistic"] = test_statistic
                    res["P-Value"] = p_value
                    res["Conclusion"] = conclusion
                    display(res)
                if multivariate:
                    res = pd.DataFrame(columns=["Test", "Statistic", "P-Value", "Conclusion"])
                    test = ["Henze-Zirkler", "Royston", "Mardia Skew", "Mardia Kurtosis", "Mardia"]
                    test_statistic = []
                    p_value = []
                    if clust_i_df_train.shape[0]>sample_size:
                        print(f"Data is too large for multivariate tests. Performing on a sample of {sample_size}.")
                        sample_index = np.random.choice(clust_i_df_train.index.values, size=sample_size, replace=False, p=None)
                        clust_i_df_sample = clust_i_df_train.loc[sample_index].copy() 
                        # test
                        test_statistic.append(pg.multivariate_normality(clust_i_df_sample[list_of_vars], alpha=alpha)[0])
                        p_value.append(pg.multivariate_normality(clust_i_df_sample[list_of_vars], alpha=alpha)[1])
                        test_statistic.append(self.Royston_H_test(df=clust_i_df_sample[list_of_vars])[0])
                        p_value.append(self.Royston_H_test(df=clust_i_df_sample[list_of_vars])[1])
                        test_statistic.append(self.mardia_test(df=clust_i_df_sample[list_of_vars], scale=False)[0])
                        p_value.append(self.mardia_test(df=clust_i_df_sample[list_of_vars], scale=False)[1])
                        test_statistic.append(self.mardia_test(df=clust_i_df_sample[list_of_vars], scale=False)[2])
                        p_value.append(self.mardia_test(df=clust_i_df_sample[list_of_vars], scale=False)[3])
                        test_statistic.append(np.nan)
                        p_value.append(np.nan)
                        self.Royston_QQ_plot(clust_i_df_sample[list_of_vars])
                    else:
                        test_statistic.append(pg.multivariate_normality(clust_i_df_train[list_of_vars], alpha=alpha)[0])
                        p_value.append(pg.multivariate_normality(clust_i_df_train[list_of_vars], alpha=alpha)[1])
                        test_statistic.append(self.Royston_H_test(df=clust_i_df_train[list_of_vars])[0])
                        p_value.append(self.Royston_H_test(df=clust_i_df_train[list_of_vars])[1])
                        test_statistic.append(self.mardia_test(df=clust_i_df_train[list_of_vars], scale=False)[0])
                        p_value.append(self.mardia_test(df=clust_i_df_train[list_of_vars], scale=False)[1])
                        test_statistic.append(self.mardia_test(df=clust_i_df_train[list_of_vars], scale=False)[2])
                        p_value.append(self.mardia_test(df=clust_i_df_train[list_of_vars], scale=False)[3])
                        test_statistic.append(np.nan)
                        p_value.append(np.nan)
                        self.Royston_QQ_plot(clust_i_df_train[list_of_vars])
                    conclusion = ["Reject Null" if p < alpha else "Fail to Reject Null" for p in p_value[:-1]]
                    conclusion += ["Fail to Reject Null" if conclusion[-1]==conclusion[-2]=="Fail to Reject Null" else "Reject Null"]
                    res["Test"] = test
                    res["Statistic"] = test_statistic
                    res["P-Value"] = p_value
                    res["Conclusion"] = conclusion
                    display(res)

    def calculate_mean_std(self):
        """This calculates the mean and standard deviation of the training set.
        
        New Attributes
        --------------
        mu : array-like
            Mean of the training set
        sigma : array-like
            Standard deviation of the training set
        """
        # Store as attributes
        if self.use_pca:
            self.mu = self.df_train[self.continuous_variables+self.discrete_variables+self.pca_variables].mean(axis=0)
            self.sigma = self.df_train[self.continuous_variables+self.discrete_variables+self.pca_variables].std(axis=0)
        else:
            self.mu = self.df_train[self.continuous_variables+self.discrete_variables].mean(axis=0)
            self.sigma = self.df_train[self.continuous_variables+self.discrete_variables].std(axis=0)

    def scale_data(self, df, list_of_vars=None):
        """ This scales the dataframe by the mean and standard deviation of the training set.

        Parameters
        ----------
        df : Dataframe

        Returns
        -------
        Dataframe

        Notes
        -----
        This returns the scaled dataframe.
        Can specify certain columns.
        """
        if list_of_vars is None:            
            return (df[self.continuous_variables + self.discrete_variables] - self.mu[self.continuous_variables + self.discrete_variables]) / self.sigma[self.continuous_variables + self.discrete_variables]
        else:
            return (df[list_of_vars] - self.mu[list_of_vars]) / self.sigma[list_of_vars]

    def unscale_data(self, df, list_of_vars=None):
        """This unscales the dataframe by the mean and standard deviation of the training set.

        Parameters
        ----------
        df : Dataframe

        Returns
        -------
        Dataframe
        """
        if list_of_vars is None:   
            return self.sigma * df[self.continuous_variables +self.discrete_variables] + self.mu
        else:
            return self.sigma[list_of_vars] * df[list_of_vars] + self.mu[list_of_vars]
        
    def fit_pca(self, 
                p, 
                pca_vars=["Pct_drive_2hrs", "Pct_drive_3hrs", "Pct_drive_4hrs",
                          "Accel_06miles", "Accel_08miles", "Accel_09miles",
                          "Accel_11miles", "Accel_12miles", "Accel_14miles",
                          "Brake_06miles", "Brake_08miles", "Brake_09miles",
                          "Brake_11miles", "Brake_12miles", "Brake_14miles",
                          "Left_turn_intensity08", "Left_turn_intensity09",
                          "Left_turn_intensity10", "Left_turn_intensity11",
                          "Left_turn_intensity12", "Right_turn_intensity08",
                          "Right_turn_intensity09", "Right_turn_intensity10",
                          "Right_turn_intensity11", "Right_turn_intensity12"],
                kwargs={}
               ):
        """ This fits the PCA model and adds the PCA variables to the training, validation and test sets.

        New Attributes:
        ----------------------
        pca_vars : List
            List of variables used in PCA
        pca : Instance of PCA from sklearn.decomposition
        pca_variables : List 
            List of PCA variables, i.e. PCA1, etc.
        loading_matrix : array-like
            Loading matrix
        
        Modifie Attributes:
        ----------------------------
        use_pca
        df_train
        df_valid
        df_test 
        """
        self.use_pca = True # Store attribute
        self.pca_vars = pca_vars # Store attribute
        self.p = p # Store attribute
        df_train = self.df_train.copy() # Load data
        df_train[self.continuous_variables + self.discrete_variables] = self.scale_data(df_train) # Scale data
        self.pca = PCA().fit(df_train[self.pca_vars], **kwargs) # Fit PCA, Store attribute
        pca_list = [] # Add PCA-transformed Vars
        for i in range(1, self.p + 1):
            pca_list += ['PCA' + str(i)]
        self.df_train[pca_list] = self.pca.transform(df_train[self.pca_vars])[:, :self.p] # Add pca variables to train set
        self.pca_variables = pca_list # Store as attribute
        loadings = self.pca.components_.T * np.sqrt(self.pca.explained_variance_) # Loading Matrix
        n = len(self.pca_vars)
        self.loading_matrix = pd.DataFrame(
            loadings,
            columns=['PC' + str(i) for i in np.arange(1, n + 1)],
            index=self.pca_vars)
        if self.valid_set:
            df_valid = self.df_valid.copy() # Load data
            df_valid[self.continuous_variables + self.discrete_variables] = self.scale_data(df_valid) # Scale data
            self.df_valid[self.pca_variables] = self.pca.transform(df_valid[self.pca_vars])[:, :self.p]
        if self.test_set:
            df_test = self.df_test.copy() # Load data
            df_test[self.continuous_variables + self.discrete_variables] = self.scale_data(df_test) # Scale data
            self.df_test[self.pca_variables] = self.pca.transform(df_test[self.pca_vars])[:, :self.p]

        self.print_both("PCA fitted.", log_level="info")

    def predict_pca(self, df, scaling=True, return_df=False):
        """Transforms variables to PCA.

        Parameters
        ----------
        df : Dataframe
        scaling : bool
            Should the data be scaled before PCA
        return_df : bool
            Should the entire dataframe be returned or just the PCA columns

        Returns
        -------
        Dataframe or Series


        Notes
        -----
        There is no need to run this on the validation or test set as we did this during the fit.
        """
        df_copy = df.copy()
        if scaling == True: # Scaling
            df_copy_scaled = self.scale_data(df=df_copy)
            df_copy[self.pca_variables] = self.pca.transform(
                df_copy_scaled[self.pca_vars])[:, :self.p]
        else:
            df_copy[self.pca_variables] = self.pca.transform(
                df_copy[self.pca_vars])[:, :self.p]
        if return_df:
            return df_copy
        else:
            return df_copy[self.pca_variables]

    def optimal_cutoff_point(self, y, y_prob):
        """Optimal cutoff point based on ROC curve for binary classifiers.

        Parameters
        ----------
        y : array-like
            Target variable
        y_prob : array-like
            Predicted probabilities

        Returns
        -------
        float

        Notes
        -----
        We can extend this to the multiclass setting by calculating Prob(X=0).
        For now just binary.
        """
        if self.problem_type == "classification":
            fpr, tpr, thresholds = metrics.roc_curve(y, y_prob)
            return thresholds[np.argmax(tpr - fpr)]
        else:
            print("Multiclass problem. Optimal cut-off can only be used for binary classifiers.")
            sys.exit()

    def gmm_stability(self, n_samples=10, gmm_kwargs={}):
        """Stability of Gaussian Mixture Model. Compares clustering with different runs
        """
        df_train = self.df_train.copy() # Training set
        if self.valid_set:
            df_test = self.df_valid.copy() # Test set
            clust_test_true = self.clust_valid_preds.copy()
        elif self.test_set:
            df_test = self.df_test.copy()
            clust_test_true = self.clust_test_preds.copy()
        else:
            print("No validation or test set.")
            sys.exit()
        if self.gmm_scaling:
            df_train[self.continuous_variables + self.discrete_variables] = self.scale_data(df_train)
            df_test[self.continuous_variables + self.discrete_variables] = self.scale_data(df_test)
        gmm_metrics = pd.DataFrame(columns=["ARI", "NMI", "Jaccard"])
        ari = []
        nmi = []
        jac = []
        for i in range(n_samples):
            
            gmm = GaussianMixture(n_components=self.n_components,
                                  covariance_type=self.covariance_type, 
                                  **gmm_kwargs).fit(df_train[self.cluster_variables])
            self.print_both("Sample {} complete...".format(i), log_level="info")
            clust_test_preds = gmm.predict(df_test[self.cluster_variables])
            ari.append(metrics.adjusted_rand_score(clust_test_true, clust_test_preds))
            nmi.append(metrics.normalized_mutual_info_score(clust_test_true, clust_test_preds))
            jac.append(metrics.jaccard_score(clust_test_true, clust_test_preds, average='weighted'))
        gmm_metrics["ARI"] = ari
        gmm_metrics["NMI"] = nmi
        gmm_metrics["Jaccard"] = jac
        self.print_both("Average Adjusted Rand Index: {:.4f} with standard deviation: {:.4f}".format(gmm_metrics["ARI"].mean(), gmm_metrics["ARI"].std()), log_level="info")
        self.print_both("Average Normalized Mutual Information: {:.4f} with standard deviation: {:.4f}".format(gmm_metrics["NMI"].mean(), gmm_metrics["NMI"].std()), log_level="info")
        self.print_both("Average Jaccard Index: {:.4f} with standard deviation: {:.4f}".format(gmm_metrics["Jaccard"].mean(), gmm_metrics["Jaccard"].std()), log_level="info")
        fig, ax = plt.subplots(1,3, figsize=(15, 5))
        sns.histplot(ax=ax[0], data=gmm_metrics, x="ARI", kde=True)
        sns.histplot(ax=ax[1], data=gmm_metrics, x="NMI", kde=True)
        sns.histplot(ax=ax[2], data=gmm_metrics, x="Jaccard", kde=True)
        plt.show()
        return gmm_metrics
            
    def gmm_selection(self, 
                n_components, 
                covariance_type, 
                cluster_variables,
                gmm_scaling = True,
                gmm_kwargs={}):
        """Fit Gaussian Mixture Model.
        Parameters
        ----------
        n_components : int
            Number of components
        covariance_type : str
            Covariance type
        cluster_variables : List
          List of variables to cluster
        gmm_scaling : bool
            Should the data be scaled before fitting the GMM. Default is True.

        """
        df_train = self.df_train.copy() # Training set
        if gmm_scaling:
            df_train[self.continuous_variables + self.discrete_variables] = self.scale_data(df_train) # Scale data
        gmm_results = pd.DataFrame(columns=["n_components", "covariance_type", "bic", "avg_silhouette"])
        bic = []
        silhouette = []
        for i in range(1, n_components+1):
            gmm = GaussianMixture(n_components=i,
                                    covariance_type=covariance_type,
                                    random_state=self.random_state,
                                    **gmm_kwargs).fit(df_train[cluster_variables])  # Fit GMM
            self.print_both("Converged: {}".format(gmm.converged_), log_level="info")
            self.print_both("Iterations: {}".format(gmm.n_iter_), log_level="info")
            clust_train_preds = gmm.predict(df_train[cluster_variables]) # Train set predictions
            #clust_train_proba = gmm.predict_proba(df_train[self.cluster_variables]) # Train set probabilities
            self.print_both("GMM fitted.", log_level="info")
            bic.append(gmm.bic(df_train[cluster_variables]))
            if i==1:
                silhouette.append(1)
            else:
                silhouette.append(metrics.silhouette_score(df_train[cluster_variables], clust_train_preds))
        gmm_results["n_components"] = np.arange(1, n_components+1)
        gmm_results["covariance_type"] = covariance_type
        gmm_results["bic"] = bic
        gmm_results["avg_silhouette"] = silhouette
        return gmm_results

    def fit_gmm(self, 
                n_components, 
                covariance_type, 
                cluster_variables,
                gmm_scaling = True,
                gmm_kwargs={}):
        """Fit Gaussian Mixture Model.

        Parameters
        ----------
        n_components : int
            Number of components
        covariance_type : str
            Covariance type
        cluster_variables : List
          List of variables to cluster
        gmm_scaling : bool
            Should the data be scaled before fitting the GMM. Default is True.

            
        New Attributes
        --------------
        n_components
        covariance_type
        cluster_variables 
        gmm_scaling
        gmm
        clust_train_preds
        clust_train_proba
        clust_valid_preds
        clust_valid_proba
        clust_test_preds
        clust_test_proba
        """
        self.n_components = n_components # Store Attribute
        self.covariance_type = covariance_type # Store Attribute
        self.cluster_variables = cluster_variables # Store Attribute
        self.gmm_scaling = gmm_scaling # Store Attribute
        df_train = self.df_train.copy() # Training set
        if self.gmm_scaling:
            df_train[self.continuous_variables + self.discrete_variables] = self.scale_data(df_train) # Scale data
        self.gmm = GaussianMixture(n_components=self.n_components,
                                   covariance_type=self.covariance_type,
                                   random_state=self.random_state,
                                   **gmm_kwargs).fit(df_train[self.cluster_variables])  # Fit GMM
        self.print_both("Converged: {}".format(self.gmm.converged_), log_level="info")
        self.print_both("Iterations: {}".format(self.gmm.n_iter_), log_level="info")
        self.clust_train_preds = self.gmm.predict(df_train[self.cluster_variables]) # Train set predictions
        self.clust_train_proba = self.gmm.predict_proba(df_train[self.cluster_variables]) # Train set probabilities
        if self.valid_set:
            df_valid = self.df_valid.copy()
            if self.gmm_scaling:
                df_valid[self.continuous_variables + self.discrete_variables] = self.scale_data(df_valid) # Scale data
            self.clust_valid_preds = self.gmm.predict(df_valid[self.cluster_variables]) # Validation set predictions
            self.clust_valid_proba = self.gmm.predict_proba(df_valid[self.cluster_variables]) # Validation set probabilities
        if self.test_set:
            df_test = self.df_test.copy()
            if self.gmm_scaling:
                df_test[self.continuous_variables + self.discrete_variables] = self.scale_data(df_test) # Scale data
            self.clust_test_preds = self.gmm.predict(df_test[self.cluster_variables]) # Test set predictions
            self.clust_test_proba = self.gmm.predict_proba(df_test[self.cluster_variables]) # Test set probabilities

        self.print_both("GMM fitted.", log_level="info")

    def glm_information_criterion(self,glm, which="bic_llf"):
        """Calculate BIC or AIC. BIC can be based on log-likelihood function or deviance.

        Parameters
        ----------
        glm : Instance of GLM from statsmodels
        which : str
            Which information criterion to use. Default is bic_llf.
        
        Returns
        -------
        float
        """
        if which == "bic_llf":
            return glm.bic_llf
        elif which == "bic_deviance":
            return glm.bic_deviance
        elif which == "aic":
            return glm.aic
        else:
            print("Invalid which parameter. Choose from 'bic_llf', 'bic_deviance', 'aic'")

    def glm_metric(self, y_true, y_pred, metric="mavg", problem_type="classification", metric_kwargs={}):
        """Calculate metric.

        Parameters
        ----------
        y_true : array-like
            Target variable
        y_pred : array-like
            Predicted values
        metric : str
            Which metric to use. Default is rmse.
        
        Returns
        -------
        float
        """
        if problem_type=="classification": # This works but may need improvement for multiclass or when class predictions are all 0 or 1.
        ## Classification metrics
            if metric == "mavg":
                return gmean(metrics.recall_score(y_true=y_true, y_pred=y_pred, average=None, zero_division=0.))
            elif metric == "accuracy_score":
                return metrics.accuracy_score(y_true=y_true, y_pred=y_pred, **metric_kwargs)
            elif metric == "balanced_accuracy_score":
                return metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred, **metric_kwargs)
            elif metric == "top_k_accuracy_score":
                return metrics.top_k_accuracy_score(y_true=y_true, y_score=y_pred, k=1, **metric_kwargs)
            elif metric == "average_precision_score":
                return metrics.average_precision_score(y_true=y_true, y_score=y_pred, **metric_kwargs)
            elif metric == "brier_score_loss":
                return metrics.brier_score_loss(y_true=y_true, y_prob=y_pred, **metric_kwargs)
            elif metric == "f1_score":
                return metrics.f1_score(y_true=y_true, y_pred=y_pred, **metric_kwargs)
            elif metric == "log_loss":
                return metrics.log_loss(y_true=y_true, y_pred=y_pred, **metric_kwargs)
            elif metric == "precision_score":
                return metrics.precision_score(y_true=y_true, y_pred=y_pred, zero_division=0., **metric_kwargs)
            elif metric == "recall_score":
                return metrics.recall_score(y_true=y_true, y_pred=y_pred, zero_division=0., **metric_kwargs)
            elif metric == "jaccard_score":
                return metrics.jaccard_score(y_true=y_true, y_pred=y_pred, **metric_kwargs)
            elif metric == "roc_auc_score":
                return metrics.roc_auc_score(y_true=y_true, y_score=y_pred, **metric_kwargs)
            else:
                print("Metric provided is for Regression problems but predictions are classes. Choose from 'mavg, 'accuracy_score', 'balanced_accuracy_score', 'top_k_accuracy_score', 'average_precision_score', 'brier_score_loss', 'f1_score', 'log_loss', 'precision_score', 'recall_score', 'jaccard_score', 'roc_auc_score'.")
                sys.exit()
        else:
        ## Regression metrics
            if metric == "r2_score":
                return metrics.r2_score(y_true=y_true, y_pred=y_pred, **metric_kwargs)
            elif metric == "mean_absolute_error":
                return metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred, **metric_kwargs)
            elif metric == "mean_squared_error":
                return metrics.mean_squared_error(y_true=y_true, y_pred=y_pred, **metric_kwargs)
            elif metric == "mean_squared_log_error":
                return metrics.mean_squared_log_error(y_true=y_true, y_pred=y_pred, **metric_kwargs)
            elif metric == "mean_absolute_percentage_error":
                return metrics.mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred, **metric_kwargs)
            elif metric == "median_absolute_error":
                return metrics.median_absolute_error(y_true=y_true, y_pred=y_pred, **metric_kwargs)
            elif metric == "max_error":
                return metrics.max_error(y_true=y_true, y_pred=y_pred, **metric_kwargs)
            elif metric == "explained_variance_score":
                return metrics.explained_variance_score(y_true=y_true, y_pred=y_pred, **metric_kwargs)
            elif metric == "mean_tweedie_deviance":
                return metrics.mean_tweedie_deviance(y_true=y_true, y_pred=y_pred, **metric_kwargs)
            elif metric == "mean_pinball_loss":
                return metrics.mean_pinball_loss(y_true=y_true, y_pred=y_pred, **metric_kwargs)
            elif metric == "root_mean_squared_error":
                return metrics.root_mean_squared_error(y_true=y_true, y_pred=y_pred, **metric_kwargs)
            elif metric == "root_mean_squared_log_error":
                return metrics.root_mean_squared_log_error(y_true=y_true, y_pred=y_pred, **metric_kwargs)
            elif metric == "mean_poisson_deviance":
                return metrics.mean_poisson_deviance(y_true=y_true, y_pred=y_pred, **metric_kwargs)
            elif metric == "mean_gamma_deviance":
                return metrics.mean_gamma_deviance(y_true=y_true, y_pred=y_pred, **metric_kwargs)
            elif metric == "d2_absolute_error_score":
                return metrics.d2_absolute_error_score(y_true=y_true, y_pred=y_pred, **metric_kwargs)
            else:
                print("Metric provided is for classifications problems but predictions are numerical. Choose from 'r2_score', 'mean_absolute_error', 'mean_squared_error', 'mean_squared_log_error', 'mean_absolute_percentage_error', 'median_absolute_error', 'max_error', 'explained_variance_score', 'mean_tweedie_deviance', 'mean_pinball_loss', 'root_mean_squared_error', 'root_mean_squared_log_error', 'mean_poisson_deviance', 'mean_gamma_deviance', 'd2_absolute_error_score'.")
                sys.exit()

    def stepwise_glm(self,
                     glm_family=sm.families.Binomial(link=sm.families.links.Logit()), # should be in GLM
                     problem_type='classification',
                     metric='bic_llf',
                     poly_order=1,
                     use_sampling=False,
                     use_opt_cutoff=False,
                     sample_size=None,
                     glm_scaling=True,
                     offset=None, 
                     exposure=None, 
                     freq_weights=None, 
                     var_weights=None, 
                     missing='raise',
                     glm_intercept=True,
                     glm_kwargs={},
                     fit_kwargs={}, 
                     metric_kwargs={}):
        """
        Parameters
        ----------
        metric : str
            Metric to use for stepwise regression. Default is 'bic'.
        poly_order : int
            Order of polynomial terms. Default is 1.
        use_sampling : bool
            Should sampling be used. Default is False.
        sample_size : int
            Size of sample if sampling is used. Default is None.
        glm_scaling : bool
            Should data be scaled before regression. Default is True.
        offset : str or None
            Offset variable for regression. Default is None.
        exposure : str or None
            Exposure variable for regression. Default is None.
        freq_weights : str or None
            Frequency weights for regression. Default is None.
        var_weights : str or None
            Variance weights for regression. Default is None.
        missing : str
            Available options are none, drop, and raise. If none, no nan checking is done. If drop, any observations with nans are dropped. If raise, an error is raised. Default is raise.

        New Attributes
        --------------
        stepwise_regression_variables
        
        Modified Attributes
        -------------------
        use_sampling - Also created in fit_glm
        glm_scaling - Also created in fit_glm

        Notes
        -----
        We can extend the multiclass setting by calculating Prob(X=0) so we can use accuracy.
        For now just binary.
        """
        # Catch errors
        if metric not in {'bic_llf', 'aic', 'bic_deviance', 'pvalue', 'mavg', 'r2_score', 'mean_absolute_error', 'mean_squared_error', 'mean_squared_log_error', 'mean_absolute_percentage_error', 'median_absolute_error', 'max_error', 'explained_variance_score', 'mean_tweedie_deviance', 'mean_pinball_loss', 'accuracy_score', 'balanced_accuracy_score', 'top_k_accuracy_score', 'average_precision_score', 'brier_score_loss', 'f1_score', 'log_loss', 'precision_score', 'recall_score', 'jaccard_score', 'roc_auc_score', 'root_mean_squared_error', 'root_mean_squared_log_error', 'mean_poisson_deviance', 'mean_gamma_deviance', 'd2_absolute_error_score'}:
            self.print_both("Invalid metric. Choose from 'bic_llf', 'aic', 'bic_deviance', 'pvalue', 'mavg' 'r2_score', 'mean_absolute_error', 'mean_squared_error', 'mean_squared_log_error', 'mean_absolute_percentage_error', 'median_absolute_error', 'max_error', 'explained_variance_score', 'mean_tweedie_deviance', 'mean_pinball_loss', 'accuracy_score', 'balanced_accuracy_score', 'top_k_accuracy_score', 'average_precision_score', 'brier_score_loss', 'f1_score', 'log_loss', 'precision_score', 'recall_score', 'jaccard_score', 'roc_auc_score', 'root_mean_squared_error', 'root_mean_squared_log_error', 'mean_poisson_deviance', 'mean_gamma_deviance', 'd2_absolute_error_score'.", log_level="error")
            sys.exit()  
        if metric not in {'bic_llf', 'aic', 'bic_deviance', 'pvalue'}:
            if self.valid_set:
                df_valid = self.df_valid.copy()
            else:
                self.print_both("Validation set is required for rmse and accuracy", log_level="error")
                sys.exit()
        if use_sampling and sample_size is None:
            self.print_both("Sample size is required when using sampling", log_level="error")
            sys.exit()
        self.problem_type=problem_type
        self.glm_family = glm_family # should be default of glm
        self.use_sampling = use_sampling
        self.glm_scaling = glm_scaling
        self.glm_offset = offset
        self.glm_exposure = exposure
        self.glm_freq_weights = freq_weights
        self.glm_var_weights = var_weights
        self.glm_missing = missing
        # create empty dictionary to store regression_variables
        regression_variables_dict = {}
        for i in range(self.n_components):
            regression_variables_dict[i] = []
        # Variables that can be added
        regression_variables = self.continuous_variables + self.discrete_variables + self.nominal_variables
        if self.glm_offset is not None:
            if isinstance(self.glm_offset, list): # Neeeds to be a list for set
                regression_variables = list(set(regression_variables) - set(self.glm_offset))
            else:
                regression_variables = list(set(regression_variables) - set([self.glm_offset]))
        if self.glm_exposure is not None:
            if isinstance(self.glm_exposure, list): # Neeeds to be a list for set
                regression_variables = list(set(regression_variables) - set(self.glm_exposure))
            else:
                regression_variables = list(set(regression_variables) - set([self.glm_exposure]))
        if self.use_pca:
            regression_variables = list(set(regression_variables) - set(self.pca_vars))
            regression_variables += self.pca_variables
        if poly_order > 1:
            poly_variables = []
            proposed_variables = regression_variables.copy()
            if self.use_pca: # Dont want to add PCA variables to the polynomial terms
                proposed_variables = list(set(proposed_variables) - set(self.pca_variables))
            for p in range(2, poly_order + 1):
                for var in proposed_variables:
                    if 'nominal' in var:
                        continue
                    poly_variables += ['I(' + var + ' ** ' + str(p) + ')']
            regression_variables += poly_variables
        # Copy of training and validation set
        df_train = self.df_train.copy()
        # Iterate over n_components
        for i in tqdm(range(self.n_components)):
            self.print_both("Cluster {}:\n".format(i), log_level="info")
            # training and validation set for cluster i
            if self.use_sampling:
                try:
                    clust_i_df_sample = getattr(self, 'clust' + str(i) + '_glm_df_sample').copy()
                    clust_i_df_sample_index = clust_i_df_sample.index.values
                except:
                    np.random.seed(self.random_state)
                    clust_i_prob = self.clust_train_proba[:, i] / sum(self.clust_train_proba[:, i])
                    clust_i_df_sample_index = np.random.choice(
                        df_train.index.values,
                        size=sample_size,
                        replace=False,
                        p=clust_i_prob)
                    clust_i_df_sample = df_train.loc[clust_i_df_sample_index].copy() # unscaled
                    setattr(self, 'clust' + str(i) + '_glm_df_sample', clust_i_df_sample)
                clust_i_df_train = df_train.loc[clust_i_df_sample_index].copy()
            else:
                clust_i_df_train = df_train[self.clust_train_preds == i].copy() # Xtrain
            clust_i_glm_vars = regression_variables_dict[i] # glm variables
            init_formula = self.glm_formula(regression_variables=clust_i_glm_vars, intercept=glm_intercept) # formula
            #clust_i_glm_vars.extend(self.target_variable) #+= [self.target_variable] # Add target variables
            if self.glm_offset is None: # Offset
                clust_i_df_train_offset = None
            else:
                clust_i_df_train_offset = np.asarray(clust_i_df_train[self.glm_offset].copy()) # this is unscaled
            if self.glm_exposure is None: # Exposure
                clust_i_df_train_exposure = None
            else:
                clust_i_df_train_exposure = np.asarray(clust_i_df_train[self.glm_exposure].copy()) # this is unscaled
            if self.glm_freq_weights is None: # freq weights
                clust_i_df_train_freq_weights = None
            else:
                clust_i_df_train_freq_weights = np.asarray(clust_i_df_train[self.glm_freq_weights].copy()) # this is unscaled
            if self.glm_var_weights is None: # var weights
                clust_i_df_train_var_weights = None
            else:
                clust_i_df_train_var_weights = np.asarray(clust_i_df_train[self.glm_var_weights].copy()) # this is unscaled
            if glm_scaling:  
                clust_i_df_train[self.continuous_variables + self.discrete_variables] = self.scale_data(clust_i_df_train)
            init_model = smf.glm( # Initial model   
                formula=init_formula,
                data=clust_i_df_train, 
                offset=clust_i_df_train_offset,
                exposure=clust_i_df_train_exposure,
                freq_weights=clust_i_df_train_freq_weights,
                var_weights=clust_i_df_train_var_weights,
                missing=self.glm_missing,
                family=self.glm_family,
                **glm_kwargs).fit(**fit_kwargs)
            # Edit this
            if metric == 'bic_llf' or metric == 'aic' or metric == 'bic_deviance':
                threshold = self.glm_information_criterion(glm=init_model, which=metric)
            elif metric == 'pvalue':
                threshold = 0.05
            else: # classification or regressoion metrics
                clust_i_df_valid = df_valid[self.clust_valid_preds == i].copy() # Xvalid
                if self.glm_exposure is None:
                    clust_i_df_valid_exposure = None
                else:
                    clust_i_df_valid_exposure = np.asarray(clust_i_df_valid[self.glm_exposure].copy())
                if self.glm_offset is None:
                    clust_i_df_valid_offset = None
                else:
                    clust_i_df_valid_offset = np.asarray(clust_i_df_valid[self.glm_offset].copy())
                if glm_scaling:
                    clust_i_df_valid[self.continuous_variables + self.discrete_variables] = self.scale_data(clust_i_df_valid)
                y_valid_true = clust_i_df_valid[self.target_variable[0]]
                if self.problem_type == 'classification':
                    if use_opt_cutoff:
                        opt_cutoff = self.optimal_cutoff_point(
                            y=clust_i_df_train[self.target_variable],
                            y_prob=init_model.predict(clust_i_df_train, which="mean", offset=clust_i_df_train_offset, exposure=clust_i_df_train_exposure))
                    else:
                        opt_cutoff = 0.5
                    y_valid_pred = init_model.predict(clust_i_df_valid, which="mean", exposure=clust_i_df_valid_exposure, offset=clust_i_df_valid_offset)
                    if metric not in {'top_k_accuracy_score', 'average_precision_score', 'brier_score_loss', 'log_loss', 'roc_auc_score'}: # They require probabilities, not 0 or 1
                        y_valid_pred = (y_valid_pred>=opt_cutoff) * 1
                    threshold = self.glm_metric(y_true=y_valid_true, y_pred=y_valid_pred, metric=metric, problem_type=self.problem_type, metric_kwargs=metric_kwargs)
                else:
                    y_valid_pred = init_model.predict(clust_i_df_valid, which="mean", exposure=clust_i_df_valid_exposure, offset=clust_i_df_valid_offset)
                    #if self.trials:
                    #    y_valid_pred = y_valid_pred * clust_i_df_valid['trials']
                    threshold = self.glm_metric(y_true=y_valid_true, y_pred=y_valid_pred, metric=metric, problem_type=self.problem_type, metric_kwargs=metric_kwargs)        
            self.print_both("Initial Threshold: {}".format(threshold), log_level="info")
            #clust_i_glm_vars = list(set(clust_i_glm_vars) - set(self.target_variable))#.remove(self.target_variable)
            while True:
                changed = False
                # forward step
                included = regression_variables_dict[i].copy()
                excluded = list(set(regression_variables) - set(included))
                if excluded == []:  # Catch if excluded is empty
                    break
                new_threshold = pd.Series(index=excluded, dtype='float64')
                for variable in excluded:
                    clust_i_glm_vars = included + [variable]
                    formula = self.glm_formula(regression_variables=clust_i_glm_vars, intercept=glm_intercept) # formula
                    #clust_i_glm_vars.extend(self.target_variable) #+= [self.target_variable] # Add target variable      
                    try:  # Catch singular matrix
                        model = smf.glm(
                            formula=formula,
                            data=clust_i_df_train, 
                            offset=clust_i_df_train_offset,
                            exposure=clust_i_df_train_exposure,
                            freq_weights=clust_i_df_train_freq_weights,
                            var_weights=clust_i_df_train_var_weights,
                            missing=self.glm_missing,
                            family=self.glm_family, 
                            **glm_kwargs).fit(**fit_kwargs)
                            # calculate bic
                        #clust_i_glm_vars = list(set(clust_i_glm_vars) - set(self.target_variable))#.remove(self.target_variable)
                        if np.isnan(model.llf):#mle_retvals['converged'] == False:
                            if metric in {'pvalue', 'brier_score_loss'}:
                                new_threshold[variable] = 1.
                            elif metric in {'mavg',  'accuracy_score', 'balanced_accuracy_score', 'f1_score', 'precision_score', 'recall_score', 'jaccard_score', 'top_k_accuracy_score', 'average_precision_score', 'roc_auc_score'}:
                                new_threshold[variable] = 0.
                            elif metric in {'r2_score', 'explained_variance_score', 'd2_absolute_error_score'}:
                                new_threshold[variable] = threshold - 1
                            else:
                                new_threshold[variable] = threshold + 1
                            self.print_both('{} failed to converge'.format(variable), log_level="info")
                        else:
                            #clust_i_glm_vars = list(set(clust_i_glm_vars) - set(self.target_variable))#.remove(self.target_variable)
                            # Edit this
                            if metric == 'bic_llf' or metric == 'aic' or metric == 'bic_deviance':
                                new_threshold[variable] = self.glm_information_criterion(glm=model, which=metric)
                            elif metric == 'pvalue':
                                if 'nominal' in variable:
                                        # Add the smallest p value if it is nominal with multiple levels
                                        # first identify which pvalue belongs to the variable using a list of booleans
                                    boolean =[variable in string for string in model.pvalues.index]
                                    new_threshold[variable] = min(model.pvalues[boolean])
                                else:
                                    new_threshold[variable] = model.pvalues[variable]
                            else: # rmse, accuracy
                                if self.problem_type == 'classification':
                                    if use_opt_cutoff:
                                        opt_cutoff = self.optimal_cutoff_point(
                                            y=clust_i_df_train[self.target_variable],
                                            y_prob=model.predict(clust_i_df_train, which="mean", exposure=clust_i_df_train_exposure, offset=clust_i_df_train_offset))
                                    y_valid_pred = model.predict(clust_i_df_valid, which="mean", exposure=clust_i_df_valid_exposure, offset=clust_i_df_valid_offset)
                                    if metric not in {'top_k_accuracy_score', 'average_precision_score', 'brier_score_loss', 'log_loss', 'roc_auc_score'}: # They require probabilities, not 0 or 1
                                        y_valid_pred = (y_valid_pred>=opt_cutoff) * 1
                                    new_threshold[variable] = self.glm_metric(y_true=y_valid_true, y_pred=y_valid_pred, metric=metric, problem_type=self.problem_type, metric_kwargs=metric_kwargs)
                                else:
                                    y_valid_pred = model.predict(clust_i_df_valid, which="mean", exposure=clust_i_df_valid_exposure, offset=clust_i_df_valid_offset)
                                    if self.trials:
                                        y_valid_pred = y_valid_pred * clust_i_df_valid['trials']
                                    new_threshold[variable] = self.glm_metric(y_true=y_valid_true, y_pred=y_valid_pred, metric=metric, problem_type=self.problem_type, metric_kwargs=metric_kwargs)  
                    except:
                        #clust_i_glm_vars = list(set(clust_i_glm_vars) - set(self.target_variable))#.remove(self.target_variable)
                        self.print_both("Singular Matrix for {}".format(variable), log_level="warning")
                        if metric in {'pvalue', 'brier_score_loss'}:
                            new_threshold[variable] = 1.
                        elif metric in {'mavg',  'accuracy_score', 'balanced_accuracy_score', 'f1_score', 'precision_score', 'recall_score', 'jaccard_score', 'top_k_accuracy_score', 'average_precision_score', 'roc_auc_score'}:
                            new_threshold[variable] = 0.
                        elif metric in {'r2_score', 'explained_variance_score', 'd2_absolute_error_score'}:
                            new_threshold[variable] = threshold - 1
                        else:
                            new_threshold[variable] = threshold + 1
                self.print_both(new_threshold, log_level="info")
                if metric in {'bic_llf', 'aic', 'bic_deviance', 'pvalue', 'brier_score_loss', 'log_loss', 'max_error', 'mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error', 'mean_squared_log_error', 'root_mean_squared_log_error', 'median_absolute_error', 'mean_absolute_percentage_error', 'mean_poisson_deviance', 'mean_tweedie_deviance', 'mean_pinball_loss', 'mean_poisson_deviance'}: # Minimize
                    best_threshold = new_threshold.min()
                    if best_threshold < threshold:
                        best_variable = new_threshold.index[new_threshold.argmin()]
                        regression_variables_dict[i].append(best_variable)
                        changed = True
                        self.print_both('Add {} with {}: {}'.format(best_variable, metric, best_threshold), log_level="info")
                        threshold = best_threshold
                else: # Maximize 
                    best_threshold = new_threshold.max()
                    if best_threshold > threshold:
                        best_variable = new_threshold.index[new_threshold.argmax()]
                        regression_variables_dict[i].append(best_variable)
                        changed = True
                        self.print_both('Add {} with {}: {}'.format(best_variable, metric, best_threshold), log_level="info")
                        threshold = best_threshold
                if not changed:
                    self.print_both('No variable to add.', log_level="info")
                    break
        self.stepwise_regression_variables = regression_variables_dict

    def fit_glm(self,
                regression_variables,
                glm_family=sm.families.Binomial(link=sm.families.links.Logit()), # should be in GLM
                problem_type='classification',
                use_opt_cutoff=True,
                use_sampling=False,
                sample_size=None,
                weighted=False,
                glm_scaling=True,
                offset=None, 
                exposure=None, 
                freq_weights=None, 
                var_weights=None, 
                missing='raise',
                glm_intercept=True,
                glm_kwargs=None,
                fit_kwargs=None):
        """
        Parameters
        ----------
        regression_variables : List of list of regression variables (list[list[(str)]])
        use_opt_cutoff : Should optimal cutoff be used in binary regression (bool, default=True)
        use_sampling : Should sampling be used (bool, default=False)
        sample_size : Size of sample if sampling is used (int, default=None)
        weighted : Should weighted predictions be used for validation and test sets (bool, default=False)
        glm_scaling : Should data be scaled before regression (bool, default=True)

        New variables created:
        ----------------------
        use_sampling - Also created in stepwise approach
        glm_scaling - Also created in stepwise approach
        clusti_glm_vars
        clusti_glm_df_sample - Also created in stepwise approach

        Notes
        -----
        Add Class for Multiclass
        """
        if glm_kwargs is None:
            glm_kwargs = {}
        if fit_kwargs is None:
            fit_kwargs = {}
        self.problem_type=problem_type
        self.glm_family = glm_family # should be default of glm
        self.use_sampling = use_sampling # Store attribute whether to use sampling
        self.glm_scaling = glm_scaling # Store attribute whether to scale data before regression
        self.glm_offset = offset
        self.glm_exposure = exposure
        self.glm_freq_weights = freq_weights
        self.glm_var_weights = var_weights
        self.glm_missing = missing
        df_train = self.df_train.copy() # Training set
        glm_train_pred_mean = np.zeros((df_train.shape[0], )) # np.array for regression predictions
        glm_train_pred_linear = np.zeros((df_train.shape[0], ))  # np.array for regression fitted values
        if self.problem_type=="classification":
            glm_train_pred_class = np.zeros((df_train.shape[0], ))
        if self.glm_exposure is not None and self.glm_offset is not None:
            glm_train_pred_mean_exposure_offset_adjusted = np.zeros((df_train.shape[0], ))
        elif self.glm_exposure is not None: # If there is an offset then fitted equals predictions * offset
            glm_train_pred_mean_exposure_adjusted = np.zeros((df_train.shape[0], ))
        elif self.glm_offset is not None: # If there is an offset then fitted equals predictions * offset
            glm_train_pred_mean_offset_adjusted = np.zeros((df_train.shape[0], ))
        if self.valid_set: # If there is a validation set
            df_valid = self.df_valid.copy()
            glm_valid_pred_mean = np.zeros((df_valid.shape[0], )) # np.array for regression predictions
            glm_valid_pred_linear = np.zeros((df_valid.shape[0], ))  # np.array for regression fitted values
            if self.problem_type=="classification":
                glm_valid_pred_class = np.zeros((df_valid.shape[0], ))  # np.array for regression fitted values
            if self.glm_exposure is not None and self.glm_offset is not None:
                glm_valid_pred_mean_exposure_offset_adjusted = np.zeros((df_valid.shape[0], ))
            elif self.glm_exposure is not None: # If there is an offset then fitted equals predictions * offset
                glm_valid_pred_mean_exposure_adjusted = np.zeros((df_valid.shape[0], ))
            elif self.glm_offset is not None: # If there is an offset then fitted equals predictions * offset
                glm_valid_pred_mean_offset_adjusted = np.zeros((df_valid.shape[0], ))
        if self.test_set: # If there is a test set
            df_test = self.df_test.copy()
            glm_test_pred_mean = np.zeros((df_test.shape[0], )) # np.array for regression predictions
            glm_test_pred_linear = np.zeros((df_test.shape[0], )) # np.array for regression fitted values
            if self.problem_type=="classification":
                glm_test_pred_class = np.zeros((df_test.shape[0], ))
            if self.glm_exposure is not None and self.glm_offset is not None:
                glm_test_pred_mean_exposure_offset_adjusted = np.zeros((df_test.shape[0], ))
            elif self.glm_exposure is not None: # If there is an offset then fitted equals predictions * offset
                glm_test_pred_mean_exposure_adjusted = np.zeros((df_test.shape[0], ))
            elif self.glm_offset is not None: # If there is an offset then fitted equals predictions * offset
                glm_test_pred_mean_offset_adjusted = np.zeros((df_test.shape[0], ))
        # Fit GLMs for each cluster. Training and prediction occurs in this loop
        for i in range(self.n_components):
            setattr(self, 'clust' + str(i) + '_glm_vars', regression_variables[i].copy()) # Regression variables for cluster i
            clust_i_glm_vars = regression_variables[i].copy()  # If this is not a copy then NB_Claim will be added to the list
            formula = self.glm_formula(regression_variables=clust_i_glm_vars, intercept=glm_intercept) # GLM Formula
            #clust_i_glm_vars.extend(self.target_variable) #+= [self.target_variable] # Add target variable
            clust_i_df_train = df_train[self.clust_train_preds == i].copy() # (Unscaled) Xtrain
            # Fit the GLM
            if self.use_sampling: # Sampling
                try:
                    clust_i_df_sample = getattr(self, 'clust' + str(i) + '_glm_df_sample').copy() # Get sampled data set if it exists
                except: # Perform sampling if it doesn't exist
                    np.random.seed(self.random_state)
                    clust_i_prob = self.clust_train_proba[:, i] / sum(self.clust_train_proba[:, i])
                    clust_i_df_sample_index = np.random.choice(
                        df_train.index.values,
                        size=sample_size,
                        replace=False,
                        p=clust_i_prob)
                    clust_i_df_sample = df_train.loc[clust_i_df_sample_index].copy() # This allows any observation to be selected
                    setattr(self, 'clust' + str(i) + '_glm_df_sample', clust_i_df_sample.copy()) # Set sampled as attribute
                if self.glm_offset is None: # Offset    
                    clust_i_df_sample_offset = None
                else:
                    clust_i_df_sample_offset = np.asarray(clust_i_df_sample[self.glm_offset].copy())
                if self.glm_exposure is None: # Exposure
                    clust_i_df_sample_exposure = None
                else:
                    clust_i_df_sample_exposure = np.asarray(clust_i_df_sample[self.glm_exposure].copy())
                if self.glm_freq_weights is None: # freq weights
                    clust_i_df_sample_freq_weights = None
                else:
                    clust_i_df_sample_freq_weights = np.asarray(clust_i_df_sample[[self.glm_freq_weights]].copy())
                if self.glm_var_weights is None: # var weights
                    clust_i_df_sample_var_weights = None
                else:
                    clust_i_df_sample_var_weights = np.asarray(clust_i_df_sample[[self.glm_var_weights]].copy())
                if glm_scaling:
                    clust_i_df_sample[self.continuous_variables+self.discrete_variables] = self.scale_data(clust_i_df_sample)
                    clust_i_df_train[self.continuous_variables+self.discrete_variables] = self.scale_data(clust_i_df_train) # Scaling for the predictions
                glm = smf.glm(formula=formula,
                              data=clust_i_df_sample,
                              offset=clust_i_df_sample_offset,
                              exposure=clust_i_df_sample_exposure,
                              freq_weights=clust_i_df_sample_freq_weights,
                              var_weights=clust_i_df_sample_var_weights,
                              missing=self.glm_missing,
                              family=self.glm_family, 
                              **glm_kwargs).fit(**fit_kwargs) # GLM
            else:
                if self.glm_offset is None: # Offset    
                    clust_i_df_train_offset = None
                else:
                    clust_i_df_train_offset = np.asarray(clust_i_df_train[self.glm_offset].copy())
                if self.glm_exposure is None: # Exposure
                    clust_i_df_train_exposure = None
                else:
                    clust_i_df_train_exposure = np.asarray(clust_i_df_train[self.glm_exposure].copy())
                if self.glm_freq_weights is None: # freq weights
                    clust_i_df_train_freq_weights = None
                else:
                    clust_i_df_train_freq_weights = np.asarray(clust_i_df_train[self.glm_freq_weights].copy())
                if self.glm_var_weights is None: # var weights
                    clust_i_df_train_var_weights = None
                else:
                    clust_i_df_train_var_weights = np.asarray(clust_i_df_train[self.glm_var_weights].copy())
                if glm_scaling:
                    clust_i_df_train[self.continuous_variables+self.discrete_variables] = self.scale_data(clust_i_df_train)
                glm = smf.glm(formula=formula,
                              data=clust_i_df_train, 
                              offset=clust_i_df_train_offset,
                              exposure=clust_i_df_train_exposure,
                              freq_weights=clust_i_df_train_freq_weights,
                              var_weights=clust_i_df_train_var_weights,
                              missing=self.glm_missing,
                              family=self.glm_family,
                              **glm_kwargs).fit(**fit_kwargs)  # GLM
            # Predictions
            self.print_both(glm.summary())
            setattr(self, 'clust' + str(i) + '_glm', glm) # Set GLM as attribute
            if weighted:
                if glm_scaling:
                    scaled_df_train = df_train.copy()
                    scaled_df_train[self.continuous_variables+self.discrete_variables] = self.scale_data(scaled_df_train)
                    glm_train_pred_mean += self.clust_train_proba[:, i] * glm.predict(scaled_df_train, which="mean").values # Predict train set
                    if self.glm_exposure is not None and self.glm_offset is not None: # If there is an offset then fitted equals predictions * offset
                        glm_train_pred_mean_exposure_offset_adjusted += self.clust_train_proba[:, i] * glm.predict(scaled_df_train, which="mean", exposure=np.asarray(df_train[self.glm_exposure]), offset=np.asarray(df_train[self.glm_offset])).values
                    elif self.glm_exposure is not None: # If there is an offset then fitted equals predictions * offset
                        glm_train_pred_mean_exposure_adjusted += self.clust_train_proba[:, i] * glm.predict(scaled_df_train, which="mean", exposure=np.asarray(df_train[self.glm_exposure])).values
                    elif self.glm_offset is not None: # If there is an offset then fitted equals predictions * offset
                        glm_train_pred_mean_offset_adjusted += self.clust_train_proba[:, i] * glm.predict(scaled_df_train, which="mean", offset=np.asarray(df_train[self.glm_offset])).values
                else:
                    glm_train_pred_mean += self.clust_train_proba[:, i] * glm.predict(df_train, which="mean").values # Predict train set
                    # Linear prediction must be done at the end
                    if self.glm_exposure is not None and self.glm_offset is not None: # If there is an offset then fitted equals predictions * offset
                        glm_train_pred_mean_exposure_offset_adjusted += self.clust_train_proba[:, i] * glm.predict(df_train, which="mean", exposure=np.asarray(df_train[self.glm_exposure]), offset=np.asarray(df_train[self.glm_offset])).values
                    elif self.glm_exposure is not None: # If there is an offset then fitted equals predictions * offset
                        glm_train_pred_mean_exposure_adjusted += self.clust_train_proba[:, i] * glm.predict(df_train, which="mean", exposure=np.asarray(df_train[self.glm_exposure])).values
                    elif self.glm_offset is not None: # If there is an offset then fitted equals predictions * offset
                        glm_train_pred_mean_offset_adjusted += self.clust_train_proba[:, i] * glm.predict(df_train, which="mean", offset=np.asarray(df_train[self.glm_offset])).values
            else:
                # No scaling here because clust_i_df_train will be the same that was used to fit the model
                glm_train_pred_mean[self.clust_train_preds == i] = glm.predict(clust_i_df_train, which="mean").values # Predict train set
                glm_train_pred_linear[self.clust_train_preds == i] = glm.predict(clust_i_df_train, which="linear").values # Predict train set
                if self.problem_type=="classification":
                    if use_opt_cutoff:
                        if use_sampling: # Need to decide opt cutoff on in-sample data, exclude out-of-sample
                            opt_cutoff = self.optimal_cutoff_point(clust_i_df_sample[self.target_variable], glm.predict(clust_i_df_sample).values)
                        else:
                            opt_cutoff = self.optimal_cutoff_point(clust_i_df_train[self.target_variable], glm_train_pred_mean[self.clust_train_preds == i])
                    else:
                        opt_cutoff = 0.5
                    setattr(self, 'clust' + str(i) + '_glm_cutoff', opt_cutoff.copy()) # Set optimal cutoff as attribute
                    glm_train_pred_class[self.clust_train_preds == i] = (glm_train_pred_mean[self.clust_train_preds == i] >= opt_cutoff) * 1
                if self.glm_exposure is not None and self.glm_offset is not None: # If there is an offset then fitted equals predictions * offset
                    glm_train_pred_mean_exposure_offset_adjusted[self.clust_train_preds == i] = glm.predict(clust_i_df_train, which="mean", exposure=clust_i_df_train_exposure, offset=clust_i_df_train_offset).values
                elif self.glm_exposure is not None: # If there is an offset then fitted equals predictions * offset
                    glm_train_pred_mean_exposure_adjusted[self.clust_train_preds == i] = glm.predict(clust_i_df_train, which="mean", exposure=clust_i_df_train_exposure).values
                elif self.glm_offset is not None: # If there is an offset then fitted equals predictions * offset
                    glm_train_pred_mean_offset_adjusted[self.clust_train_preds == i] = glm.predict(clust_i_df_train, which="mean", offset=clust_i_df_train_offset).values
            if self.valid_set: # Could I just use predict()?
                clust_i_df_valid = df_valid[self.clust_valid_preds == i].copy() # (Unscaled) Xvalid
                if weighted: # Predict each observation and weight by cluster probability
                    # The Linear and class prediction must be done at the end
                    if glm_scaling:
                        scaled_df_valid = df_valid.copy()
                        scaled_df_valid[self.continuous_variables+self.discrete_variables] = self.scale_data(scaled_df_valid)
                        glm_valid_pred_mean += self.clust_valid_proba[:, i] * glm.predict(scaled_df_valid, which="mean").values
                        if self.glm_exposure is not None and self.glm_offset is not None: # If there is an offset then fitted equals predictions * offset
                            glm_valid_pred_mean_exposure_offset_adjusted += self.clust_valid_proba[:, i] * glm.predict(scaled_clust_i_df_valid, which="mean", exposure=np.asarray(clust_i_df_valid[self.glm_exposure]), offset=np.asarray(clust_i_df_valid[self.glm_offset])).values
                        elif self.glm_exposure is not None: # If there is an offset then fitted equals predictions * offset
                            glm_valid_pred_mean_exposure_adjusted += self.clust_valid_proba[:, i] * glm.predict(scaled_clust_i_df_valid, which="mean", exposure=np.asarray(clust_i_df_valid[self.glm_exposure])).values
                        elif self.glm_offset is not None: # If there is an offset then fitted equals predictions * offset
                            glm_valid_pred_mean_offset_adjusted += self.clust_valid_proba[:, i] * glm.predict(scaled_clust_i_df_valid, which="mean", offset=np.asarray(clust_i_df_valid[self.glm_offset])).values
                    else:
                        glm_valid_pred_mean += self.clust_valid_proba[:, i] * glm.predict(df_valid, which="mean").values
                        if self.glm_exposure is not None and self.glm_offset is not None: # If there is an offset then fitted equals predictions * offset
                            glm_valid_pred_mean_exposure_offset_adjusted += self.clust_valid_proba[:, i] * glm.predict(clust_i_df_valid, which="mean", exposure=np.asarray(clust_i_df_valid[self.glm_exposure]), offset=np.asarray(clust_i_df_valid[self.glm_offset])).values
                        elif self.glm_exposure is not None: # If there is an offset then fitted equals predictions * offset
                            glm_valid_pred_mean_exposure_adjusted += self.clust_valid_proba[:, i] * glm.predict(clust_i_df_valid, which="mean", exposure=np.asarray(clust_i_df_valid[self.glm_exposure])).values
                        elif self.glm_offset is not None: # If there is an offset then fitted equals predictions * offset
                            glm_valid_pred_mean_offset_adjusted += self.clust_valid_proba[:, i] * glm.predict(clust_i_df_valid, which="mean", offset=np.asarray(clust_i_df_valid[self.glm_offset])).values
                else:
                    if glm_scaling:
                        scaled_clust_i_df_valid = clust_i_df_valid.copy()
                        scaled_clust_i_df_valid[self.continuous_variables+self.discrete_variables] = self.scale_data(scaled_clust_i_df_valid)
                        glm_valid_pred_mean[self.clust_valid_preds == i] = glm.predict(scaled_clust_i_df_valid, which="mean").values # Predict validation set for cluster i
                        glm_valid_pred_linear[self.clust_valid_preds == i] = glm.predict(scaled_clust_i_df_valid, which="linear").values # Predict validation set for cluster i
                        if self.problem_type=="classification":
                            glm_valid_pred_class[self.clust_valid_preds == i] = (glm_valid_pred_mean[self.clust_valid_preds == i]>= opt_cutoff) * 1 
                        if self.glm_exposure is not None and self.glm_offset is not None: # If there is an offset then fitted equals predictions * offset
                            glm_valid_pred_mean_exposure_offset_adjusted[self.clust_valid_preds == i] = glm.predict(scaled_clust_i_df_valid, which="mean", exposure=np.asarray(clust_i_df_valid[self.glm_exposure]), offset=np.asarray(clust_i_df_valid[self.glm_offset])).values
                        elif self.glm_exposure is not None: # If there is an offset then fitted equals predictions * offset
                            glm_valid_pred_mean_exposure_adjusted[self.clust_valid_preds == i] = glm.predict(scaled_clust_i_df_valid, which="mean", exposure=np.asarray(clust_i_df_valid[self.glm_exposure])).values
                        elif self.glm_offset is not None: # If there is an offset then fitted equals predictions * offset
                            glm_valid_pred_mean_offset_adjusted[self.clust_valid_preds == i] = glm.predict(scaled_clust_i_df_valid, which="mean", offset=np.asarray(clust_i_df_valid[self.glm_offset])).values
                    else:
                        glm_valid_pred_mean[self.clust_valid_preds == i] = glm.predict(clust_i_df_valid, which="mean").values 
                        glm_valid_pred_linear[self.clust_valid_preds == i] = glm.predict(clust_i_df_valid, which="linear").values 
                        if self.problem_type=="classification":
                            glm_valid_pred_class[self.clust_valid_preds == i] = (glm_valid_pred_mean[self.clust_valid_preds == i]>= opt_cutoff) * 1 
                        if self.glm_exposure is not None and self.glm_offset is not None: # If there is an offset then fitted equals predictions * offset
                            glm_valid_pred_mean_exposure_offset_adjusted[self.clust_valid_preds == i] = glm.predict(clust_i_df_valid, which="mean", exposure=np.asarray(clust_i_df_valid[self.glm_exposure]), offset=np.asarray(clust_i_df_valid[self.glm_offset])).values
                        elif self.glm_exposure is not None: # If there is an offset then fitted equals predictions * offset
                            glm_valid_pred_mean_exposure_adjusted[self.clust_valid_preds == i] = glm.predict(clust_i_df_valid, which="mean", exposure=np.asarray(clust_i_df_valid[self.glm_exposure])).values
                        elif self.glm_offset is not None: # If there is an offset then fitted equals predictions * offset
                            glm_valid_pred_mean_offset_adjusted[self.clust_valid_preds == i] = glm.predict(clust_i_df_valid, which="mean", offset=np.asarray(clust_i_df_valid[self.glm_offset])).values
            if self.test_set: # Could I just use predict()?
                clust_i_df_test = df_test[self.clust_test_preds == i].copy() # (Unscaled) Xtest
                if weighted: # Predict each observation and weight by cluster probability
                    if glm_scaling:
                        scaled_df_test = df_test.copy()
                        scaled_df_test[self.continuous_variables+self.discrete_variables] = self.scale_data(scaled_df_test)
                        glm_test_pred_mean += self.clust_test_proba[:, i] * glm.predict(scaled_df_test, which="mean").values
                        if self.glm_exposure is not None and self.glm_offset is not None: # If there is an offset then fitted equals predictions * offset
                            glm_test_pred_mean_exposure_offset_adjusted += self.clust_test_proba[:, i] * glm.predict(scaled_df_test, which="mean", exposure=np.asarray(df_test[self.glm_exposure]), offset=np.asarray(df_test[self.glm_offset])).values
                        elif self.glm_exposure is not None: # If there is an offset then fitted equals predictions * offset
                            glm_test_pred_mean_exposure_adjusted += self.clust_test_proba[:, i] * glm.predict(scaled_df_test, which="mean", exposure=np.asarray(df_test[self.glm_exposure])).values
                        elif self.glm_offset is not None: # If there is an offset then fitted equals predictions * offset
                            glm_test_pred_mean_offset_adjusted += self.clust_test_proba[:, i] * glm.predict(scaled_df_test, which="mean", offset=np.asarray(df_test[self.glm_offset])).values
                    else:
                        glm_test_pred_mean += self.clust_test_proba[:, i] * glm.predict(df_test, which="mean").values
                        if self.glm_exposure is not None and self.glm_offset is not None: # If there is an offset then fitted equals predictions * offset
                            glm_test_pred_mean_exposure_offset_adjusted += self.clust_test_proba[:, i] * glm.predict(clust_i_df_test, which="mean", exposure=np.asarray(clust_i_df_test[self.glm_exposure]), offset=np.asarray(clust_i_df_test[self.glm_offset])).values
                        elif self.glm_exposure is not None: # If there is an offset then fitted equals predictions * offset
                            glm_test_pred_mean_exposure_adjusted += self.clust_test_proba[:, i] * glm.predict(clust_i_df_test, which="mean", exposure=np.asarray(clust_i_df_test[self.glm_exposure])).values
                        elif self.glm_offset is not None: # If there is an offset then fitted equals predictions * offset
                            glm_test_pred_mean_offset_adjusted += self.clust_test_proba[:, i] * glm.predict(clust_i_df_test, which="mean", offset=np.asarray(clust_i_df_test[self.glm_offset])).values
                else:
                    if glm_scaling:
                        scaled_clust_i_df_test = clust_i_df_test.copy()
                        scaled_clust_i_df_test[self.continuous_variables+self.discrete_variables] = self.scale_data(scaled_clust_i_df_test)
                        glm_test_pred_mean[self.clust_test_preds == i] = glm.predict(scaled_clust_i_df_test, which="mean").values # Predict test set for cluster i
                        glm_test_pred_linear[self.clust_test_preds == i] = glm.predict(scaled_clust_i_df_test, which="linear").values # Predict test set for cluster i
                        if self.problem_type=="classification":
                            glm_test_pred_class[self.clust_test_preds == i] = (glm_test_pred_mean[self.clust_test_preds == i]>= opt_cutoff) * 1 
                        if self.glm_exposure is not None and self.glm_offset is not None: # If there is an offset then fitted equals predictions * offset
                            glm_test_pred_mean_exposure_offset_adjusted[self.clust_test_preds == i] = glm.predict(scaled_clust_i_df_test, which="mean", exposure=np.asarray(clust_i_df_test[self.glm_exposure]), offset=np.asarray(clust_i_df_test[self.glm_offset])).values
                        elif self.glm_exposure is not None: # If there is an offset then fitted equals predictions * offset
                            glm_test_pred_mean_exposure_adjusted[self.clust_test_preds == i] = glm.predict(scaled_clust_i_df_test, which="mean", exposure=np.asarray(clust_i_df_test[self.glm_exposure])).values
                        elif self.glm_offset is not None: # If there is an offset then fitted equals predictions * offset
                            glm_test_pred_mean_offset_adjusted[self.clust_test_preds == i] = glm.predict(scaled_clust_i_df_test, which="mean", offset=np.asarray(clust_i_df_test[self.glm_offset])).values
                    else:
                        glm_test_pred_mean[self.clust_test_preds == i] = glm.predict(clust_i_df_test, which="mean").values 
                        glm_test_pred_linear[self.clust_test_preds == i] = glm.predict(clust_i_df_test, which="linear").values 
                        if self.problem_type=="classification":
                            glm_test_pred_class[self.clust_test_preds == i] = (glm_test_pred_mean[self.clust_test_preds == i]>= opt_cutoff) * 1 
                        if self.glm_exposure is not None and self.glm_offset is not None: # If there is an offset then fitted equals predictions * offset
                            glm_test_pred_mean_exposure_offset_adjusted[self.clust_test_preds == i] = glm.predict(clust_i_df_test, which="mean", exposure=np.asarray(clust_i_df_test[self.glm_exposure]), offset=np.asarray(clust_i_df_test[self.glm_offset])).values
                        elif self.glm_exposure is not None: # If there is an offset then fitted equals predictions * offset
                            glm_test_pred_mean_exposure_adjusted[self.clust_test_preds == i] = glm.predict(clust_i_df_test, which="mean", exposure=np.asarray(clust_i_df_test[self.glm_exposure])).values
                        elif self.glm_offset is not None: # If there is an offset then fitted equals predictions * offset
                            glm_test_pred_mean_offset_adjusted[self.clust_test_preds == i] = glm.predict(clust_i_df_test, which="mean", offset=np.asarray(clust_i_df_test[self.glm_offset])).values
        if self.trials: # This is only true for Binomial regression
            glm_train_pred_mean *= df_train['trials'].values ##### Verify
            glm_train_pred_linear = self.glm_family.predict(glm_train_pred_mean)
        self.glm_train_pred_mean = glm_train_pred_mean # Store as attribute
        if weighted:
            glm_train_pred_linear = self.glm_family.predict(glm_train_pred_mean)  ##### Verify
            # set opt cutoff
            if self.problem_type=="classification":
                if use_opt_cutoff:
                    opt_cutoff = self.optimal_cutoff_point(df_train[self.target_variable], glm_train_pred_mean)
                else:
                    opt_cutoff = 0.5
                for j in range(self.n_components):
                    setattr(self, 'clust' + str(j) + '_glm_cutoff', opt_cutoff.copy()) # Set optimal cutoff as attribute
                glm_train_pred_class = (glm_train_pred_mean >= opt_cutoff) * 1
        self.glm_train_pred_linear = glm_train_pred_linear
        if self.problem_type=="classification":
            self.glm_train_pred_class = glm_train_pred_class
        if self.glm_exposure is not None and self.glm_offset is not None:
            if self.trials: # This is only true for Binomial regression
                glm_train_pred_mean_exposure_offset_adjusted *= df_train['trials'].values
            self.glm_train_pred_mean_exposure_offset_adjusted = glm_train_pred_mean_exposure_offset_adjusted
        elif self.glm_exposure is not None: 
            if self.trials: # This is only true for Binomial regression
                glm_train_pred_mean_exposure_adjusted *= df_train['trials'].values
            self.glm_train_pred_mean_exposure_adjusted = glm_train_pred_mean_exposure_adjusted # Store as attribute
        elif self.glm_offset is not None:
            if self.trials: # This is only true for Binomial regression
                glm_train_pred_mean_offset_adjusted *= df_train['trials'].values
            self.glm_train_pred_mean_offset_adjusted = glm_train_pred_mean_offset_adjusted # Store as attribute
        if self.valid_set:
            if self.trials: # This is only true for Binomial regression
                glm_valid_pred_mean *= df_valid['trials'].values
                glm_valid_pred_linear = self.glm_family.predict(glm_valid_pred_mean)
            self.glm_valid_pred_mean = glm_valid_pred_mean # Store as attribute
            if weighted:
                glm_valid_pred_linear = self.glm_family.predict(glm_valid_pred_mean)
                if self.problem_type=="classification":
                    if use_opt_cutoff:
                        opt_cutoff = np.zeros((df_valid.shape[0], ))
                        for j in range(self.n_components):
                            opt_cutoff += self.clust_valid_proba[:, j] * getattr(self, 'clust' + str(j) + '_glm_cutoff')
                        glm_valid_pred_class = (glm_valid_pred_mean >= opt_cutoff) * 1
                    else:
                        glm_valid_pred_class = (glm_valid_pred_mean >= 0.5) * 1 
            self.glm_valid_pred_linear = glm_valid_pred_linear # Store as attribute
            if self.problem_type=="classification":
                self.glm_valid_pred_class = glm_valid_pred_class
            if self.glm_exposure is not None and self.glm_offset is not None:
                if self.trials:# This is only true for Binomial regression
                    glm_valid_pred_mean_exposure_offset_adjusted *= df_valid['trials'].values
                self.glm_valid_pred_mean_exposure_offset_adjusted = glm_valid_pred_mean_exposure_offset_adjusted
            elif self.glm_exposure is not None: 
                if self.trials:# This is only true for Binomial regression
                    glm_valid_pred_mean_exposure_adjusted *= df_valid['trials'].values
                self.glm_valid_pred_mean_exposure_adjusted = glm_valid_pred_mean_exposure_adjusted # Store as attribute
            elif self.glm_offset is not None:
                if self.trials:# This is only true for Binomial regression
                    glm_valid_pred_mean_offset_adjusted *= df_valid['trials'].values
                self.glm_valid_pred_mean_offset_adjusted = glm_valid_pred_mean_offset_adjusted # Store as attribute
        if self.test_set:
            if self.trials:# This is only true for Binomial regression
                glm_test_pred_mean *= df_test['trials'].values
                glm_test_pred_linear = self.glm_family.predict(glm_test_pred_mean)
            self.glm_test_pred_mean = glm_test_pred_mean # Store as attribute
            if weighted:
                self.glm_test_pred_linear = self.glm_family.predict(glm_test_pred_mean)
                if self.problem_type=="classification":
                    if use_opt_cutoff:
                        opt_cutoff = np.zeros((df_test.shape[0], ))
                        for j in range(self.n_components):
                            opt_cutoff += self.clust_test_proba[:, j] * getattr(self, 'clust' + str(j) + '_glm_cutoff')
                        glm_test_pred_class = (glm_test_pred_mean >= opt_cutoff) * 1
                    else:
                        glm_test_pred_class = (glm_test_pred_mean >= 0.5) * 1 
            self.glm_test_pred_linear = glm_test_pred_linear # Store as attribute
            if self.problem_type=="classification":
                self.glm_test_pred_class = glm_test_pred_class # Store as attribute
            if self.glm_exposure is not None and self.glm_offset is not None:
                if self.trials:# This is only true for Binomial regression
                    glm_test_pred_mean_exposure_offset_adjusted *= df_test['trials'].values
                self.glm_test_pred_mean_exposure_offset_adjusted = glm_test_pred_mean_exposure_offset_adjusted
            elif self.glm_exposure is not None: 
                if self.trials:# This is only true for Binomial regression
                    glm_test_pred_mean_exposure_adjusted *= df_test['trials'].values
                self.glm_test_pred_mean_exposure_adjusted = glm_test_pred_mean_exposure_adjusted # Store as attribute
            elif self.glm_offset is not None:
                if self.trials:# This is only true for Binomial regression
                    glm_test_pred_mean_offset_adjusted *= df_test['trials'].values
                self.glm_test_pred_mean_offset_adjusted = glm_test_pred_mean_offset_adjusted # Store as attribute
        self.print_both("GLM fit.", log_level="info")

    def predict_gmm(self, df, prob=False, return_both=False, gmm_scaling=True):
        """
        Parameters
        ----------
        df : DataFrame
        prob : Should the probabilities be returned (bool, default=False)
        return_both : Should both predictions and probabilities be returned (bool, default=False)
        gmm_scaling : Should the data be scaled before prediction (bool, default=True)
        """
        df_copy = df.copy()
        if gmm_scaling: # Scaling
            df_copy[self.continuous_variables + self.discrete_variables] = self.scale_data(df_copy)
        if self.use_pca: # Add PCA variables to dataframe
            df_copy[self.pca_variables] = self.predict_pca(df_copy, scaling=False) # scaling would already be done if true
        if return_both: # Return predictions
            return self.gmm.predict_proba(df_copy[self.cluster_variables]), self.gmm.predict(df_copy[self.cluster_variables])
        elif prob:
            return self.gmm.predict_proba(df_copy[self.cluster_variables])
        else:
            return self.gmm.predict(df_copy[self.cluster_variables])

    def predict_glm(self, df, which="mean", return_all=False, glm_scaling=True, weighted=False):
        """Predict using GLM.

        Parameters
        ----------
        df : DataFrame
        which : "mean", "linear", "class", "offset_adjusted", "exposure_adjusted", "exposure_offset_adjusted" (str, default="mean")
        return_all : Should mean predictions, linear predictions, class predictions and offset adjusted prediction be returned (bool, default=False)
        glm_scaling : Should the data be scaled before prediction (bool, default=True)
        weighted : Should the predictions be weighted by cluster probabilities. Can only be used if linear is false. (bool, default=False)
        
        Notes
        -----
        We can extend the multiclass setting by calculating class.
        This requires a trials column in df if the problem is Binomial regression
        
        Review if the weighted matters. Maybe scaling happened twice?
        Should sampling be included?
        """
        df_copy = df.copy()
        df_copy.reset_index(drop=True, inplace=True)
        if glm_scaling == True:
            df_copy[self.continuous_variables + self.discrete_variables] = self.scale_data(df_copy)
        if self.use_pca:
            df_copy[self.pca_variables] = self.predict_pca(df_copy, scaling=False) # scaling would already be done if true
        if return_all: # return mean, linear, class, exposure_offset_adjusted, exposure_adjusted, offset_adjusted
            glm_pred_mean = np.zeros((df_copy.shape[0], )) # np.array for regression predictions
            glm_pred_linear = np.zeros((df_copy.shape[0], )) # np.array for regression predictions
            if self.problem_type=="classification":
                glm_pred_class = np.zeros((df_copy.shape[0], ))
            if self.glm_exposure is not None and self.glm_offset is  not None:
                glm_pred_mean_exposure_offset_adjusted = np.zeros((df_copy.shape[0], ))
            if self.glm_exposure is not None:
                glm_pred_mean_exposure_adjusted = np.zeros((df_copy.shape[0], ))
            if self.glm_offset is not None: # If there is an offset then fitted equals predictions * offset
                glm_pred_mean_offset_adjusted = np.zeros((df_copy.shape[0], ))
        elif which=="mean":
            glm_pred_mean = np.zeros((df_copy.shape[0], ))
        elif which=="linear":   
            glm_pred_linear = np.zeros((df_copy.shape[0], ))
            if weighted: # weighted linear requires mean
                glm_pred_mean = np.zeros((df_copy.shape[0], ))
        elif which=="class": # class required mean
            if self.problem_type=="regression":
                return(print("Class prediction can only be used for binary classifiers."))
            else:
                glm_pred_mean = np.zeros((df_copy.shape[0], ))
                glm_pred_class = np.zeros((df_copy.shape[0], ))
        elif which=="exposure_offset_adjusted":
            if self.glm_exposure is None and self.glm_offset is None:
                return(print("No exposure or offset variable in model."))
            elif self.glm_exposure is None:
                return(print("No exposure variable in model."))
            elif self.glm_offset is None:
                return(print("No offset variable in model."))
            else:
                glm_pred_mean_exposure_offset_adjusted = np.zeros((df_copy.shape[0], ))
        elif which=="exposure_adjusted":
            if self.glm_exposure is None:
                return(print("No exposure variable in model."))
            else:
                glm_pred_mean_exposure_adjusted = np.zeros((df_copy.shape[0], ))
        elif which=="offset_adjusted":
            if self.glm_offset is None:
                return(print("No offset variable in model."))
            else:
                glm_pred_mean_offset_adjusted = np.zeros((df_copy.shape[0], ))
        if weighted:
            if self.gmm_scaling:
                clust_probs = self.predict_gmm(df_copy, prob=True, return_both=False, gmm_scaling=False)
            else:
                unscaled_df_copy = df_copy.copy()
                unscaled_df_copy[self.continuous_variables+self.discrete_variables] = self.scale_data(df_copy)
                clust_probs = self.predict_gmm(unscaled_df_copy, prob=True, return_both=False, gmm_scaling=False)
        else:
            if self.gmm_scaling:
                clust_preds = self.predict_gmm(df_copy, prob=False, return_both=False, gmm_scaling=False)  
            else:
                unscaled_df_copy = df_copy.copy()
                unscaled_df_copy[self.continuous_variables+self.discrete_variables] = self.scale_data(df_copy)
                clust_preds = self.predict_gmm(unscaled_df_copy, prob=False, return_both=False, gmm_scaling=False)
        for i in range(self.n_components):
            glm = getattr(self, 'clust' + str(i) + '_glm') # Cluster i GLM
            # Predict
            if weighted:
                if return_all or which=="mean" or which=="linear" or which=="class":
                    glm_pred_mean += clust_probs[:, i] * glm.predict(df_copy).values
                    if self.glm_exposure is not None and self.glm_offset is not None:
                        glm_pred_mean_exposure_offset_adjusted += clust_probs[:, i] * glm.predict(df_copy, which="mean", exposure=np.asarray(df[self.glm_exposure]), offset=np.asarray(df[self.glm_offset])).values
                    if self.glm_exposure is not None:
                        glm_pred_mean_exposure_adjusted += clust_probs[:, i] * glm.predict(df_copy, which="mean", exposure=np.asarray(df[self.glm_exposure])).values
                    if self.glm_offset is not None:
                        glm_pred_mean_offset_adjusted += clust_probs[:, i] * glm.predict(df_copy, which="mean", offset=np.asarray(df[self.glm_offset])).values
                elif which=="exposure_offset_adjusted":
                    glm_pred_mean_exposure_offset_adjusted += clust_probs[:, i] * glm.predict(df_copy, which="mean", exposure=np.asarray(df[self.glm_exposure]), offset=np.asarray(df[self.glm_offset])).values
                elif which=="exposure_adjusted":
                    glm_pred_mean_exposure_adjusted += clust_probs[:, i] * glm.predict(df_copy, which="mean", exposure=np.asarray(df[self.glm_exposure])).values
                elif which=="offset_adjusted":
                    glm_pred_mean_offset_adjusted += clust_probs[:, i] * glm.predict(df_copy, which="mean", offset=np.asarray(df[self.glm_offset])).values
                # if it is linear do it after this loop
            else:
                if return_all:
                    glm_pred_linear[clust_preds == i] = glm.predict(df_copy.loc[clust_preds == i], which="linear").values
                    glm_pred_mean[clust_preds == i] = glm.predict(df_copy.loc[clust_preds == i], which="mean").values
                    if self.problem_type=="classification":
                        opt_cutoff = getattr(self,'clust' + str(i) + '_glm_cutoff')
                        glm_pred_class[clust_preds == i] = (glm_pred_mean[clust_preds == i] >= opt_cutoff) * 1
                    if self.glm_exposure is not None and self.glm_offset is not None:
                        glm_pred_mean_exposure_offset_adjusted[clust_preds == i] = glm.predict(df_copy.loc[clust_preds == i], which="mean", exposure=np.asarray(df.loc[clust_preds == i, self.glm_exposure]), offset=np.asarray(df.loc[clust_preds == i, self.glm_offset])).values
                    if self.glm_exposure is not None:
                        glm_pred_mean_exposure_adjusted[clust_preds == i] = glm.predict(df_copy.loc[clust_preds == i], which="mean", exposure=np.asarray(df.loc[clust_preds == i, self.glm_exposure])).values
                    if self.glm_offset is not None: # df_copy could be scaled so use original df
                        glm_pred_mean_offset_adjusted[clust_preds == i] = glm.predict(df_copy.loc[clust_preds == i], which="mean", offset=np.asarray(df.loc[clust_preds == i, self.glm_offset])).values
                elif which=="mean":
                    glm_pred_mean[clust_preds == i] = glm.predict(df_copy.loc[clust_preds == i], which="mean").values
                elif which=="linear":
                    glm_pred_linear[clust_preds == i] = glm.predict(df_copy.loc[clust_preds == i], which="linear").values
                elif which=="class":
                    opt_cutoff = getattr(self,'clust' + str(i) + '_glm_cutoff')
                    glm_pred_class[clust_preds == i] = (glm.predict(df_copy.loc[clust_preds == i]) >= opt_cutoff) * 1
                elif which=="exposure_offset_adjusted":
                    glm_pred_mean_exposure_offset_adjusted[clust_preds == i] = glm.predict(df_copy.loc[clust_preds == i], which="mean", exposure=np.asarray(df.loc[clust_preds == i, self.glm_exposure]), offset=np.asarray(df.loc[clust_preds == i, self.glm_offset])).values
                elif which=="exposure_adjusted":
                    glm_pred_mean_exposure_adjusted[clust_preds == i] = glm.predict(df_copy.loc[clust_preds == i], which="mean", exposure=np.asarray(df.loc[clust_preds == i, self.glm_exposure])).values
                elif which=="offset_adjusted": # df_copy could be scaled so use original df
                    glm_pred_mean_offset_adjusted[clust_preds == i] = glm.predict(df_copy.loc[clust_preds == i], which="mean", offset=np.asarray(df.loc[clust_preds == i, self.glm_offset])).values
        # After looping over clusters
        # Weighted
        if weighted:
            if self.trials: # This only happens if the problem is Binomial regression
                glm_pred_mean *= df_copy['trials']
            glm_pred_linear = self.glm_family.predict(glm_pred_mean)
            if (return_all and self.problem_type=="classification") or which=="class":
                opt_cutoff = np.zeros((df_copy.shape[0], )) #opt_cutoff will be a vector
                for i in range(self.n_components):
                    opt_cutoff += clust_probs[:, i] * getattr(self, 'clust' + str(i) + '_glm_cutoff')
                glm_pred_class = (glm_pred_mean >= opt_cutoff) * 1
        # return
        if return_all:
            if self.trials and not weighted: # This only happens if the problem is Binomial regression
                glm_pred_mean *= df_copy['trials']
                glm_pred_linear = self.glm_family.predict(glm_pred_mean)
            dict_return = {"mean": glm_pred_mean, "linear": glm_pred_linear}
            if self.problem_type=="classification":
                dict_return["class"] = glm_pred_class
            if self.glm_exposure is not None and self.glm_offset is not None:
                if self.trials: # This only happens if the problem is Binomial regression
                    glm_pred_mean_exposure_offset_adjusted *= df_copy['trials']
                    glm_pred_mean_exposure_adjusted *= df_copy['trials']
                    glm_pred_mean_offset_adjusted *= df_copy['trials']
                dict_return["exposure_offset_adjusted"] = glm_pred_mean_exposure_offset_adjusted
                dict_return["exposure_adjusted"] = glm_pred_mean_exposure_adjusted
                dict_return["offset_adjusted"] = glm_pred_mean_offset_adjusted
            elif self.glm_exposure is not None:
                if self.trials:
                    glm_pred_mean_exposure_adjusted *= df_copy['trials']
                dict_return["exposure_adjusted"] = glm_pred_mean_exposure_adjusted
            elif self.glm_offset is not None:
                if self.trials:
                    glm_pred_mean_offset_adjusted *= df_copy['trials']
                dict_return["offset_adjusted"] = glm_pred_mean_offset_adjusted
            return dict_return
        elif which=="mean":
            if self.trials and not weighted:
                glm_pred_mean *= df_copy['trials']
            return glm_pred_mean
        elif which=="linear":
            if self.trials and not weighted:
                glm_pred_mean *= df_copy['trials']
                glm_pred_linear = self.glm_family.predict(glm_pred_mean)
            return glm_pred_linear
        elif which=="class":
            return glm_pred_class
        elif which=="exposure_offset_adjusted":
            if self.trials:
                glm_pred_mean_exposure_offset_adjusted *= df_copy['trials']
            return glm_pred_mean_exposure_offset_adjusted
        elif which=="exposure_adjusted":
            if self.trials:
                glm_pred_mean_exposure_adjusted *= df_copy['trials']
            return glm_pred_mean_exposure_adjusted
        elif which=="offset_adjusted":
            if self.trials:
                glm_pred_mean_offset_adjusted *= df_copy['trials']
            return glm_pred_mean_offset_adjusted
        
    def get_gmm_summary(self,verbose=True, gmm_scaling=True, return_results=True):
        """Returns the BIC for the training, validation and test set set
        Parameters
        ----------
        verbose : bool
            Print results. Default is True.
        gmm_scaling : bool
            Scale data before fitting. Default is True.
        return_results : bool
            Return results. Default is True.

        Returns
        -------
        tuple
            BIC for training, validation and test set

        New Attributes
        --------------
        gmm_means : DataFrame
        """
        gmm_means = pd.DataFrame(columns=self.cluster_variables)
        for i in range(self.n_components):
            gmm_means.loc[i] = list(
                (self.mu[self.cluster_variables] +
                 self.sigma[self.cluster_variables] * self.gmm.means_[i]).values)
        self.gmm_means = gmm_means
        df_train = self.df_train.copy()
        if self.valid_set:
            df_valid = self.df_valid.copy()
        if self.test_set:
            df_test = self.df_test.copy()
        if gmm_scaling:
            df_train[self.continuous_variables+self.discrete_variables] = self.scale_data(df_train)
            if self.valid_set:
                df_valid[self.continuous_variables+self.discrete_variables] = self.scale_data(df_valid)
            if self.test_set:
                df_test[self.continuous_variables+self.discrete_variables] = self.scale_data(df_test)
        if verbose:
            print("\nTraining set BIC:\n{}".format(self.gmm.bic(df_train[self.cluster_variables]))) 
            if self.valid_set:
                print("\nValidation set BIC:\n{}".format(self.gmm.bic(df_valid[self.cluster_variables])))
            if self.test_set:
                print("\nTest set BIC:\n{}".format(self.gmm.bic(df_test[self.cluster_variables])))
        if return_results:
            if self.valid_set and self.test_set:
                return self.gmm.bic(df_train[self.cluster_variables]), self.gmm.bic(df_valid[self.cluster_variables]), self.gmm.bic(df_test[self.cluster_variables])
            elif self.valid_set:
                return self.gmm.bic(df_train[self.cluster_variables]), self.gmm.bic(df_valid[self.cluster_variables])
            elif self.test_set:
                return self.gmm.bic(df_train[self.cluster_variables]), self.gmm.bic(df_test[self.cluster_variables])
            else:
                self.gmm.bic(df_train[self.cluster_variables])

    def get_glm_summary(self):
        """Prints the summary of the GLM for each cluster.
        """
        for i in range(self.n_components):
            print("Cluster {}:".format(str(i)))
            glm = getattr(self, 'clust' + str(i) + '_glm')
            print(glm.summary())

    def get_gmm_variable_plots(self, verbose=True, savefig=False, figloc=None):
        """Plots the distribution of the cluster variables by cluster.

        Parameters
        ----------
        verbose : bool
            Print results. Default is True.
        savefig : bool
            Save figure. Default is False.
        figloc : str
            Location to save figure. Default is None.
        """
        df_train = self.df_train[self.cluster_variables].copy()
        df_train['cluster'] = self.clust_train_preds
        for variable in self.cluster_variables:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            sns.histplot(data=df_train,
                         x=variable,
                         stat='percent',
                         hue='cluster',
                         common_norm=False,
                         palette='colorblind',
                         kde=True,
                         ax=axs[0])
            sns.boxplot(ax=axs[1], data=df_train, x='cluster', y=variable)
            if savefig:
                plt.savefig(fname=figloc + 'train_gmm_' + str(variable) + '_plot')
            if verbose:
                print("\nTraining Set")
                plt.show()
            else:
                plt.close(fig)
        if self.valid_set:
            df_valid = self.df_valid[self.cluster_variables].copy()
            df_valid['cluster'] = self.clust_valid_preds
            for variable in self.cluster_variables:
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                sns.histplot(data=df_valid,
                            x=variable,
                            stat='percent',
                            hue='cluster',
                            common_norm=False,
                            palette='colorblind',
                            kde=True,
                            ax=axs[0])
                sns.boxplot(ax=axs[1], data=df_valid, x='cluster', y=variable)
                if savefig:
                    plt.savefig(fname=figloc + 'valid_gmm_' + str(variable) + '_plot')
                if verbose:
                    print("\nValidation Set")
                    plt.show()
                else:
                    plt.close(fig)
        if self.test_set:
            df_test = self.df_test[self.cluster_variables].copy()
            df_test['cluster'] = self.clust_test_preds
            for variable in self.cluster_variables:
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                sns.histplot(data=df_test,
                            x=variable,
                            stat='percent',
                            hue='cluster',
                            common_norm=False,
                            palette='colorblind',
                            kde=True,
                            ax=axs[0])
                sns.boxplot(ax=axs[1], data=df_test, x='cluster', y=variable)
                if savefig:
                    plt.savefig(fname=figloc + 'test_gmm_' + str(variable) + '_plot')
                if verbose:
                    print("\nTest Set")
                    plt.show()
                else:
                    plt.close(fig)

    def get_gmm_silhouette_plot(self, df=None, cluster_labels=None, gmm_scaling=True, verbose=True, savefig=False, figloc=None):
        """ Silhouette plot for GMM
        """
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        if df is None:
            df = self.df_train.copy()
            if gmm_scaling:
                df[self.continuous_variables+self.discrete_variables] = self.scale_data(df)
            cluster_labels = self.clust_train_preds
        else:
            if gmm_scaling:
                df[self.continuous_variables+self.discrete_variables] = self.scale_data(df)
        axs.set_xlim([-1, 1])# The silhouette coefficient can range from -1, 1 
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        axs.set_ylim([0, len(df) + (self.n_components + 1) * 10])
        silhouette_avg = metrics.silhouette_score(df[self.cluster_variables], cluster_labels) # The silhouette_score gives the average value for all the samples.
        print(
            "For n_clusters =",
            self.n_components,
            "The average silhouette_score is :",
            silhouette_avg,
        )
        # Compute the silhouette scores for each sample
        sample_silhouette_values = metrics.silhouette_samples(df[self.cluster_variables], cluster_labels)
        y_lower = 10
        for i in range(self.n_components):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / self.n_components)
            axs.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            # Label the silhouette plots with their cluster numbers at the middle
            axs.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
        axs.set_title("The silhouette plot for the various clusters.")
        axs.set_xlabel("The silhouette coefficient values")
        axs.set_ylabel("Cluster label")
        axs.axvline(x=silhouette_avg, color="red", linestyle="--") # The vertical line for average silhouette score of all the values
        axs.set_yticks([])  # Clear the yaxis labels / ticks
        axs.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
        if savefig:
                plt.savefig(fname=figloc + 'silhuouette_plot')
        if verbose:
            plt.show()
        else:
            plt.close(fig)

    def get_gmm_claims_plots(self, verbose=True, savefig=False, figloc=None):
        """ Stacked bar chart of claims by cluster with pie chart of cluster membership

        Parameters
        ----------
        verbose : bool
            Print results. Default is True.
        savefig : bool
            Save figure. Default is False.
        figloc : str
            Location to save figure. Default is None.
        """
        #### TO DO: Adapt for regression
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        # Stacked Bar Chart ---------
        axs[0].bar(x=np.arange(self.n_components),
                   height=self.n_components * [100],
                   color="blue",
                   label="No Claim")
        height = 100 * (sum(self.df_train.loc[self.clust_train_preds == 0, self.target_variable[0]]) / sum(self.clust_train_preds == 0))
        axs[0].bar(x=[0], height=[height], color='red', label="Claim")
        axs[0].text(0,
                    height,
                    f'{height:.2f}%',
                    ha='center',
                    va='bottom',
                    color="red")
        for i in range(1, self.n_components):
            height = 100 * (sum(self.df_train.loc[self.clust_train_preds == i, self.target_variable[0]]) / sum(self.clust_train_preds == i))
            axs[0].bar(x=[i], height=[height], color='red')
            axs[0].text(i,
                        height,
                        f'{height:.2f}%',
                        ha='center',
                        va='bottom',
                        color="red")
        axs[0].set_xticks(np.arange(self.n_components))
        axs[0].set_ylabel("%")
        axs[0].set_xlabel("Cluster")
        axs[0].set_title("Percentage of Claims by Cluster")
        axs[0].legend(bbox_to_anchor=(1, 1))
        # Pie chart ---------
        (unique_clusters, clusters_counts) = np.unique(self.clust_train_preds, return_counts=True) # Get unique clusters
        _, _, autotexts = axs[1].pie(x=clusters_counts,
                                     labels=unique_clusters,
                                     autopct="%.2f%%",
                                     colors=sns.color_palette("colorblind"))
        for autotext in autotexts:
            autotext.set_color('white')
        axs[1].axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
        axs[1].set_title("Pie Chart for Cluster Proportion")
        if savefig:
            plt.savefig(fname=figloc + 'train_gmm_claims_plots')
        if verbose:
            print("\nTraining set")
            plt.show()
        else:
            plt.close(fig)
        if self.valid_set:
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            # Stacked Bar Chart ---------
            axs[0].bar(x=np.arange(self.n_components),
                    height=self.n_components * [100],
                    color="blue",
                    label="No Claim")
            height = 100 * (sum(self.df_valid.loc[self.clust_valid_preds == 0, self.target_variable[0]]) /sum(self.clust_valid_preds == 0))
            axs[0].bar(x=[0], height=[height], color='red', label="Claim")
            axs[0].text(0,
                        height,
                        f'{height:.2f}%',
                        ha='center',
                        va='bottom',
                        color="red")
            for i in range(1, self.n_components):
                height = 100 * (sum(self.df_valid.loc[self.clust_valid_preds == i, self.target_variable[0]]) / sum(self.clust_valid_preds == i))
                axs[0].bar(x=[i], height=[height], color='red')
                axs[0].text(i,
                            height,
                            f'{height:.2f}%',
                            ha='center',
                            va='bottom',
                            color="red")
            axs[0].set_xticks(np.arange(self.n_components))
            axs[0].set_ylabel("%")
            axs[0].set_xlabel("Cluster")
            axs[0].set_title("Percentage of Claims by Cluster")
            axs[0].legend(bbox_to_anchor=(1, 1))
            # Pie chart ---------
            (unique_clusters, clusters_counts) = np.unique(self.clust_valid_preds, return_counts=True) # Get unique clusters
            _, _, autotexts = axs[1].pie(x=clusters_counts,
                                        labels=unique_clusters,
                                        autopct="%.2f%%",
                                        colors=sns.color_palette("colorblind"))
            for autotext in autotexts:
                autotext.set_color('white')
            axs[1].axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
            axs[1].set_title("Pie Chart for Cluster Proportion")
            if savefig:
                plt.savefig(fname=figloc + 'valid_gmm_claims_plots')
            if verbose:
                print("\nValidation set")
                plt.show()
            else:
                plt.close(fig)
        if self.test_set:
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            # Stacked Bar Chart ---------
            axs[0].bar(x=np.arange(self.n_components),
                    height=self.n_components * [100],
                    color="blue",
                    label="No Claim")
            height = 100 * (sum(self.df_test.loc[self.clust_test_preds == 0, self.target_variable[0]]) /sum(self.clust_test_preds == 0))
            axs[0].bar(x=[0], height=[height], color='red', label="Claim")
            axs[0].text(0,
                        height,
                        f'{height:.2f}%',
                        ha='center',
                        va='bottom',
                        color="red")
            for i in range(1, self.n_components):
                height = 100 * (sum(self.df_test.loc[self.clust_test_preds == i, self.target_variable[0]]) / sum(self.clust_test_preds == i))
                axs[0].bar(x=[i], height=[height], color='red')
                axs[0].text(i,
                            height,
                            f'{height:.2f}%',
                            ha='center',
                            va='bottom',
                            color="red")
            axs[0].set_xticks(np.arange(self.n_components))
            axs[0].set_ylabel("%")
            axs[0].set_xlabel("Cluster")
            axs[0].set_title("Percentage of Claims by Cluster")
            axs[0].legend(bbox_to_anchor=(1, 1))
            # Pie chart ---------
            (unique_clusters, clusters_counts) = np.unique(self.clust_test_preds, return_counts=True) # Get unique clusters
            _, _, autotexts = axs[1].pie(x=clusters_counts,
                                        labels=unique_clusters,
                                        autopct="%.2f%%",
                                        colors=sns.color_palette("colorblind"))
            for autotext in autotexts:
                autotext.set_color('white')
            axs[1].axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
            axs[1].set_title("Pie Chart for Cluster Proportion")
            if savefig:
                plt.savefig(fname=figloc + 'test_gmm_claims_plots')
            if verbose:
                print("\nTest set")
                plt.show()
            else:
                plt.close(fig)

    def get_glm_curve_plots(self,
                            verbose=True,
                            savefig=False,
                            figloc=None,
                            kwargs_for_roc={},
                            kwargs_for_pr={}):
        """
        Parameters
        ----------
        verbose : bool
            Print results. Default is True.
        savefig : bool
            Save figure. Default is False.
        figloc : str
            Location to save figure. Default is None.
        kwargs_for_roc : dict
            Keyword arguments for ROC curve. Default is {}.
        kwargs_for_pr : dict
            Keyword arguments for Precision-Recall curve. Default is {}.

        Notes
        -----
        We can extend this to the multiclass setting by calculating Prob(X=0).
        """
        if self.problem_type=="classification":
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
            # ROC plot
            if self.glm_exposure is not None and self.glm_offset is not None:
                y_pred_train = self.glm_train_pred_mean_exposure_offset_adjusted
            elif self.glm_exposure is not None:
                y_pred_train = self.glm_train_pred_mean_exposure_adjusted
            elif self.glm_offset is not None:
                y_pred_train = self.glm_train_pred_mean_offset_adjusted
            else:
                y_pred_train = self.glm_train_pred_mean
            metrics.RocCurveDisplay.from_predictions(y_true=self.df_train[self.target_variable[0]],
                                                y_pred=y_pred_train,
                                                name="GLM",
                                                color="darkorange",
                                                ax=axs[0],
                                                **kwargs_for_roc)
            axs[0].plot([0, 1], [0, 1],"k--", label="chance level (AUC = 0.5)")  # Chance line
            axs[0].axis("square")
            axs[0].set_xlabel("False Positive Rate")
            axs[0].set_ylabel("True Positive Rate")
            axs[0].set_title("GLM ROC (Training Set)")
            axs[0].legend(loc='lower right')
            # PrecisionRecall plot
            metrics.PrecisionRecallDisplay.from_predictions(
                y_true=self.df_train[self.target_variable[0]],
                y_pred=y_pred_train,
                name="GLM Precision-Recall Curve",
                color="darkorange",
                ax=axs[1],
                **kwargs_for_pr)
            axs[1].axis("square")
            axs[1].set_xlabel("Recall")
            axs[1].set_ylabel("Precision")
            axs[1].set_title("GLM PRC (Training Set)")
            axs[1].legend(loc='upper right')
            if savefig:
                plt.savefig(fname=figloc + 'train_roc_pr_curve_plots')
            if verbose:
                plt.show()
            else:
                plt.close(fig)
            if self.valid_set:
                if self.glm_exposure is not None and self.glm_offset is not None:
                    y_pred_valid = self.glm_valid_pred_mean_exposure_offset_adjusted
                elif self.glm_exposure is not None:
                    y_pred_valid = self.glm_valid_pred_mean_exposure_adjusted
                elif self.glm_offset is not None:
                    y_pred_valid = self.glm_valid_pred_mean_offset_adjusted
                else:
                    y_pred_valid = self.glm_valid_pred_mean
                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
                metrics.RocCurveDisplay.from_predictions(y_true=self.df_valid[self.target_variable[0]],
                                                    y_pred=y_pred_valid,
                                                    name="GLM",
                                                    color="darkorange",
                                                    ax=axs[0],
                                                    **kwargs_for_roc)
                axs[0].plot([0, 1], [0, 1],"k--", label="chance level (AUC = 0.5)")  # Chance line
                axs[0].axis("square")
                axs[0].set_xlabel("False Positive Rate")
                axs[0].set_ylabel("True Positive Rate")
                axs[0].set_title("GLM ROC (Validation Set)")
                axs[0].legend(loc='lower right')
                # PrecisionRecall plot
                metrics.PrecisionRecallDisplay.from_predictions(
                    y_true=self.df_valid[self.target_variable[0]],
                    y_pred=y_pred_valid,
                    name="GLM Precision-Recall Curve",
                    color="darkorange",
                    ax=axs[1],
                    **kwargs_for_pr)
                axs[1].axis("square")
                axs[1].set_xlabel("Recall")
                axs[1].set_ylabel("Precision")
                axs[1].set_title("GLM PRC (Validation Set)")
                axs[1].legend(loc='upper right')
                if savefig:
                    plt.savefig(fname=figloc + 'valid_roc_pr_curve_plots')
                if verbose:
                    plt.show()
                else:
                    plt.close(fig)
            if self.test_set:
                if self.glm_exposure is not None and self.glm_offset is not None:
                    y_pred_test = self.glm_test_pred_mean_exposure_offset_adjusted
                elif self.glm_exposure is not None:
                    y_pred_test = self.glm_test_pred_mean_exposure_adjusted
                elif self.glm_offset is not None:
                    y_pred_test = self.glm_test_pred_mean_offset_adjusted
                else:
                    y_pred_test = self.glm_test_pred_mean
                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
                metrics.RocCurveDisplay.from_predictions(y_true=self.df_test[self.target_variable[0]],
                                                    y_pred=y_pred_test,
                                                    name="GLM",
                                                    color="darkorange",
                                                    ax=axs[0],
                                                    **kwargs_for_roc)
                axs[0].plot([0, 1], [0, 1],"k--", label="chance level (AUC = 0.5)")  # Chance line
                axs[0].axis("square")
                axs[0].set_xlabel("False Positive Rate")
                axs[0].set_ylabel("True Positive Rate")
                axs[0].set_title("GLM ROC (Test Set)")
                axs[0].legend(loc='lower right')
                # PrecisionRecall plot
                metrics.PrecisionRecallDisplay.from_predictions(
                    y_true=self.df_test[self.target_variable[0]],
                    y_pred=y_pred_test,
                    name="GLM Precision-Recall Curve",
                    color="darkorange",
                    ax=axs[1],
                    **kwargs_for_pr)
                axs[1].axis("square")
                axs[1].set_xlabel("Recall")
                axs[1].set_ylabel("Precision")
                axs[1].set_title("GLM PRC (Test Set)")
                axs[1].legend(loc='upper right')
                if savefig:
                    plt.savefig(fname=figloc + 'test_roc_pr_curve_plots')
                if verbose:
                    plt.show()
                else:
                    plt.close(fig)
        else:
            print("Incorrect GLM Family specified.")
            sys.exit()

    def probability_plot(self, 
                         df,
                         verbose=True,
                         savefig=False,
                         figloc=None, 
                         figloc_name=None,
                         set="Training Set"):
        """ This plots Probabilities by cluster with claim/no claim coloured differently
        """
        fig, ax = plt.subplots(figsize=(20, 5))
        sns.stripplot(x=df.columns[0],
                    y="Cluster",
                    data=df,
                    hue=df.columns[1],
                    dodge=True,
                    orient="h",
                    alpha=0.2)
        min_x = min(self.df[self.target_variable[0]])
        max_x = max(self.df[self.target_variable[0]])
        ax.set_xticks(np.arange(min_x, max_x, (max_x-min_x)/10))
        ax.set_yticks(np.arange(0, self.n_components, 1))
        if savefig:
            plt.savefig(fname=figloc+figloc_name)
        if verbose:
            print("\n"+set)
            plt.show()
        else:
            plt.close(fig)

    def histogram_plot(self, 
                       df,
                       verbose=True,
                       savefig=False,
                       figloc=None, 
                       figloc_name=None):
        """ Histogram of Probabilities for claims and no claims
        """
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.histplot(ax=ax, data=df, x=df.columns[0], kde=True, hue=self.target_variable[0])
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Count")
        fig.tight_layout()
        if savefig:
            plt.savefig(fname=figloc+figloc_name)
        if verbose:
            plt.show()
        else:
            plt.close(fig)
    
    def calibaration_plot(self,
                          df,
                          verbose=True,
                          savefig=False,
                          figloc=None, 
                          figloc_name=None):
        """ Calibration curve"""
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(df['Bins'], df['ObsProb'], label='GLM')
        ax.errorbar(df['Bins'],
                    df['ObsProb'],
                    yerr=df['SE'],
                    fmt='none',
                    label='95% Wald interval')
        ax.plot([0, 1], [0, 1],
                linestyle='--',
                label='Perfectly Calibrated',
                color='orange')
        plt.title('')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Observed Proportion')
        plt.legend(loc='upper left')
        fig.tight_layout()
        if savefig:
            plt.savefig(fname=figloc+figloc_name)
        if verbose:
            plt.show()
        else:
            plt.close(fig)


    def get_glm_histogram_plots(self,
                                verbose=True,
                                savefig=False,
                                figloc=None):
        """Histogram of probabilities for claims and no claims
        
        Parameters
        ----------
        verbose : bool
            Print results. Default is True.
        savefig : bool
            Save figure. Default is False.
        figloc : str
            Location to save figure. Default is None.

        Notes  
        -----
        We can extend this to the multiclass setting by calculating Prob(X=0).
        For now just binary.
        """
        if self.problem_type=="regression":
            df = pd.DataFrame({"Mean": self.glm_train_pred_mean, self.target_variable[0]: self.df_train[self.target_variable[0]], "Cluster": self.clust_train_preds})
            # Plot 1
            self.probability_plot(df=df, verbose=verbose, savefig=savefig, figloc=figloc, figloc_name='train_glm_claim_plots', set="Training Set")
            # Histogram of Probabilities for claims and no claims
            self.histogram_plot(df=df, verbose=verbose, savefig=savefig, figloc=figloc, figloc_name='train_glm_histogram_plots')
            if self.valid_set:
                df = pd.DataFrame({"Mean": self.glm_valid_pred_mean, self.target_variable[0]: self.df_valid[self.target_variable[0]], "Cluster": self.clust_valid_preds})
                # Plot 1
                self.probability_plot(df=df, verbose=verbose, savefig=savefig, figloc=figloc, figloc_name='valid_glm_claim_plots', set="Valid Set")
                # Histogram of Probabilities for claims and no claims
                self.histogram_plot(df=df, verbose=verbose, savefig=savefig, figloc=figloc, figloc_name='valid_glm_histogram_plots')
            if self.test_set:
                df = pd.DataFrame({"Mean": self.glm_test_pred_mean, self.target_variable[0]: self.df_test[self.target_variable[0]], "Cluster": self.clust_test_preds})
                # Plot 1
                self.probability_plot(df=df, verbose=verbose, savefig=savefig, figloc=figloc, figloc_name='test_glm_claim_plots', set="Test Set")
                # Histogram of Probabilities for claims and no claims
                self.histogram_plot(df=df, verbose=verbose, savefig=savefig, figloc=figloc, figloc_name='test_glm_histogram_plots')
        elif self.trials:
            df = pd.DataFrame({"Mean": self.glm_train_pred_mean, self.target_variable[0]: self.df_train[self.target_variable[0]], "Cluster": self.clust_train_preds})
            # Plot 1
            self.probability_plot(df=df, verbose=verbose, savefig=savefig, figloc=figloc, figloc_name='train_glm_claim_plots', set="Training Set")
            # Histogram of Probabilities for claims and no claims
            self.histogram_plot(df=df, verbose=verbose, savefig=savefig, figloc=figloc, figloc_name='train_glm_histogram_plots')
            if self.valid_set:
                df = pd.DataFrame({"Mean": self.glm_valid_pred_mean, self.target_variable[0]: self.df_valid[self.target_variable[0]], "Cluster": self.clust_valid_preds})
                # Plot 1
                self.probability_plot(df=df, verbose=verbose, savefig=savefig, figloc=figloc, figloc_name='valid_glm_claim_plots', set="Valid Set")
                # Histogram of Probabilities for claims and no claims
                self.histogram_plot(df=df, verbose=verbose, savefig=savefig, figloc=figloc, figloc_name='valid_glm_histogram_plots')
            if self.test_set:
                df = pd.DataFrame({"Mean": self.glm_test_pred_mean, self.target_variable[0]: self.df_test[self.target_variable[0]], "Cluster": self.clust_test_preds})
                # Plot 1
                self.probability_plot(df=df, verbose=verbose, savefig=savefig, figloc=figloc, figloc_name='test_glm_claim_plots', set="Test Set")
                # Histogram of Probabilities for claims and no claims
                self.histogram_plot(df=df, verbose=verbose, savefig=savefig, figloc=figloc, figloc_name='test_glm_histogram_plots')
        else:
            # Plot of Probabilities by cluster with claim/no claim coloured differently
            df = pd.DataFrame({"Prob": self.glm_train_pred_mean, self.target_variable[0]: self.df_train[self.target_variable[0]], "Cluster": self.clust_train_preds})
            # Plot 1
            self.probability_plot(df=df, verbose=verbose, savefig=savefig, figloc=figloc, figloc_name='train_glm_claim_plots', set="Training Set")
            # Histogram of Probabilities for claims and no claims
            self.histogram_plot(df=df, verbose=verbose, savefig=savefig, figloc=figloc, figloc_name='train_glm_histogram_plots')
            # Calibration curve
            rounded_probs = np.round(self.glm_train_pred_mean, 2) # Calibration curve
            claims = self.df_train[self.target_variable[0]].values
            bins = np.arange(0, 101, 1) / 100
            binned_counts = np.zeros(len(bins))
            binned_claims = np.zeros(len(bins))
            for i in range(len(bins)):
                binned_counts[i] = sum(rounded_probs == bins[i])
                binned_claims[i] = sum(claims[rounded_probs == bins[i]])
            calibration_data = pd.DataFrame([bins, binned_counts, binned_claims]).transpose()
            calibration_data.columns = ['Bins', 'N', 'Claims']
            calibration_data['ObsProb'] = calibration_data['Claims'] / calibration_data['N']
            calibration_data['SE'] = norm.ppf(0.975) * np.sqrt( calibration_data['ObsProb'] * (1 - calibration_data['ObsProb']) / calibration_data['N'])
            calibration_data.dropna(inplace=True)
            self.calibaration_plot(df=calibration_data,
                          verbose=verbose,
                          savefig=savefig,
                          figloc=figloc, figloc_name='train_glm_calibration_plots')
            if self.valid_set:
                # Validation data
                df = pd.DataFrame({"Prob": self.glm_valid_pred_mean, self.target_variable[0]: self.df_valid[self.target_variable[0]], "Cluster": self.clust_valid_preds})
                rounded_probs = np.round(self.glm_valid_pred_mean, 2) # Calibration curve
                claims = self.df_valid[self.target_variable[0]].values
                bins = np.arange(0, 101, 1) / 100
                binned_counts = np.zeros(len(bins))
                binned_claims = np.zeros(len(bins))
                for i in range(len(bins)):
                    binned_counts[i] = sum(rounded_probs == bins[i])
                    binned_claims[i] = sum(claims[rounded_probs == bins[i]])
                calibration_data = pd.DataFrame([bins, binned_counts, binned_claims]).transpose()
                calibration_data.columns = ['Bins', 'N', 'Claims']
                calibration_data['ObsProb'] = calibration_data['Claims'] / calibration_data['N']
                calibration_data['SE'] = norm.ppf(0.975) * np.sqrt(calibration_data['ObsProb'] * (1 - calibration_data['ObsProb']) / calibration_data['N'])
                calibration_data.dropna(inplace=True)
                # Plot 1
                self.probability_plot(df=df, verbose=verbose, savefig=savefig, figloc=figloc, figloc_name='valid_glm_claim_plots', set="Validation Set")
                # Histogram of Probabilities for claims and no claims
                self.histogram_plot(df=df,
                                verbose=verbose,
                                savefig=savefig,
                                figloc=figloc, figloc_name='valid_glm_histogram_plots')
                # Calibration curve
                self.calibaration_plot(df=calibration_data,
                          verbose=verbose,
                          savefig=savefig,
                          figloc=figloc, figloc_name='valid_glm_calibration_plots')
            if self.test_set:
                # Test data
                df = pd.DataFrame({"Prob": self.glm_test_pred_mean, self.target_variable[0]: self.df_test[self.target_variable[0]], "Cluster": self.clust_test_preds})
                rounded_probs = np.round(self.glm_test_pred_mean, 2) # Calibration curve
                claims = self.df_test[self.target_variable[0]].values
                bins = np.arange(0, 101, 1) / 100
                binned_counts = np.zeros(len(bins))
                binned_claims = np.zeros(len(bins))
                for i in range(len(bins)):
                    binned_counts[i] = sum(rounded_probs == bins[i])
                    binned_claims[i] = sum(claims[rounded_probs == bins[i]])
                calibration_data = pd.DataFrame([bins, binned_counts, binned_claims]).transpose()
                calibration_data.columns = ['Bins', 'N', 'Claims']
                calibration_data['ObsProb'] = calibration_data['Claims'] / calibration_data['N']
                calibration_data['SE'] = norm.ppf(0.975) * np.sqrt(calibration_data['ObsProb'] * (1 - calibration_data['ObsProb']) / calibration_data['N'])
                calibration_data.dropna(inplace=True)
                # Plot 1
                self.probability_plot(df=df, verbose=verbose, savefig=savefig, figloc=figloc, figloc_name='test_glm_claim_plots', set="Test Set")
                # Histogram of Probabilities for claims and no claims
                self.histogram_plot(df=df,
                                verbose=verbose,
                                savefig=savefig,
                                figloc=figloc, figloc_name='test_glm_histogram_plots')
                # Calibration curve
                self.calibaration_plot(df=calibration_data,
                          verbose=verbose,
                          savefig=savefig,
                          figloc=figloc, figloc_name='valid_glm_calibration_plots') # typo

    def partial_dependence_plot(self,
                                ax,
                                estimator,
                                df,
                                variable,
                                regression_variables,
                                ci=0.95,
                                yaxis_title='Claim Probability'):
        """Partial Dependence Plot for logistic regression

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Pre-existing axes for the plot.
        estimator : statsmodels.discrete.discrete_model.Logit
            Logit model.
        X : pandas.Dataframe of shape (n_train_samples, n_features)
            The data to predict probabilities.
        variable : str
            The variable to produce a partial dependence plot for.
        regression_variables: list 
            List of variables used in estimator
        mu : pandas.Series of shape (n_numerical_features,)
            Contains the mean of the numerical features.
        sigma : pandas.Series of shape (n_numerical_features,)
            Contains the standard deviation of the numerical features.
        yaxis_title: str (default: 'Claim Probability')
            Y axis title.
        """
        alpha = 1 - ci
        z = norm.ppf(1 - alpha / 2, loc=0, scale=1)
        df = df.copy()
        df.reset_index(drop=True, inplace=True)
        if self.glm_scaling:
            df[self.continuous_variables + self.discrete_variables] = self.scale_data(df)
        if 'nominal' in variable:  # Categorical Data - Boxplot
            values = df[variable].cat.categories.values  # Sample space
            length = df.shape[0]
            pdp_values = pd.DataFrame(np.zeros(shape=(length * len(values),  2)), columns=[variable, yaxis_title])  # DF to store probabilities
            for count, n in enumerate(values):  # Iterate over sample space
                df[variable] = n  # Change all values of variable of interest to iteratable value
                pdp_values.loc[(0 + count * length):(length * (count + 1)) - 1, variable] = n  # Store iterable value
                pdp_values.loc[(0 + count * length):(length * (count + 1)) - 1, yaxis_title] = estimator.predict(df).values  # Store probabilities
            sns.boxplot(data=pdp_values,
                        x=variable,
                        y=yaxis_title,
                        ax=ax,
                        showfliers=False,
                        whis=[100 * alpha / 2, 100 - 100 * alpha / 2])  # Boxplot with ci% Quantile
            if self.problem_type=="classification":
                max_yvalue = 1.0
                min_yvalue = -0.1
            else:
                max_yvalue = max(pdp_values[yaxis_title])*1.1
                min_yvalue = min(pdp_values[yaxis_title])*0.9
        else:  # Numerical Data - Lineplot
            values = np.linspace(np.min(df[variable]), np.max(df[variable]))  # Sample space
            pdp_mean = np.array([])  # Stores means
            pdp_lb = np.array([])  # Stores quantile lower bound
            pdp_ub = np.array([])  # Stores quantile upper bound
            pdp_std = np.array([])  # Store standard deviations
            for n in values:  # Iterate over sample space
                df[variable] = n  # Change all values of variable of interest to iteratable value
                pdp_preds = estimator.predict(df)  # Store probabilities
                pdp_mean = np.append(pdp_mean, [np.mean(pdp_preds)])  # Calculate and store mean probability
                pdp_lb = np.append(pdp_lb, [np.quantile(pdp_preds, q=0.05)])  # Calculate and store 5% quantile
                pdp_ub = np.append(pdp_ub, [np.quantile(pdp_preds, q=0.95)])  # Calculate and store 95% quantile
                pdp_std = np.append(pdp_std, [np.std(pdp_preds)])  # Calculate and store standard deviation of probability
            ax.plot(values * self.sigma[variable] + self.mu[variable], pdp_mean)  # Plot mean prob vs sample space (adjusted for standardisation)
            ax.fill_between(
                values * self.sigma[variable] + self.mu[variable],
                pdp_lb,
                pdp_ub,
                alpha=0.2,
                label="{ci}% Quantile".format(ci=ci * 100))  # Shade in region containing ci% of probabilities
            ax.plot(
                values * self.sigma[variable] + self.mu[variable],
                np.add(pdp_mean, pdp_std * z / np.sqrt(len(pdp_mean))),
                color="red",
                linestyle="dashed",
                label="{ci}% Sample Mean CI".format(ci=ci * 100))  # Plot upper bound of 90% CI for sample mean
            ax.plot(values * self.sigma[variable] + self.mu[variable],
                    np.subtract(pdp_mean, pdp_std * z / np.sqrt(len(pdp_mean))),
                    color="red",
                    linestyle="dashed")  # Plot lower bound of 90% CI for sample mean
            ax.legend()
            if self.problem_type=="classification":
                max_yvalue = 1.0
                min_yvalue = -0.1
            else:
                max_yvalue = max(pdp_ub)*1.1
                min_yvalue = min(pdp_lb)*0.9

        ax.set_ylim(bottom=min_yvalue, top=max_yvalue)
        ax.set_ylabel(yaxis_title)
        ax.set_xlabel(variable)
        ax.set_title('Partial Dependence Plot')

    def get_glm_partial_dependency_plots(self,
                                         verbose=True,
                                         savefig=False,
                                         figloc=None):
        """Partial Dependency Plots for GLM.

        Parameters
        ----------
        verbose : bool
            Print results. Default is True.
        savefig : bool
            Save figure. Default is False.
        figloc : str
            Location to save figure. Default is None.
        """
        if self.problem_type=="classification":
            yaxis_title="Claim Probability"
        else:
            yaxis_title="Claim Rate"
        for i in range(self.n_components):# Partial Dependency Plot
            glm_vars = getattr(self, 'clust' + str(i) + '_glm_vars').copy()# Cluster i variables
            glm = getattr(self, 'clust' + str(i) + '_glm') # Clust i glm
            poly_list = []
            vars_to_remove = []
            for var in glm_vars:
                if '**' in var:
                    vars_to_remove += [var]
                    poly_list += [var[len('I('):var.rfind(' **')]]
            glm_vars = list(set(glm_vars)-set(vars_to_remove))
            glm_vars += poly_list
            if self.use_sampling:
                df_train_sample = getattr(self,'clust' + str(i) + '_glm_df_sample')
                df_train_sample_index = df_train_sample.index.values
                df_train = self.df_train.loc[df_train_sample_index].copy()
            else:
                df_train = self.df_train[self.clust_train_preds == i].copy()
            fig, axs = plt.subplots(np.ceil(len(glm_vars) / 2).astype('int64'), 2, figsize=(6, 3 * len(glm_vars) / 2), squeeze=False)
            for ix, variable in enumerate(glm_vars):
                self.partial_dependence_plot(
                    ax=axs[np.floor(ix / 2).astype('int64'), ix % 2],
                    estimator=glm,
                    df=df_train,
                    variable=variable,
                    regression_variables=glm_vars, 
                    yaxis_title=yaxis_title)
            if ix % 2 == 0: # Delete axis if empty plot
                fig.delaxes(axs[np.floor(ix / 2).astype('int64'), 1])
            fig.tight_layout()
            if savefig:
                plt.savefig(fname=figloc + 'clust' + str(i) + '_glm_partial_dependency_plots')
            if verbose:
                print("\nGLM Partial Dependency Plots for cluster {i}.".format(i=i))
                plt.show()
            else:
                plt.close(fig)

    def get_glm_influence(self, kwargs={}):
        """Returns influence instance for GLMs.
        """
        for i in range(self.n_components):
            glm = getattr(self, 'clust' + str(i) + '_glm')
            setattr(self, 'clust' + str(i) + '_glm_influence', glm.get_influence(**kwargs).copy())

    def get_glm_diagnostics_tests(self, Q=10):
        """Performs GLM diagnostic tests (Durbin-Watson, Jarque-Bera and Omnibus tests).
        """
        for i in range(self.n_components):
            # Get GLM
            glm = getattr(self, 'clust' + str(i) + '_glm')
            # Tests
            print("Durbin-Watson statistic: ", stattools.durbin_watson(resids=glm.resid_response, axis=0), "\n")
            print(f"Jarque-Bera test statistic: {stattools.jarque_bera(resids=glm.resid_response, axis=0)[0]}, Pvalue: {stattools.jarque_bera(resids=glm.resid_response, axis=0)[1]}, Skew: {stattools.jarque_bera(resids=glm.resid_response, axis=0)[2]}, Kurtosis: {stattools.jarque_bera(resids=glm.resid_response, axis=0)[3]}\n")
            print(f"Omni Norm test statistic: {stattools.omni_normtest(resids=glm.resid_response, axis=0)[0]}, Pvalue: {stattools.omni_normtest(resids=glm.resid_response, axis=0)[1]}\n")
            if self.problem_type=="classification":
                if self.df[self.target_variable[0]].nunique() == 2:
                    y_train_pred = self.glm_train_pred_class
                    if self.glm_exposure is not None and self.glm_offset is not None:
                        y_train_proba = self.glm_train_pred_mean_exposure_offset_adjusted
                        if self.valid_set:
                            y_valid_proba = self.glm_valid_pred_mean_exposure_offset_adjusted
                        if self.test_set:
                            y_test_proba = self.glm_test_pred_mean_exposure_offset_adjusted
                    elif self.glm_exposure is not None:
                        y_train_proba = self.glm_train_pred_mean_exposure_adjusted
                        if self.valid_set:
                            y_valid_proba = self.glm_valid_pred_mean_exposure_adjusted
                        if self.test_set:
                            y_test_proba = self.glm_test_pred_mean_exposure_adjusted
                    elif self.glm_offset is not None:
                        y_train_proba = self.glm_train_pred_mean_offset_adjusted
                        if self.valid_set:
                            y_valid_proba = self.glm_valid_pred_mean_offset_adjusted
                        if self.test_set:
                            y_test_proba = self.glm_test_pred_mean_offset_adjusted
                    else:
                        y_train_proba = self.glm_train_pred_mean
                        if self.valid_set:
                            y_valid_proba = self.glm_valid_pred_mean
                        if self.test_set:    
                            y_test_proba = self.glm_test_pred_mean
                    res = pd.DataFrame({'pred':y_train_pred, 'prob':y_train_proba, 'claims':self.df_train[self.target_variable[0]], 'cluster':self.clust_train_preds})
                    sorted_res = res.sort_values('prob').reset_index(drop=True)
                    pvalue = []
                    chisq = []
                    df = []
                    for j in range(self.n_components):
                        cluster_res = sorted_res[sorted_res['cluster']==j].copy().reset_index(drop=True)
                        n = int(cluster_res.shape[0]/Q)
                        chisq_value = 0
                        for i in range(Q):
                            if i == Q-1:
                                pi = np.mean(cluster_res[(i * n):]['prob'])
                                obs = sum(cluster_res[(i * n):]['claims'])
                                n = len(cluster_res[(i * n):])
                                exp = n * pi
                                chisq_value += ((obs-exp)**2) / (n * pi * (1-pi))                
                            else:
                                pi = np.mean(cluster_res[(i * n):(i*n+n)]['prob'])
                                obs = sum(cluster_res[(i * n):(i*n+n)]['claims'])
                                exp = n * pi
                                chisq_value += ((obs-exp)**2) / (n * pi * (1-pi))
                        pvalue.append(np.round(1 - chi2.cdf(chisq_value, Q - 2), 4))
                        chisq.append(np.round(chisq_value,4))
                        df.append(Q-2)
                    chi_df = pd.DataFrame({'df':df, "chisq_value":chisq,"pvalue":pvalue})
                    chi_df.reset_index(inplace=True)
                    chi_df.rename(columns={'index':'Cluster'}, inplace=True)
                    print("Hosmer-Lemeshow Test for training set:")
                    display(chi_df)
                    if self.valid_set:
                        y_valid_pred = self.glm_valid_pred_class
                        res = pd.DataFrame({'pred':y_valid_pred, 'prob':y_valid_proba, 'claims':self.df_valid[self.target_variable[0]], 'cluster':self.clust_valid_preds})
                        sorted_res = res.sort_values('prob').reset_index(drop=True)
                        pvalue = []
                        chisq = []
                        df = []
                        for j in range(self.n_components):
                            cluster_res = sorted_res[sorted_res['cluster']==j].copy().reset_index(drop=True)
                            n = int(cluster_res.shape[0]/Q)
                            chisq_value = 0
                            for i in range(Q):
                                if i == Q-1:
                                    pi = np.mean(cluster_res[(i * n):]['prob'])
                                    obs = sum(cluster_res[(i * n):]['claims'])
                                    n = len(cluster_res[(i * n):])
                                    exp = n * pi
                                    chisq_value += ((obs-exp)**2) / (n * pi * (1-pi))                
                                else:
                                    pi = np.mean(cluster_res[(i * n):(i*n+n)]['prob'])
                                    obs = sum(cluster_res[(i * n):(i*n+n)]['claims'])
                                    exp = n * pi
                                    chisq_value += ((obs-exp)**2) / (n * pi * (1-pi))
                            pvalue.append(np.round(1 - chi2.cdf(chisq_value, Q - 2), 4))
                            chisq.append(np.round(chisq_value,4))
                            df.append(Q-2)
                        chi_df = pd.DataFrame({'df':df, "chisq_value":chisq,"pvalue":pvalue})
                        chi_df.reset_index(inplace=True)
                        chi_df.rename(columns={'index':'Cluster'}, inplace=True)
                        print("Hosmer-Lemeshow Test for validation set:")
                        display(chi_df)
                    if self.test_set:
                        y_test_pred = self.glm_test_pred_class
                        res = pd.DataFrame({'pred':y_test_pred, 'prob':y_test_proba, 'claims':self.df_test[self.target_variable[0]], 'cluster':self.clust_test_preds})
                        sorted_res = res.sort_values('prob').reset_index(drop=True)
                        pvalue = []
                        chisq = []
                        df = []
                        for j in range(self.n_components):
                            cluster_res = sorted_res[sorted_res['cluster']==j].copy().reset_index(drop=True)
                        
                            n = int(cluster_res.shape[0]/Q)
                            chisq_value = 0
                            for i in range(Q):
                                if i == Q-1:
                                    pi = np.mean(cluster_res[(i * n):]['prob'])
                                    obs = sum(cluster_res[(i * n):]['claims'])
                                    n = len(cluster_res[(i * n):])
                                    exp = n * pi
                                    chisq_value += ((obs-exp)**2) / (n * pi * (1-pi))                
                                else:
                                    pi = np.mean(cluster_res[(i * n):(i*n+n)]['prob'])
                                    obs = sum(cluster_res[(i * n):(i*n+n)]['claims'])
                                    exp = n * pi
                                    chisq_value += ((obs-exp)**2) / (n * pi * (1-pi))
                            pvalue.append(np.round(1 - chi2.cdf(chisq_value, Q - 2), 4))
                            chisq.append(np.round(chisq_value,4))
                            df.append(Q-2)
                        chi_df = pd.DataFrame({'df':df, "chisq_value":chisq,"pvalue":pvalue})
                        chi_df.reset_index(inplace=True)
                        chi_df.rename(columns={'index':'Cluster'}, inplace=True)
                        print("Hosmer-Lemeshow Test for validation set: ") # typo
                        display(chi_df)

    def get_glm_diagnostic_plots(self,
                                 verbose=True,
                                 savefig=False,
                                 figloc=None,
                                 res_type='Pearson'):
        """Diagnostic plots for GLM. Plot 1 is Residuals vs Fitted, Plot 2 is a QQ plot, Plot 3 is a histogram of residuals, Plot 4 is a scale-location plot, Plot 5 is a residuals vs leverage plot and Plot 6 is Cook's Distance.

        Parameters
        ----------
        verbose : bool
            Print results. Default is True.
        savefig : bool
            Save figure. Default is False.
        figloc : str
            Location to save figure. Default is None.
        res_type : str
            Residual type. Default is 'Pearson'.

        Notes
        -----
        We use either Pearson or Deviance residuals. The residuals are "internally studentized" or "standardized". This is the R equivalent to rstandard().
        """
        for i in range(self.n_components):
            # Get GLM
            glm = getattr(self, 'clust' + str(i) + '_glm')
            p = glm.df_model + 1 # model rank
            mod_leverage = glm.get_hat_matrix_diag(observed=True)
            mod_cooks = glm.get_influence(observed=True).cooks_distance[0]
            if res_type == 'Pearson':
                resid = glm.resid_pearson.copy()
                std_resid = glm.get_influence(observed=True).resid_studentized
            elif res_type == 'Deviance':
                resid = glm.resid_deviance.copy()
                std_resid = resid / np.sqrt(glm.scale*(1-mod_leverage))
            else:
                print('Unsupported residual type')
                return None
            # Plots
            fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8, 12))
            sns.regplot(x=glm.fittedvalues,
                        y=resid,
                        scatter=True,
                        ci=None,
                        lowess=True,
                        line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
                        ax=axs[0, 0])
            axs[0, 0].axhline(linestyle='dotted')
            axs[0, 0].set_title('Residuals vs Fitted')
            axs[0, 0].set_xlabel('Predicted values')
            axs[0,0].set_ylabel('{res_type} Residuals'.format(res_type=res_type))
            for j in abs(resid).sort_values(ascending=False)[:3].index:
                axs[0, 0].annotate(j, xy=(glm.fittedvalues[j], resid[j]))
            sm.qqplot(std_resid, line='45', ax=axs[0, 1])
            axs[0, 1].set_title('Normal Q-Q Plot')
            axs[0, 1].set_ylabel('Standardized {res_type} Residuals'.format(res_type=res_type))
            axs[0, 1].set_xlabel('Theoretical Quantiles')
            axs[1, 0].hist(std_resid, bins=25)
            axs[1, 0].set_title('Histogram of Standardized {res_type} Residuals'.format(res_type=res_type))
            axs[1, 0].set_ylabel('Count')
            axs[1, 0].set_xlabel('Standardized {res_type} Residuals'.format(res_type=res_type))
            sns.regplot(x=glm.fittedvalues,
                        y=np.sqrt(abs(std_resid)),
                        scatter=True,
                        ci=None,
                        lowess=True,
                        line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
                        ax=axs[1, 1])
            axs[1, 1].set_title('Scale-Location plot')
            axs[1, 1].set_ylabel(r'$\sqrt{| Standardized %s Residuals |}$' % (res_type))
            axs[1, 1].set_xlabel('Predicted values')
            for j in np.flip(np.argsort(np.sqrt(abs(std_resid))))[:3]: # annotate top 3 residuals
                axs[1, 1].annotate(j, xy=(glm.fittedvalues.values[j], np.sqrt(abs(std_resid))[j]))
            sns.regplot(x=mod_leverage,
                        y=std_resid,
                        scatter=True,
                        ci=None,
                        lowess=True,
                        line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
                        ax=axs[2, 0])
            axs[2, 0].plot(np.linspace(min(mod_leverage), max(mod_leverage), 50), 
                           (lambda x: np.sqrt((0.5 * p * (1 - x)) / x))(np.linspace( min(mod_leverage), max(mod_leverage), 50)),
                           label='Cooks Distance',
                           lw=1,
                           ls='--',
                           color='red')
            axs[2, 0].set_title('Residuals vs Leverage')
            axs[2, 0].set_xlabel('Leverage')
            axs[2, 0].set_ylabel('Standardized {res_type} Residuals'.format(res_type=res_type))
            axs[2, 0].set_ylim(min(std_resid) - 1, max(std_resid) + 1)
            axs[2, 0].legend()
            for j in np.flip(np.argsort(mod_leverage))[:3]: # annotate top 3 leverage points
                axs[2, 0].annotate(j, xy=(mod_leverage[j], std_resid[j]))
            axs[2, 1].plot(mod_cooks, 'o')
            axs[2, 1].set_title("Cook's Distance")
            axs[2, 1].set_ylabel("Cook's Distance")
            axs[2, 1].set_xlabel("Index")
            for j in np.flip(np.argsort(mod_cooks))[:3]: # annotate top 3 cooks distance points
                axs[2, 1].annotate(j, xy=(j, mod_cooks[j]))
            fig.tight_layout()
            if savefig:
                plt.savefig(fname=figloc + 'clust' + str(i) + '_glm_diagnostics')
            if verbose:
                print("\nGLM Diagnostics for cluster {i}.".format(i=i))
                plt.show()
            else:
                plt.close(fig)

    def get_glm_diagnostic_plots_2(self, verbose=True, savefig=False, figloc=None):
        for i in range(self.n_components):
            glm = getattr(self, 'clust'+str(i)+'_glm')
            if self.use_sampling:
                clust_i_df_train = getattr(self, 'clust'+str(i)+'_glm_df_sample').copy()
            else:
                clust_i_df_train = self.df_train[self.clust_train_preds==i].copy()
                
            if self.glm_offset is None: # Offset
                clust_i_df_sample_offset = None
            else:
                clust_i_df_train_offset = np.asarray(clust_i_df_train[self.glm_offset].copy()) # this is unscaled
            if self.glm_exposure is None: # Exposure
                clust_i_df_train_exposure = None
            else:
                clust_i_df_train_exposure = np.asarray(clust_i_df_train[self.glm_exposure].copy())
                    
            if self.glm_scaling:
                clust_i_df_train[self.continuous_variables+self.discrete_variables] = self.scale_data(clust_i_df_train)
            y_train_proba = glm.predict(clust_i_df_train, which="mean", exposure=clust_i_df_train_exposure, offset=clust_i_df_sample_offset)
            cooks = glm.get_influence(observed=True).cooks_distance[0]
            leverage = glm.get_hat_matrix_diag(observed=True)
            delta_d = (glm.resid_deviance.copy()**2) / (1-leverage)
            delta_chi = (glm.resid_pearson/np.sqrt(1-leverage))**2
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
            axs[0,0].plot(y_train_proba, leverage, 'o')
            axs[0,0].set_xlabel('Estimated Probability')
            axs[0,0].set_ylabel('Leverage')
            #
            axs[0,1].plot(y_train_proba, delta_chi, 'o')
            axs[0,1].set_xlabel('Estimated Probability')
            axs[0,1].set_ylabel('Change in Pearson chi-square')
            #
            axs[1,0].plot(y_train_proba, delta_d, 'o')
            axs[1,0].set_xlabel('Estimated Probability')
            axs[1,0].set_ylabel('Change in deviance')
            #
            axs[1,1].plot(y_train_proba, cooks, 'o')
            axs[1,1].set_xlabel('Estimated Probability')
            axs[1,1].set_ylabel("Cook's distance")
            fig.tight_layout()
            if savefig:
                plt.savefig(fname=figloc + 'clust' + str(i) +'_glm_other_diagnostics1')
            if verbose:
                print("\nGLM Diagnostics for cluster {i}.".format(i=i))
                plt.show()
            else:
                plt.close(fig)   
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
            axs[0,0].plot(leverage, delta_chi, 'o')
            axs[0,0].set_xlabel('Leverage')
            axs[0,0].set_ylabel('Change in Pearson chi-square')
            #
            axs[0,1].plot(leverage, delta_d, 'o')
            axs[0,1].set_xlabel('Leverage')
            axs[0,1].set_ylabel('Change in deviance')
            #
            axs[1,0].plot(leverage, cooks, 'o')
            axs[1,0].set_xlabel('Leverage')
            axs[1,0].set_ylabel("Cook's distance")
            fig.delaxes(axs[1,1])
            fig.tight_layout()
            if savefig:
                plt.savefig(fname=figloc + 'clust' + str(i) + '_glm_other_diagnostics2')
            if verbose:
                plt.show()
            else:
                plt.close(fig)
            df = pd.DataFrame({'Estimated Probability':y_train_proba, 'Change in Pearson chi-square':delta_chi, "Cook's D":cooks})
            relplot = sns.relplot(data=df, 
                                x="Estimated Probability", 
                                y="Change in Pearson chi-square", 
                                size="Cook's D")
            relplot.tight_layout()
            if savefig:
                relplot.savefig(fname=figloc + 'clust' + str(i) +'_glm_other_diagnostics3')
            if verbose:
                plt.show()
            else:
                plt.close(relplot)

    def get_glm_metrics(self,
                        by_cluster=True,
                        metric_kwargs={}):
        """Compute GLM metrics.

        Parameters
        ----------
        by_cluter : bool
            Compute metrics by cluster. Default is True.
        return_results : bool
            Return results. Default is True.
        
        Returns
        -------
        list
            List of results. 
        """
        #### TO DO: this didn't print if by cluster is False
        # removed mean gamma deviance 
        regression_metric_list = ['r2_score', 'mean_absolute_error', 'mean_squared_error', 'mean_squared_log_error', 'mean_absolute_percentage_error', 'median_absolute_error', 'max_error', 'explained_variance_score', 'mean_tweedie_deviance', 'mean_pinball_loss', 'root_mean_squared_error', 'root_mean_squared_log_error', 'mean_poisson_deviance', 'd2_absolute_error_score']
        classification_metric_list_pred=['mavg',  'accuracy_score', 'balanced_accuracy_score', 'f1_score', 'precision_score', 'recall_score', 'jaccard_score']
        classification_metric_list_proba=['top_k_accuracy_score', 'average_precision_score', 'brier_score_loss', 'log_loss', 'roc_auc_score'] # They require probabilities, not 0 or 1
        if by_cluster:
            for i in range(self.n_components):
                if self.glm_exposure is not None and self.glm_offset is not None:
                    which = "exposure_offset_adjusted"
                elif self.glm_exposure is not None:
                    which = "exposure_adjusted"
                elif self.glm_offset is not None:
                    which = "offset_adjusted"
                else:
                    which = "mean"
                clust_i_train_pred_mean = self.predict_glm(self.df_train[self.clust_train_preds == i], which=which, return_all=False, glm_scaling=self.glm_scaling, weighted=False)
                #if self.trials:
                #    clust_i_train_pred_mean *= self.df_train[self.clust_train_preds == i]["trials"]
                if self.problem_type=="classification":
                    clust_i_train_pred_class = self.predict_glm(self.df_train[self.clust_train_preds == i], which="class", return_all=False, glm_scaling=self.glm_scaling, weighted=False)
                if self.valid_set:
                    clust_i_valid_pred_mean = self.predict_glm(self.df_valid[self.clust_valid_preds == i], which=which, return_all=False, glm_scaling=self.glm_scaling, weighted=False)
                    #if self.trials:
                    #    clust_i_valid_pred_mean *= self.df_valid[self.clust_valid_preds == i]["trials"]
                    if self.problem_type=="classification":
                        clust_i_valid_pred_class = self.predict_glm(self.df_valid[self.clust_valid_preds == i], which="class", return_all=False, glm_scaling=self.glm_scaling, weighted=False)
                if self.test_set:
                    clust_i_test_pred_mean = self.predict_glm(self.df_test[self.clust_test_preds == i], which=which, return_all=False, glm_scaling=self.glm_scaling, weighted=False)
                    #if self.trials:
                    #    clust_i_test_pred_mean *= self.df_test[self.clust_test_preds == i]["trials"]
                    if self.problem_type=="classification":
                        clust_i_test_pred_class = self.predict_glm(self.df_test[self.clust_test_preds == i], which="class", return_all=False, glm_scaling=self.glm_scaling, weighted=False)
                t = PrettyTable()
                t.title = "Cluster " + str(i)
                if self.valid_set and self.test_set:
                    t.field_names = ['Metric', 'Training', 'Validation', 'Test']
                    if self.problem_type=="regression":
                        for metric in regression_metric_list:
                            t.add_row([metric, 
                                    np.round(self.glm_metric(y_true=self.df_train.loc[self.clust_train_preds == i, self.target_variable[0]], y_pred=clust_i_train_pred_mean, metric=metric, problem_type="regression", metric_kwargs=metric_kwargs), 4), 
                                    np.round(self.glm_metric(y_true=self.df_valid.loc[self.clust_valid_preds == i, self.target_variable[0]], y_pred=clust_i_valid_pred_mean, metric=metric, problem_type="regression", metric_kwargs=metric_kwargs), 4), 
                                    np.round(self.glm_metric(y_true=self.df_test.loc[self.clust_test_preds == i, self.target_variable[0]], y_pred=clust_i_test_pred_mean, metric=metric, problem_type="regression", metric_kwargs=metric_kwargs), 4)])
                    if self.problem_type=="classification":
                        t.add_row(['baseline',
                                np.round(1 - sum(self.df_train.loc[self.clust_train_preds == i, self.target_variable[0]]) / len(self.df_train.loc[self.clust_train_preds == i, self.target_variable[0]]), 4),
                                np.round(1 - sum(self.df_valid.loc[self.clust_valid_preds == i, self.target_variable[0]]) / len(self.df_valid.loc[self.clust_valid_preds == i, self.target_variable[0]]), 4),
                                np.round(1 - sum(self.df_test.loc[self.clust_test_preds == i, self.target_variable[0]]) / len(self.df_test.loc[self.clust_test_preds == i, self.target_variable[0]]), 4) ])
                        for metric in classification_metric_list_pred:
                            t.add_row([metric, 
                                    np.round(self.glm_metric(y_true=self.df_train.loc[self.clust_train_preds == i, self.target_variable[0]], y_pred=clust_i_train_pred_class, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4), 
                                    np.round(self.glm_metric(y_true=self.df_valid.loc[self.clust_valid_preds == i, self.target_variable[0]], y_pred=clust_i_valid_pred_class, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4), 
                                    np.round(self.glm_metric(y_true=self.df_test.loc[self.clust_test_preds == i, self.target_variable[0]], y_pred=clust_i_test_pred_class, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4)])
                        for metric in classification_metric_list_proba:
                            t.add_row([metric, 
                                    np.round(self.glm_metric(y_true=self.df_train.loc[self.clust_train_preds == i, self.target_variable[0]], y_pred=clust_i_train_pred_mean, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4), 
                                    np.round(self.glm_metric(y_true=self.df_valid.loc[self.clust_valid_preds == i, self.target_variable[0]], y_pred=clust_i_valid_pred_mean, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4), 
                                    np.round(self.glm_metric(y_true=self.df_test.loc[self.clust_test_preds == i, self.target_variable[0]], y_pred=clust_i_test_pred_mean, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4)])
                elif self.valid_set:
                    t.field_names = ['Metric', 'Training', 'Validation']
                    if self.problem_type=="regression":
                        for metric in regression_metric_list:
                            t.add_row([metric, 
                                    np.round(self.glm_metric(y_true=self.df_train.loc[self.clust_train_preds == i, self.target_variable[0]], y_pred=clust_i_train_pred_mean, metric=metric, problem_type="regression", metric_kwargs=metric_kwargs), 4), 
                                    np.round(self.glm_metric(y_true=self.df_valid.loc[self.clust_valid_preds == i, self.target_variable[0]], y_pred=clust_i_valid_pred_mean, metric=metric, problem_type="regression", metric_kwargs=metric_kwargs), 4)])
                    if self.problem_type=="classification":
                        t.add_row(['baseline',
                                np.round(1 - sum(self.df_train.loc[self.clust_train_preds == i, self.target_variable[0]]) / len(self.df_train.loc[self.clust_train_preds == i, self.target_variable[0]], problem_type="classification", metric_kwargs=metric_kwargs), 4),
                                np.round(1 - sum(self.df_valid.loc[self.clust_valid_preds == i, self.target_variable[0]]) / len(self.df_valid.loc[self.clust_valid_preds == i, self.target_variable[0]], problem_type="classification", metric_kwargs=metric_kwargs), 4) ])
                        for metric in classification_metric_list_pred:
                            t.add_row([metric, 
                                    np.round(self.glm_metric(y_true=self.df_train.loc[self.clust_train_preds == i, self.target_variable[0]], y_pred=clust_i_train_pred_class, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4), 
                                    np.round(self.glm_metric(y_true=self.df_valid.loc[self.clust_valid_preds == i, self.target_variable[0]], y_pred=clust_i_valid_pred_class, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4)])
                        for metric in classification_metric_list_proba:
                            t.add_row([metric, 
                                    np.round(self.glm_metric(y_true=self.df_train.loc[self.clust_train_preds == i, self.target_variable[0]], y_pred=clust_i_train_pred_mean, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4), 
                                    np.round(self.glm_metric(y_true=self.df_valid.loc[self.clust_valid_preds == i, self.target_variable[0]], y_pred=clust_i_valid_pred_mean, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4)])
                elif self.test_set:
                    t.field_names = ['Metric', 'Training', 'Test']
                    if self.problem_type=="regression":
                        for metric in regression_metric_list:
                            t.add_row([metric, 
                                    np.round(self.glm_metric(y_true=self.df_train.loc[self.clust_train_preds == i, self.target_variable[0]], y_pred=clust_i_train_pred_mean, metric=metric, problem_type="regression", metric_kwargs=metric_kwargs), 4), 
                                    np.round(self.glm_metric(y_true=self.df_test.loc[self.clust_test_preds == i, self.target_variable[0]], y_pred=clust_i_test_pred_mean, metric=metric, problem_type="regression", metric_kwargs=metric_kwargs), 4)])
                    if self.problem_type=="classification":
                        t.add_row(['baseline',
                                np.round(1 - sum(self.df_train.loc[self.clust_train_preds == i, self.target_variable[0]]) / len(self.df_train.loc[self.clust_train_preds == i, self.target_variable[0]], problem_type="classification", metric_kwargs=metric_kwargs), 4),
                                np.round(1 - sum(self.df_test.loc[self.clust_test_preds == i, self.target_variable[0]]) / len(self.df_test.loc[self.clust_test_preds == i, self.target_variable[0]], problem_type="classification", metric_kwargs=metric_kwargs), 4) ])
                        for metric in classification_metric_list_pred:
                            t.add_row([metric, 
                                    np.round(self.glm_metric(y_true=self.df_train.loc[self.clust_train_preds == i, self.target_variable[0]], y_pred=clust_i_train_pred_class, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4), 
                                    np.round(self.glm_metric(y_true=self.df_test.loc[self.clust_test_preds == i, self.target_variable[0]], y_pred=clust_i_test_pred_class, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4)])
                        for metric in classification_metric_list_proba:
                            t.add_row([metric, 
                                    np.round(self.glm_metric(y_true=self.df_train.loc[self.clust_train_preds == i, self.target_variable[0]], y_pred=clust_i_train_pred_mean, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4), 
                                    np.round(self.glm_metric(y_true=self.df_test.loc[self.clust_test_preds == i, self.target_variable[0]], y_pred=clust_i_test_pred_mean, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4)])
                else:
                    t.field_names = ['Metric', 'Training']
                    if self.problem_type=="regression":
                        for metric in regression_metric_list:
                            t.add_row([metric, 
                                    np.round(self.glm_metric(y_true=self.df_train.loc[self.clust_train_preds == i, self.target_variable[0]], y_pred=clust_i_train_pred_mean, metric=metric, problem_type="regression", metric_kwargs=metric_kwargs), 4)])
                    if self.problem_type=="classification":
                        t.add_row(['baseline',
                                np.round(1 - sum(self.df_train.loc[self.clust_train_preds == i, self.target_variable[0]]) / len(self.df_train.loc[self.clust_train_preds == i, self.target_variable[0]]), 4)])
                        for metric in classification_metric_list_pred:
                            t.add_row([metric, 
                                    np.round(self.glm_metric(y_true=self.df_train.loc[self.clust_train_preds == i, self.target_variable[0]], y_pred=clust_i_train_pred_class, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4)])
                        for metric in classification_metric_list_proba:
                            t.add_row([metric, 
                                    np.round(self.glm_metric(y_true=self.df_train.loc[self.clust_train_preds == i, self.target_variable[0]], y_pred=clust_i_train_pred_mean, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4)])
                print(t)
            t = PrettyTable() # Combined results
            t.title = "Total Data"
            if self.valid_set and self.test_set:
                t.field_names = ['Metric', 'Training', 'Validation', 'Test']
                if self.glm_exposure is not None and self.glm_offset is not None:
                    y_pred_train = self.glm_train_pred_mean_exposure_offset_adjusted
                    y_pred_valid = self.glm_valid_pred_mean_exposure_offset_adjusted
                    y_pred_test = self.glm_test_pred_mean_exposure_offset_adjusted
                elif self.glm_exposure is not None:
                    y_pred_train = self.glm_train_pred_mean_exposure_adjusted
                    y_pred_valid = self.glm_valid_pred_mean_exposure_adjusted
                    y_pred_test = self.glm_test_pred_mean_exposure_adjusted
                elif self.glm_offset is not None:
                    y_pred_train = self.glm_train_pred_mean_offset_adjusted
                    y_pred_valid = self.glm_valid_pred_mean_offset_adjusted
                    y_pred_test = self.glm_test_pred_mean_offset_adjusted
                else:
                    y_pred_train = self.glm_train_pred_mean
                    y_pred_valid = self.glm_valid_pred_mean
                    y_pred_test = self.glm_test_pred_mean
                if self.problem_type=="regression":
                    for metric in regression_metric_list:
                        t.add_row([metric, 
                                np.round(self.glm_metric(y_true=self.df_train[self.target_variable[0]], y_pred=y_pred_train, metric=metric, problem_type="regression", metric_kwargs=metric_kwargs), 4), 
                                np.round(self.glm_metric(y_true=self.df_valid[self.target_variable[0]], y_pred=y_pred_valid, metric=metric, problem_type="regression", metric_kwargs=metric_kwargs), 4), 
                                np.round(self.glm_metric(y_true=self.df_test[self.target_variable[0]], y_pred=y_pred_test, metric=metric, problem_type="regression", metric_kwargs=metric_kwargs), 4)])
                if self.problem_type=="classification":
                    t.add_row(['baseline',
                            np.round(1 - sum(self.df_train[self.target_variable[0]]) / self.df_train.shape[0], 4),
                            np.round(1 - sum(self.df_valid[self.target_variable[0]]) / self.df_valid.shape[0], 4),
                            np.round(1 - sum(self.df_test[self.target_variable[0]]) / self.df_test.shape[0], 4) ])
                    for metric in classification_metric_list_pred:
                        t.add_row([metric, 
                                   np.round(self.glm_metric(y_true=self.df_train[self.target_variable[0]], y_pred=self.glm_train_pred_class, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4), 
                                   np.round(self.glm_metric(y_true=self.df_valid[self.target_variable[0]], y_pred=self.glm_valid_pred_class, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4), 
                                   np.round(self.glm_metric(y_true=self.df_test[self.target_variable[0]], y_pred=self.glm_test_pred_class, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4)])
                    for metric in classification_metric_list_proba:
                        t.add_row([metric, 
                                   np.round(self.glm_metric(y_true=self.df_train[self.target_variable[0]], y_pred=y_pred_train, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4), 
                                   np.round(self.glm_metric(y_true=self.df_valid[self.target_variable[0]], y_pred=y_pred_valid, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4), 
                                   np.round(self.glm_metric(y_true=self.df_test[self.target_variable[0]], y_pred=y_pred_test, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4)])
            elif self.valid_set:
                t.field_names = ['Metric', 'Training', 'Validation']
                if self.glm_exposure is not None and self.glm_offset is not None:
                    y_pred_train = self.glm_train_pred_mean_exposure_offset_adjusted
                    y_pred_valid = self.glm_valid_pred_mean_exposure_offset_adjusted
                elif self.glm_exposure is not None:
                    y_pred_train = self.glm_train_pred_mean_exposure_adjusted
                    y_pred_valid = self.glm_valid_pred_mean_exposure_adjusted
                elif self.glm_offset is not None:
                    y_pred_train = self.glm_train_pred_mean_offset_adjusted
                    y_pred_valid = self.glm_valid_pred_mean_offset_adjusted
                else:
                    y_pred_train = self.glm_train_pred_mean
                    y_pred_valid = self.glm_valid_pred_mean
                if self.problem_type=="regression":
                    for metric in regression_metric_list:
                        t.add_row([metric, 
                                np.round(self.glm_metric(y_true=self.df_train[self.target_variable[0]], y_pred=y_pred_train, metric=metric, problem_type="regression", metric_kwargs=metric_kwargs), 4), 
                                np.round(self.glm_metric(y_true=self.df_valid[self.target_variable[0]], y_pred=y_pred_valid, metric=metric, problem_type="regression", metric_kwargs=metric_kwargs), 4)])
                if self.problem_type=="classification":
                    t.add_row(['baseline',
                            np.round(1 - sum(self.df_train[self.target_variable[0]]) / self.df_train.shape[0], 4),
                            np.round(1 - sum(self.df_valid[self.target_variable[0]]) / self.df_valid.shape[0], 4) ])
                    for metric in classification_metric_list_pred:
                        t.add_row([metric, 
                                   np.round(self.glm_metric(y_true=self.df_train[self.target_variable[0]], y_pred=self.glm_train_pred_class, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4), 
                                   np.round(self.glm_metric(y_true=self.df_valid[self.target_variable[0]], y_pred=self.glm_valid_pred_class, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4)])
                    for metric in classification_metric_list_proba:
                        t.add_row([metric, 
                                   np.round(self.glm_metric(y_true=self.df_train[self.target_variable[0]], y_pred=y_pred_train, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4), 
                                   np.round(self.glm_metric(y_true=self.df_valid[self.target_variable[0]], y_pred=y_pred_valid, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4)])
            elif self.test_set:
                t.field_names = ['Metric', 'Training', 'Test']
                if self.glm_exposure is not None and self.glm_offset is not None:
                    y_pred_train = self.glm_train_pred_mean_exposure_offset_adjusted
                    y_pred_test = self.glm_test_pred_mean_exposure_offset_adjusted
                elif self.glm_exposure is not None:
                    y_pred_train = self.glm_train_pred_mean_exposure_adjusted
                    y_pred_test = self.glm_test_pred_mean_exposure_adjusted
                elif self.glm_offset is not None:
                    y_pred_train = self.glm_train_pred_mean_offset_adjusted
                    y_pred_test = self.glm_test_pred_mean_offset_adjusted
                else:
                    y_pred_train = self.glm_train_pred_mean
                    y_pred_test = self.glm_test_pred_mean
                if self.problem_type=="regression":
                    for metric in regression_metric_list:
                        t.add_row([metric, 
                                np.round(self.glm_metric(y_true=self.df_train[self.target_variable[0]], y_pred=y_pred_train, metric=metric, problem_type="regression", metric_kwargs=metric_kwargs), 4), 
                                np.round(self.glm_metric(y_true=self.df_test[self.target_variable[0]], y_pred=y_pred_test, metric=metric, problem_type="regression", metric_kwargs=metric_kwargs), 4)])
                if self.problem_type=="classification":
                    t.add_row(['baseline',
                            np.round(1 - sum(self.df_train[self.target_variable[0]]) / self.df_train.shape[0], 4),
                            np.round(1 - sum(self.df_test[self.target_variable[0]]) / self.df_test.shape[0], 4) ])
                    for metric in classification_metric_list_pred:
                        t.add_row([metric, 
                                   np.round(self.glm_metric(y_true=self.df_train[self.target_variable[0]], y_pred=self.glm_train_pred_class, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4), 
                                   np.round(self.glm_metric(y_true=self.df_test[self.target_variable[0]], y_pred=self.glm_test_pred_class, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4)])
                    for metric in classification_metric_list_proba:
                        t.add_row([metric, 
                                   np.round(self.glm_metric(y_true=self.df_train[self.target_variable[0]], y_pred=y_pred_train, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4), 
                                   np.round(self.glm_metric(y_true=self.df_test[self.target_variable[0]], y_pred=y_pred_test, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4)])
            else:
                t.field_names = ['Metric', 'Training']
                if self.glm_exposure is not None and self.glm_offset is not None:
                    y_pred_train = self.glm_train_pred_mean_exposure_offset_adjusted
                elif self.glm_exposure is not None:
                    y_pred_train = self.glm_train_pred_mean_exposure_adjusted
                elif self.glm_offset is not None:
                    y_pred_train = self.glm_train_pred_mean_offset_adjusted
                else:
                    y_pred_train = self.glm_train_pred_mean
                if self.problem_type=="regression":
                    for metric in regression_metric_list:
                        t.add_row([metric, 
                                np.round(self.glm_metric(y_true=self.df_train.loc[self.clust_train_preds == i, self.target_variable[0]], y_pred=y_pred_train, metric=metric, problem_type="regression", metric_kwargs=metric_kwargs), 4)])
                if self.problem_type=="classification":
                    t.add_row(['baseline',
                               np.round(1 - sum(self.df_train[self.target_variable[0]]) / self.df_train.shape[0], 4) ])
                    for metric in classification_metric_list_pred:
                        t.add_row([metric, 
                                   np.round(self.glm_metric(y_true=self.df_train[self.target_variable[0]], y_pred=self.glm_train_pred_class, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4)])
                    for metric in classification_metric_list_proba:
                        t.add_row([metric, 
                                   np.round(self.glm_metric(y_true=self.df_train[self.target_variable[0]], y_pred=y_pred_train, metric=metric, problem_type="classification", metric_kwargs=metric_kwargs), 4)])
            print(t)

    def get_glm_cf_matrix(self):
        """GLM Confusion Matrix. Rows are actuals and columns are predictions.
        """
        if self.problem_type=="regression":
            print("Incorrect GLM Family specified.")
            sys.exit()
        for i in range(self.n_components):
            if self.use_sampling:
                df_train_sample = getattr(self,'clust' + str(i) + '_glm_df_sample')
                df_train_sample_index = df_train_sample.index.values
                df_train = self.df_train.loc[df_train_sample_index].copy()
                clust_i_lr_sample_preds = self.predict_glm(df=df_train, which="class", return_all=False, glm_scaling=self.glm_scaling, weighted=False)
                print("\nCluster {} Sampling Confusion Matrix:".format(i))
                print(metrics.confusion_matrix(y_true=df_train[self.target_variable], y_pred=clust_i_lr_sample_preds))
            clust_i_lr_train_preds = self.predict_glm(self.df_train[self.clust_train_preds == i], which="class", return_all=False, glm_scaling=self.glm_scaling, weighted=False)
            print("\nCluster {} Training Confusion Matrix:".format(i))
            print(metrics.confusion_matrix(y_true=self.df_train.loc[self.clust_train_preds == i, self.target_variable], y_pred=clust_i_lr_train_preds))
            if self.valid_set:
                clust_i_lr_valid_preds = self.predict_glm(self.df_valid[self.clust_valid_preds == i], which="class", return_all=False, glm_scaling=self.glm_scaling, weighted=False)
                print("\nCluster {} Validation Confusion Matrix:".format(i))
                print(metrics.confusion_matrix(y_true=self.df_valid.loc[self.clust_valid_preds == i, self.target_variable], y_pred=clust_i_lr_valid_preds))
            if self.test_set:
                clust_i_lr_test_preds = self.predict_glm(self.df_test[self.clust_test_preds == i], which="class", return_all=False, glm_scaling=self.glm_scaling, weighted=False)
                print("\nCluster {} Test Confusion Matrix:".format(i))
                print(metrics.confusion_matrix(y_true=self.df_test.loc[self.clust_test_preds == i, self.target_variable], y_pred=clust_i_lr_test_preds))
        print("\nTraining Confusion Matrix:")
        print(metrics.confusion_matrix(y_true=self.df_train[self.target_variable], y_pred=self.glm_train_pred_class))
        if self.valid_set:
            print("\nValidation Confusion Matrix:")
            print(metrics.confusion_matrix(y_true=self.df_valid[self.target_variable], y_pred=self.glm_valid_pred_class))
        if self.test_set:
            print("\nTest Confusion Matrix:")
            print(metrics.confusion_matrix(y_true=self.df_test[self.target_variable], y_pred=self.glm_test_pred_class))
            
    def glm_formula(self, regression_variables, intercept=True):
        """GLM Formula

        Parameters
        ----------
        regression_variables: list
            List of variables used in the regression model.

        Returns
        -------
        f: str
            Formula for the regression model. 
        """
        f = self.target_variable[0]
        if len(self.target_variable)==2:
            f += ' + ' + self.target_variable[1]
        if intercept:
            f += '~1'
        else:
            f += '~0'
        for variable in regression_variables:
            if 'nominal' in variable:
                f += '+C({nominal_variable})'.format(nominal_variable=variable)
            else:
                f += '+' + variable
        return f

    def gmm_information_criteria(self, df=None, which="bic", gmm_scaling=True):
        """GMM Information Criteria, either AIC or BIC.

        Parameters
        ----------
        df: pandas.DataFrame
            Dataframe to calculate the BIC for. If None, the training dataframe is used.
        which: str
            Which information criterion to calculate. Default is "bic".
        gm_scaling: bool
            Whether to scale the data before calculating the BIC. Default is True.
        
        Returns
        -------
        bic: float
            Bayesian Information Criterion for the GMM.
        """
        if which not in ["bic", "aic"]:
            print("Invalid information criterion.")
            return None 
        if df is None:
            df_train = self.df_train.copy()
            if gmm_scaling:
                df_train[self.continuous_variables + self.discrete_variables] = self.scale_data(df_train)
            if self.valid_set:
                df_valid = self.df_valid.copy()
                if gmm_scaling:
                    df_valid[self.continuous_variables + self.discrete_variables] = self.scale_data(df_valid)
            if self.test_set:
                df_test = self.df_test.copy()
                if gmm_scaling:
                    df_test[self.continuous_variables + self.discrete_variables] = self.scale_data(df_test)
            if self.valid_set and self.test_set:
                if which == "bic":
                    return self.gmm.bic(df_train[self.cluster_variables]), self.gmm.bic(df_valid[self.cluster_variables]), self.gmm.bic(df_test[self.cluster_variables])
                else:
                    return self.gmm.aic(df_train[self.cluster_variables]), self.gmm.aic(df_valid[self.cluster_variables]), self.gmm.aic(df_test[self.cluster_variables])
            elif self.valid_set:
                if which == "bic":
                    return self.gmm.bic(df_train[self.cluster_variables]), self.gmm.bic(df_valid[self.cluster_variables])
                else:
                    return self.gmm.aic(df_train[self.cluster_variables]), self.gmm.aic(df_valid[self.cluster_variables])
            elif self.test_set:
                if which == "bic":
                    return self.gmm.bic(df_copy[self.cluster_variables]), self.gmm.bic(df_test[self.cluster_variables])
                if which == "aic":
                    return self.gmm.aic(df_copy[self.cluster_variables]), self.gmm.aic(df_test[self.cluster_variables])
            else:
                if which == "bic":
                    return self.gmm.bic(df_train[self.cluster_variables])
                else:
                    return self.gmm.aic(df_train[self.cluster_variables])
        else:
            df_copy = df.copy()
            if gmm_scaling:
                df_copy[self.continuous_variables + self.discrete_variables] = self.scale_data(df_copy)
            if which == "bic":
                return self.gmm.bic(df_copy[self.cluster_variables])
            else:
                return self.gmm.aic(df_copy[self.cluster_variables])

    def get_glm_marginal_effects(self): # Add kwargs
        """GLM Marginal Effects
        Is there a need for this since we can just call get_margeff from statsmodels? Actually probably fine since we need to loop over it
        """
        if self.glm_offset is not None or self.glm_exposure is not None:
            print("offset, exposure and weights (var_weights and freq_weights) are not supported by margeff.")
        for i in range(self.n_components): # Cluster i GLM
            print("\nCluster {}".format(i))
            glm = getattr(self, 'clust' + str(i) + '_glm')
            print(glm.get_margeff(at='overall', method='dydx', atexog=None, dummy=False, count=False).summary()) 

    def get_pca_summary(self, verbose=True, savefig=False, figloc=None):
        """PCA Summary

        Parameters
        ----------
        verbose : bool
            Print results. Default is True.
        savefig : bool
            Save figure. Default is False.
        figloc : str
            Location to save the figure. Default is None.
        """
        if self.use_pca:
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
            # Explained Variance Ratio
            axs[0].set_xlabel("Components")
            axs[0].set_ylabel("PCA explained variance ratio")
            axs[0].set_ylim(0, 1)
            axs[0].set_xticks(np.arange(1, self.pca.n_components_ + 1))
            axs[0].plot(np.arange(1, self.pca.n_components_ + 1), self.pca.explained_variance_ratio_, "+", linewidth=2)
            # Cumulative Explained Variance Ratio
            axs[1].set_xlabel("Components")
            axs[1].set_ylabel("Cumulative PCA explained variance ratio")
            axs[1].set_ylim(0, 1)
            axs[1].set_xticks(np.arange(1, self.pca.n_components_ + 1))
            axs[1].plot(np.arange(1, self.pca.n_components_ + 1), np.cumsum(self.pca.explained_variance_ratio_), "+", linewidth=2)
            # Eigenvalue plot
            axs[2].set_xlabel("Components")
            axs[2].set_ylabel("PCA Eigenvalues")
            #axs[2].set_ylim(0,1)
            axs[2].set_xticks(np.arange(1, self.pca.n_components_ + 1))
            axs[2].plot(np.arange(1, self.pca.n_components_ + 1), self.pca.explained_variance_, "+", linewidth=2)
            if savefig:
                plt.savefig(fname=figloc + 'pca_summary')
            if verbose:
                plt.show()
            else:
                plt.close(fig)
        else:
            print("PCA not performed.")

    def get_sampling_summary(self):
        """Sampling Summary
        """
        if self.use_sampling:
            df_copy = pd.DataFrame(np.zeros((self.df_train.shape[0], self.n_components)))
            df_copy.set_index(self.df_train.index.values, inplace=True, drop=True)
            for i in range(self.n_components):
                clust_i_glm_df_sample = getattr(self, 'clust' + str(i) + '_glm_df_sample')
                clust_i_glm_df_sample_index = clust_i_glm_df_sample.index.values
                print("Number of training observations used on wrong glm:",
                    sum(self.predict_gmm(self.df_train.loc[clust_i_glm_df_sample_index], prob=False, return_both=False, gmm_scaling=self.gmm_scaling)!=i))
                df_copy.loc[clust_i_glm_df_sample_index, i] += 1

            print("\nObservations not used in training:", sum(np.sum(df_copy, axis=1) == 0))
            for i in range(1, self.n_components + 1):
                print("\nObservations used " + str(i) + " times:", sum(np.sum(df_copy, axis=1) == i))
                print("\nObservations used " + str(i) + " times or more:", sum(np.sum(df_copy, axis=1) >= i))
            self.sampling_index_info = df_copy
        else:
            print("Sampling not performed")

    def Royston_QQ_plot(self, df, scale=False):
        df_copy = df.copy()
        if scale:
            df_copy = self.scale_data(df_copy, list_of_vars=df.columns)
        X = np.asarray(df_copy)
        n, p = X.shape
        SX = np.cov(X, rowvar=False)
        mean_X = np.mean(X, axis=0)
        # Calculate Mahalanobis distance
        D2 = np.sort(np.array([mahalanobis(X[i, :], mean_X, np.linalg.inv(SX))**2 for i in range(n)]))
        # Generate theoretical chi-square values
        chi_square_quantiles = chi2.ppf(np.linspace(0.01, 0.99, n), df=p)
        # Plotting Mahalanobis Distance vs. Chi-Square
        plt.figure(figsize=(10, 6))
        plt.scatter(D2, chi_square_quantiles)
        plt.plot([0, min(max(D2),max(chi_square_quantiles))], [0, min(max(D2),max(chi_square_quantiles))], color='black')  # Line with slope=1, intercept=0
        plt.title("Mahalanobis Distance vs Chi-Square")
        plt.xlabel("Mahalanobis Distance")
        plt.ylabel("Chi-Square")
        plt.grid(True)
        plt.show()

    def Royston_H_test(self, df, scale=False):
        df_copy = df.copy()
        if scale:
            df_copy = self.scale_data(df_copy, list_of_vars=df.columns)
        X = np.asarray(df_copy)
        n, p = X.shape
        z = pd.DataFrame(np.zeros((p,1)))
        x = np.log(n)
        g = 0
        m = -1.5861 - 0.31082*x - 0.083751*x**2 + 0.0038915*x**3
        s = np.exp(-0.4803 -0.082676*x + 0.0030302*x**2)
        for i in range(p):
            a2 = X[:,i]
            if kurtosis(a2) > 3:#Shapiro-Francia test is better for leptokurtic samples
                w = sfrancia.shapiroFrancia(array=a2)['statistics W']
            else:#Shapiro-Wilk test is better for platykurtic samples
                w = shapiro(a2)[0]
            z.loc[i] = ((np.log(1 - w)) + g - m)/s
        u = 0.715
        v = 0.21364 + 0.015124*(np.log(n))**2 - 0.0018034*(np.log(n))**3
        l = 5
        C = np.corrcoef(X, rowvar=False) #correlation matrix
        NC = (C**l)*(1 - (u*(1 - C)**u)/v)#transformed correlation matrix
        T = np.sum(np.sum(NC)) - p# %total
        mC = T/(p**2 - p)# %average correlation
        edf = p/(1 + (p - 1)*mC)# %equivalent degrees of freedom
        Res = pd.DataFrame(np.zeros((p,1)))
        Res=(norm.ppf((norm.cdf( - z))/2))**2
        RH = (edf*(np.sum(Res)))/p
        pv= 1-chi2.cdf(RH, edf)
        return RH, pv

    def mardia_test(self, df, scale=False, tol=1e-25):
        # Ensure data is a DataFrame
        df_copy = df.copy()
        if scale:
            df_copy = self.scale_data(df_copy, list_of_vars=df.columns)
        X = np.asarray(df_copy)
        n, p = X.shape
        S = ((n - 1) / n) * np.cov(X, rowvar=False)
        # Compute the Mahalanobis distances
        D = np.dot(np.dot(X, np.linalg.inv(S + np.eye(p) * tol)), X.T)
        D_diag = np.diag(D)
        # Compute Mardia's measures
        g1p = np.sum(D**3) / n**2
        g2p = np.sum(D_diag**2) / n
        edf = p * (p + 1) * (p + 2) / 6 # equivalent degrees of freedom
        # Compute skewness
        skew = n * g1p / 6
        p_skew = 1 - chi2.cdf(skew, edf)
        # Compute kurtosis
        kurt = (g2p - p * (p + 2)) * np.sqrt(n / (8 * p * (p + 2)))
        p_kurt = 2 * (1 - norm.cdf(abs(kurt)))
        return skew, p_skew, kurt, p_kurt
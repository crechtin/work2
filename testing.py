import streamlit as st
from __future__ import division, print_function
import pandas as pd
from scipy import stats
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from pandas import ExcelWriter
from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from IPython.display import display
from matplotlib.pyplot import figure
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import datetime

#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#Cumulative sum algorithm (CUSUM) to detect abrupt changes in data."""

from __future__ import division, print_function
import numpy as np
def detect_cusum(x, threshold=1, drift=0, ending=False, show=True, ax=None):
    x = np.atleast_1d(x).astype('float64')
    gp, gn = np.zeros(x.size), np.zeros(x.size)
    ta, tai, taf = np.array([[], [], []], dtype=int)
    tap, tan = 0, 0
    amp = np.array([])
    # Find changes (online form)
    for i in range(1, x.size):
        s = x[i] - x[i-1]
        gp[i] = gp[i-1] + s - drift  # cumulative sum for + change
        gn[i] = gn[i-1] - s - drift  # cumulative sum for - change
        if gp[i] < 0:
            gp[i], tap = 0, i
        if gn[i] < 0:
            gn[i], tan = 0, i
        if gp[i] > threshold or gn[i] > threshold:  # change detected!
            ta = np.append(ta, i)    # alarm index
            tai = np.append(tai, tap if gp[i] > threshold else tan)  # start
            gp[i], gn[i] = 0, 0      # reset alarm
    # THE CLASSICAL CUSUM ALGORITHM ENDS HERE

# Estimation of when the change ends (offline form)
if tai.size and ending:
    _, tai2, _, _ = detect_cusum(x[::-1], threshold, drift, show=False)
    taf = x.size - tai2[::-1] - 1
    # Eliminate repeated changes, changes that have the same beginning
    tai, ind = np.unique(tai, return_index=True)
        ta = ta[ind]
        # taf = np.unique(taf, return_index=False)  # corect later
        if tai.size != taf.size:
            if tai.size < taf.size:
                taf = taf[[np.argmax(taf >= i) for i in ta]]
            else:
                ind = [np.argmax(i >= ta[::-1])-1 for i in taf]
                ta = ta[ind]
                tai = tai[ind]
    # Delete intercalated changes (the ending of the change is after
    # the beginning of the next change)
    ind = taf[:-1] - tai[1:] > 0
        if ind.any():
            ta = ta[~np.append(False, ind)]
            tai = tai[~np.append(False, ind)]
            taf = taf[~np.append(ind, False)]
# Amplitude of changes
amp = x[taf] - x[tai]

if show:
    _plot(x, threshold, drift, ending, ax, ta, tai, taf, gp, gn)
    
    return ta, tai, taf, amp


def _plot(x, threshold, drift, ending, ax, ta, tai, taf, gp, gn):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        t = range(x.size)
        ax1.plot(t, x, 'b-', lw=2)
        if len(ta):
            ax1.plot(tai, x[tai], '>', mfc='g', mec='g', ms=10,
                     label='Start')
                     if ending:
                         ax1.plot(taf, x[taf], '<', mfc='g', mec='g', ms=10,
                                  label='Ending')
                     ax1.plot(ta, x[ta], 'o', mfc='r', mec='r', mew=1, ms=5,
                              label='Alarm')
ax1.legend(loc='best', framealpha=.5, numpoints=1)
ax1.set_xlim(-.01*x.size, x.size*1.01-1)
ax1.set_xlabel('Data #', fontsize=14)
ax1.set_ylabel('Amplitude', fontsize=14)
ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
yrange = ymax - ymin if ymax > ymin else 1
    ax1.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
    ax1.set_title('Time series and detected changes ' +
                  '(threshold= %.3g, drift= %.3g): N changes = %d'
                  % (threshold, drift, len(tai)))
                  ax2.plot(t, gp, 'y-', label='+')
                      ax2.plot(t, gn, 'm-', label='-')
                      ax2.set_xlim(-.01*x.size, x.size*1.01-1)
                      ax2.set_xlabel('Data #', fontsize=14)
                      ax2.set_ylim(-0.01*threshold, 1.1*threshold)
                      ax2.axhline(threshold, color='r')
                      ax1.set_ylabel('Amplitude', fontsize=14)
                      ax2.set_title('Time series of the cumulative sums of ' +
                                    'positive and negative changes')
                                    ax2.legend(loc='best', framealpha=.5, numpoints=1)
                                        plt.tight_layout()
                                        plt.show()

#This code creates the functions to read the data from the Auto
#variation and stability analysis tempalte and remove predictors that
#do not meet our criteria

#outputs include the cleaned data frame, response, meta,
#and predictor metrics called ddf

#This block does NOT statistically clean the response varaibles
#..that will take place in the next code block

#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
def read(filename):
    '''
        Read file in Auto Variation/Stability Analysis Tempalte format and keeps timestamp
        in 0 position, reel number in 1 poistion, grade in 2 poistion, and break indication
        in 3 position.
        '''
    df = pd.read_excel(filename, sheet_name=0, header=1)
    meta = pd.read_excel(filename, sheet_name=2, header=0)
    response = list(meta["Response Name"].unique())
    grade = meta.columns[1]
    
    return df, meta, response, grade

def remove_na(df, response):
    """
        Remove any blank column from position 4 onward (after timestamp (0), reel number (1),
        grade code (2), break indication(3))
        Remove any row that has a missing value
        """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(response, list)
    df[df.columns[4::]].dropna(axis=1, how="all")
    df = pd.DataFrame.dropna(df, subset=response)
    return df

def remove_string_in_num(df, response):
    """
        Remove any row that has a string component within a numerical value
        """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(response, list)
    y = df[response]
    del_rows = []
    for i, dtype in enumerate(y.dtypes):
        if dtype not in ["float64", "int64"]:
            col = y.iloc[:,i]
            #             results = col.string.contains("[A-Z][a-z][]", regex=True)
            for j, row in enumerate(col):
                if type(row) == str and not row.replace('.','',1).isdigit():
                    del_rows.append(j)
    df = df.drop(del_rows)
    return df

def remove_categorical(df, response):
    """
        Remove categorical variables
        Change the type of all numerical variables to "float64"
        """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(response, list)
    y = df[response]
    df2 = df.copy()
    del_cols = []
    for col_name, col in y.iteritems():
        try:
            col = col.astype("float64")
            df2[col_name] = col
        except:
            del_cols.append(col_name)
    df2 = df2.drop(del_cols, axis=1)
    return df2

def create_metrics(df2):
    '''
        calculated metrics about predcitors so that "unusable" predictors
        can be removed from the dataframe
        '''
    n=5
    metrics_dict = {
        "Tag ID": list(df2.columns[n::]),
        "Total Number of Records": [float("nan")]*len(df2.columns[n::]),
        "Number of Outliers (IQR)": [float("nan")]*len(df2.columns[n::]),
        "Number of Zeros": [float("nan")]*len(df2.columns[n::]),
        "Mean": [float("nan")]*len(df2.columns[n::]),
        "Standard Deviation": [float("nan")]*len(df2.columns[n::]),
        "Coefficient of Variation": [float("nan")]*len(df2.columns[n::]),
        "Number of Blank/String Values": [float("nan")]*len(df2.columns[n::]),
        "Percentage of Unusable Records": [float("nan")]*len(df2.columns[n::]),
    "Use as Predictor?": [float("nan")]*len(df2.columns[n::]),
        "Reason to Exclude": [float("nan")]*len(df2.columns[n::])
    }
    

    metrics_df = pd.DataFrame(metrics_dict)

    metrics_df["Total Number of Records"] = [len(df2[col]) for col in df2[df2.columns[n::]]]


num_outliers_z_ls, num_outliers_iqr_ls = [], []

    for col in df2[df2.columns[n::]]:
        # IQR
        q1, q3 = np.nanpercentile(df2[col], 25), np.nanpercentile(df2[col], 75)
        iqr = q3 - q1
        cutoff_iqr = iqr * 1.5
        lower_iqr, upper_iqr = q1 - cutoff_iqr, q3 + cutoff_iqr
        outliers_iqr = [x for x in df2[col] if x < lower_iqr or x > upper_iqr]
        num_outliers_iqr = len(outliers_iqr)
        num_outliers_iqr_ls.append(num_outliers_iqr)

metrics_df["Number of Outliers (IQR)"] = num_outliers_iqr_ls
metrics_df["Number of Zeros"] = [(df2[col] == 0).astype(int).sum() for col in df2[df2.columns[n::]]]
metrics_df["Mean"] = [np.nanmean(df2[col]) for col in df2[df2.columns[n::]]]
metrics_df["Standard Deviation"] = [np.nanstd(df2[col]) for col in df2[df2.columns[n::]]]
metrics_df["Coefficient of Variation"] = [np.nanstd(df2[col])/np.nanmean(df2[col]) for col in df2[df2.columns[n::]]]
metrics_df["Number of Blank/String Values"] = [df2[col].isna().sum() for col in df2[df2.columns[n::]]]

unusable_ls = ["Number of Outliers (IQR)", "Number of Zeros", "Number of Blank/String Values"]

metrics_df["Percentage of Unusable Records"] = 100*metrics_df[unusable_ls].sum(axis=1)/metrics_df["Total Number of Records"]

metrics_df["Use as Predictor?"] = np.where((metrics_df["Standard Deviation"] > 0)
                                           & (metrics_df["Percentage of Unusable Records"] <= 25)
                                           ,True,False
                                           )
    new_metrics_df = metrics_df.copy()
    for i, row in new_metrics_df.iterrows():
        reason = ""
        if row["Use as Predictor?"] == False:
            if row["Standard Deviation"] == 0:
                reason += "SD=0; "
            if 100*row["Number of Outliers (IQR)"]/row["Total Number of Records"] >= 12.5:
                reason += "Too many outliers; "
            if 100*row["Number of Zeros"]/row["Total Number of Records"] >= 12.5:
                reason += "Too many zeros; "
            if 100*row["Number of Blank/String Values"]/row["Total Number of Records"] >= 12.5:
                reason += "Too many blanks/strings; "
            new_metrics_df["Reason to Exclude"].iloc[i] = reason
    
    return new_metrics_df


def cleaned_data_frame(df, response, met):
    '''
        Creates new data frame with predictors removed that do not meet the critera from
        the predictor metrics requirements (create metrics function)
        '''
    return df.drop(list(met[(met["Use as Predictor?"]==False)]["Tag ID"]), axis=1)

def read_and_prep_predictors(filename):
    df, meta, response,grade=read(filename)
    df=remove_na(df, response)
    df=remove_string_in_num(df, response)
    df2=remove_categorical(df, response)
    ddf=create_metrics(df2)
    data=cleaned_data_frame(df2, response, ddf)
    return data, meta, response, grade

def export_data(df, filename):
    return df.to_csv(filename)

def remove_na_2(data):
    for col in  data.columns[3:]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
            return data.dropna()

#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------

##Clean Response Variable by Grade

#This separates and cleans by grade. Outputs are dataframes for each grade in a list called g_df_ls,
# also a list of supported_grades, and the origional meta,response,data

def separate_by_grade(df, grade):
    """
        Separate the dataframe by grade
        Return a list of dataframes, each with one grade
        """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(grade, str)
    
    g_df_ls = [g for _, g in df.groupby([grade])]
    
    return g_df_ls

def remove_extreme(g_df_ls, grade, response):
    """
        Remove any row that has a value with z-score outside of +3/-3 threshold (2 iterations)
        Return a list of dataframes, each with one grade AND a list of % size differences
        """
    assert isinstance(g_df_ls, list)
    assert isinstance(response, list)
    new_g_df_ls = []
    size_diff_ls = []
    for df in g_df_ls:
        grade_id = str(df[grade].iloc[0])
        new_df = df.copy()
        col_zscore_ls = []
        for i in range(2):
            df_all = new_df
            # # Calculate Z-score using Scipy Package
            # zscores = new_df[response].apply(zscore)
            # zscores = zscores.add_suffix("_zscore")
            # df_all = pd.concat([new_df, zscores], axis=1)
            
            
            # Calculate Z-score by hand
            for col in response:
                col_zscore = col + "_zscore"
                df_all[col_zscore] = (df_all[col] - df_all[col].mean()) / df_all[col].std(ddof=0)
                col_zscore_ls.append(col_zscore)
            
            # df_all.to_csv("zscores-" + str(i) + ".csv")
            
            for z_col in col_zscore_ls:
                # print(col)
                df_all = df_all.loc[(df_all[z_col] >= -3) & (df_all[z_col] <= 3)]
            # df_all = df_all[df_all[z_col] >= -3]
            # df_all = df_all[df_all[z_col] <= 3]
            
            new_df = df_all.drop(col_zscore_ls, axis=1)
        if len(new_df) > 0:
            new_g_df_ls.append(new_df)
            size_diff = (len(new_df) - len(df))/len(df)
            size_diff_ls.append(size_diff)
        else:
            print(grade_id + " has been eliminated from analysis (no data after z-score cleaning)")
            pass


return new_g_df_ls, size_diff_ls

def filter_grade(g_df_ls, size_diff_ls, meta, grade, threshold=180):
    """
        Filter out grades that have number of rows less than threshold (default 400)
        Filter out grades not of interest
        Return a list of dataframes, each with one grade AND a list of % size differences
        """
    assert isinstance(g_df_ls, list)
    assert isinstance(size_diff_ls, list)
    assert isinstance(meta, pd.DataFrame)
    assert isinstance(grade, str)
    new_g_df_ls, new_size_diff_ls = [], []
    
    for i, df in enumerate(g_df_ls):
        if len(list(set(df[grade]))) == 1:
            try:    # grades are int
                grade_id = int(list(set(df[grade]))[0])
                grades_ls = [int(g) for g in list(meta[grade].unique())]
            except:    # grades are str
                grade_id = str(list(set(df[grade]))[0])
                grades_ls = [str(g) for g in list(meta[grade].unique())]
            if len(df) >= threshold and grade_id in grades_ls:
                new_g_df_ls.append(df)
                new_size_diff_ls.append(size_diff_ls[i])
            else:
                
                pass
else:
    
    pass
        supported_grades=[list(new_g_df_ls[i][new_g_df_ls[0].columns[2]])[0] for i in range(0, len(new_g_df_ls))]
return new_g_df_ls, new_size_diff_ls, supported_grades

def clean_separate_grades(data,grade,meta,response):
    g_df_ls=separate_by_grade(data, grade)
    new_g_df_ls, size_diff_ls= remove_extreme(g_df_ls, grade, response)
    g_df_ls, size_diff_ls, supported_grades = filter_grade(new_g_df_ls, size_diff_ls, meta, grade)
    return g_df_ls, supported_grades, meta,response, data

#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------

#Begin EDA

#This code block defines some statistical testing.
#Test 1 is just two strings compared for mean and variation diff
#Test 2 compares two data frames where returned have both mean and varaition differences
#Test 3 compared two data frames where returned has either mean or variaiton differences

def test1(a,b,alpha=0.05):
    from scipy import stats
    t2, p2 = stats.ttest_ind(a,b)
    stat1, p3 = stats.levene(a,b)
    print("t = "+ str(round(t2,3)))
    print("p_t = "+ str(p2))
    print("Levene Stat = "+ str(round(stat1,3)))
    print("p_levene = "+ str(p3))
    print("A p-value less than the significance level leads to null hypothesis rejection.")
    if p2 > alpha:
        print("The means of the two tested distributions are NOT statistically different.")
    else:
        print("The means of the two tested distributions are different and statistically significant.")
if p3 > alpha:
    print("The variations of the two tested distributions are NOT statistically different.")
    else:
        print("The variations of the two tested distributions are different and statistically significant.")

#P
##Low p value (less than 0.05), we reject the null hypothesis and thus
#proves that the mean of the two distributions are different and statistically significant.

#Levene
##Low p value (less than 0.05), we reject the null hypothesis and thus
#proves that the variance of the two distributions are different and statistically significant.

#Test two is AND (mean and variance are different)
def test2(df1,df2,grade_tag,grade,alpha=0.05):
    from scipy import stats
    sig=[]
    for i in range(4,len(df1.columns)):
        t2, p2 = stats.ttest_ind(df1[df1[grade_tag]==grade].iloc[: , i],df2[df2[grade_tag]==grade].iloc[: , i])
        stat1, p3 = stats.levene(df1[df1[grade_tag]==grade].iloc[: , i],df2[df2[grade_tag]==grade].iloc[: , i])
        if (p2 < alpha) & (p3 < alpha):
            sig.append((str(df1.columns[i]), df1[df1.columns[i]].mean(),df2[df2.columns[i]].mean(),df1[df1.columns[i]].std(),df2[df2.columns[i]].std()))
        else:
            pass
    return pd.DataFrame(sig, columns=['Tag', 'Mean: Time 1', 'Mean: Time 2', 'StDev: Time 1', 'StDev: Time 2'])

#Test three is OR (variance or mean are different)
def test3(df1,df2,grade_tag,grade,alpha=0.05):
    from scipy import stats
    sig=[]
    for i in range(4,len(df1.columns)):
        t2, p2 = stats.ttest_ind(df1[df1[grade_tag]==grade].iloc[: , i],df2[df2[grade_tag]==grade].iloc[: , i])
        stat1, p3 = stats.levene(df1[df1[grade_tag]==grade].iloc[: , i],df2[df2[grade_tag]==grade].iloc[: , i])
        if (p2 < alpha) | (p3 < alpha):
            sig.append((str(df1.columns[i]), df1[df1.columns[i]].mean(),df2[df2.columns[i]].mean(),df1[df1.columns[i]].std(),df2[df2.columns[i]].std()))
        else:
            pass
    return pd.DataFrame(sig, columns=['Tag', 'Mean: Time 1', 'Mean: Time 2', 'StDev: Time 1', 'StDev: Time 2'])

#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#Statistical Control Limit Analysis

def scla(g_df_ls, grade, meta):
    assert isinstance(g_df_ls, list)
    assert isinstance(grade, str)
    assert isinstance(meta, pd.DataFrame)
    scla_g_df_ls, scla_filename_ls = [], []
    for df in g_df_ls:
        grade_dict = {}
        try:    # grades are int
            grade_id = int(list(set(df[grade]))[0])
            response = sorted(meta.loc[meta[grade].astype(int) == grade_id]["Response Name"])
            grade_is_int = True
        except:    # grades are str
            grade_id = str(list(set(df[grade]))[0])
            response = sorted(meta.loc[meta[grade].astype(str) == grade_id]["Response Name"])
            grade_is_int = False
        
        # Overall Moving Range Average
        mr_bar_ls = []
        for col in response:
            mr_ls = list(df[col].rolling(2).max() - df[col].rolling(2).min())
            mr_bar = np.mean(mr_ls[1:])
            mr_bar_ls.append(mr_bar)
        
        # Mean
        mean_ls = [np.mean(df[col]) for col in response]
        
        # Standard Deviation
        sd_ls = [np.std(df[col]) for col in response]
        
        # Statistical Upper Control Limit
        stat_ucl_ls = list(np.array(mean_ls) + np.array(mr_bar_ls)*2.66)
        
        # Statistical Lower Control Limit
        stat_lcl_ls = list(np.array(mean_ls) - np.array(mr_bar_ls)*2.66)
        
        # Stability %
        stability_pct_ls = []
        for i, col in enumerate(response):
            actual = np.array(df[col])
            upper = np.array([stat_ucl_ls[i]] * len(df))
            lower = np.array([stat_lcl_ls[i]] * len(df))
            inside_cnt = ((actual > lower) & (actual < upper)).sum()
            stability_pct = inside_cnt / len(df) * 100
            stability_pct_ls.append(stability_pct)
    
        # CPK Upper (If Upper Spec Limit available)
        # CPK Lower (If Lower Spec Limit available)
        cpk_upper_ls, cpk_lower_ls, usl_ls, lsl_ls = [], [], [], []
        
        for i, col in enumerate(response):
            if grade_is_int:
                usl = float(meta.loc[np.logical_and(meta["Response Name"] == col, meta[grade].astype(int) == grade_id)]["Upper Spec Limit"])
                lsl = float(meta.loc[np.logical_and(meta["Response Name"] == col, meta[grade].astype(int) == grade_id)]["Lower Spec Limit"])
            else:
                usl = float(meta.loc[np.logical_and(meta["Response Name"] == col, meta[grade].astype(str) == grade_id)]["Upper Spec Limit"])
                lsl = float(meta.loc[np.logical_and(meta["Response Name"] == col, meta[grade].astype(str) == grade_id)]["Lower Spec Limit"])
            
            if not math.isnan(usl):
                cpk_upper = abs(mean_ls[i] - usl)/(3 * sd_ls[i])
            else:
                cpk_upper = float('NaN')
            if not math.isnan(lsl):
                cpk_lower = abs(mean_ls[i] - lsl)/(3 * sd_ls[i])
            else:
                cpk_lower = float('NaN')
            usl_ls.append(usl)
            lsl_ls.append(lsl)
            cpk_upper_ls.append(cpk_upper)
            cpk_lower_ls.append(cpk_lower)
        
        # Customer Upper Control Limit, Lower Control Limit and Target
        if grade_is_int:
            cus_ucl_ls = [float(meta.loc[np.logical_and(meta["Response Name"] == col, meta[grade].astype(int) == grade_id)]["Upper Control Limit"]) for col in response]
            cus_lcl_ls = [float(meta.loc[np.logical_and(meta["Response Name"] == col, meta[grade].astype(int) == grade_id)]["Lower Control Limit"]) for col in response]
            target_ls = [float(meta.loc[np.logical_and(meta["Response Name"] == col, meta[grade].astype(int) == grade_id)]["Target"]) for col in response]

                else:
                    cus_ucl_ls = [float(meta.loc[np.logical_and(meta["Response Name"] == col, meta[grade].astype(str) == grade_id)]["Upper Control Limit"]) for col in response]
                    cus_lcl_ls = [float(meta.loc[np.logical_and(meta["Response Name"] == col, meta[grade].astype(str) == grade_id)]["Lower Control Limit"]) for col in response]
                    target_ls = [float(meta.loc[np.logical_and(meta["Response Name"] == col, meta[grade].astype(str) == grade_id)]["Target"]) for col in response]

grade_dict["Grade"] = [grade_id] * len(response)
grade_dict["Response"] = response
grade_dict["Stability %"] = stability_pct_ls
grade_dict["Mean"] = mean_ls
grade_dict["Target"] = target_ls
#         grade_dict["SD"] = sd_ls
grade_dict["Stat UCL"] = stat_ucl_ls
grade_dict["Cus UCL"] = cus_ucl_ls
grade_dict["Stat LCL"] = stat_lcl_ls
grade_dict["Cus LCL"] = cus_lcl_ls

grade_dict["USL"] = usl_ls
grade_dict["LSL"] = lsl_ls

grade_dict["CPK Upper"] = cpk_upper_ls
grade_dict["CPK Lower"] = cpk_lower_ls

    scla_g_df = pd.DataFrame.from_dict(grade_dict)
    scla_g_df = np.round(scla_g_df, decimals=2)
    scla_g_df_ls.append(scla_g_df)
    scla_all=pd.concat(scla_g_df_ls)
    
    
    return scla_g_df_ls, scla_all

#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#EDA Continued...

#Defines "good" time frame as data between a specified diff c away from mean or target
#test_frames[i][3] give "good" data frame
#test_frames[i][5] give "bad" data frame relating to "good"
#len(test_frames)==#grades*#response var

def test_frames(data, grade, response, supported_grades, scla_all):
    test_frames=[]
    for responses in response:
        for supported_grade in supported_grades:
            s=data[(data[grade]==supported_grade)][responses].std()
            me=data[(data[grade]==supported_grade)][responses].mean()
            c=0.5 #number of standard deviations away from mean or target
            t=float(scla_all[(scla_all["Grade"]==supported_grade)&((scla_all["Response"]==responses))]['Target'])
            m=me #or change m=t for calculations to be centered around customer target
            #good = data[(data[grade]==supported_grade) & (data[responses]>=(m-c*s)) & (data[responses]<=(m+c*s))]
            good = data[(data[grade]==supported_grade) & (data["Timestamp"]>'2019-10-01 00:00:00') & (data["Timestamp"]<'2019-10-15 00:00:00')]
            #bad = data[(data[grade]==supported_grade) & ((data[responses]<(m-c*s)) | (data[responses]>(m+c*s)))]
            bad = data[(data[grade]==supported_grade) & (data["Timestamp"]>'2019-10-18 00:00:00') & (data["Timestamp"]<'2019-10-31 00:00:00')]
            l = math.floor(float(len(good)/2))
            test_frames.append((supported_grade, responses, 'good',good , 'bad', bad, 'gooda', good[:l], 'goodb', good[l:]))
    return test_frames

def Run_Test_Mean_And_Var(response, supported_grades, test_frames, grade_tag):
    mandv=[]
    for i in range(0,len(test_frames)):
        AB = test2(test_frames[i][3], test_frames[i][5], grade_tag, test_frames[i][0])
        AA = test2(test_frames[i][7], test_frames[i][9], grade_tag, test_frames[i][0])
        ABmAA = AB[~AB['Tag'].isin(list(AA["Tag"]))]
        mandv.append((test_frames[i][0], test_frames[i][1], ABmAA))
    return mandv

def Run_Test_Mean_Or_Var(response, supported_grades, test_frames, grade_tag):
    morv=[]
    for i in range(0,len(test_frames)):
        AB = test3(test_frames[i][3], test_frames[i][5], grade_tag, test_frames[i][0])
        AA = test3(test_frames[i][7], test_frames[i][9], grade_tag, test_frames[i][0])
        ABmAA = AB[~AB['Tag'].isin(list(AA["Tag"]))]
        maorv.append((test_frames[i][0], test_frames[i][1], ABmAA))
    return morv

def getlist(test_result_tuple_list,grade_of_interest,response):
    for i in test_result_tuple_list:
        if (i[0]==grade_of_interest) & (i[1]==response):
            f=i[2]
        else:
            pass
    return f


st.write('HELLO to the OPTIX team from the world of CLOUD COMPUTING. YAASSS!')

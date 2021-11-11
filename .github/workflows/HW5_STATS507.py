# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# Question 0
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
from os.path import exists
file = 'tooth_growth.feather'
if exists(file):
    tg_data = pd.read_feather(file)
else:
    tg_data = sm.datasets.get_rdataset('ToothGrowth')
    tg_data = tg_data.data
    tg_data.to_feather(file)
data = tg_data
tg_data.head()

data['logY'] = np.log(data['len'])
cate = data['dose'].unique()
data['suppd'] = data['supp'].apply(lambda x: 1 if x == 'VC' else 0) 
data['X_dose'] = pd.Categorical(data['dose'])
X_1 = data['suppd']
X_2 = data['X_dose']
LR = smf.ols(formula = 'logY ~ suppd + X_dose + suppd: X_dose', data = data).fit()
LR.summary()

# calculation R square and adjusted R square
Y_hat = LR.predict()
SSE = np.sum((data['logY'] - Y_hat)**2)
SSR = np.sum((Y_hat - np.mean(data['logY']))**2)
SSY = SSE + SSR
R_square = SSR / (SSY)
R_square

# +
# degree freedom of  SSE : 54, degree freedom of SSY : 59
Adjusted_Rsquare = 1- (SSE / 54) / (SSY / 59)
Adjusted_Rsquare

# Comments 
## Computation results of Rsquare and adjusted Rsquare are the same as shown in result object.
# -

# Question 1
## (a)
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import warnings
import pickle
import patsy
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")
# demography data
demogr = pickle.load(open('./demogr_total.pkl','rb'))
# oral health data
orl_halh = pickle.load(open('./oral_dentition_total.pkl','rb')) 

data_all = pd.merge(demogr, orl_halh, on = 'id')
orl_halh.head()

# +
# I pick tooth 1: ohx01TC as response variable Y.
## From the expression of question, we should use 'age' as predictor and Y as response variable to build logistic regression.
## To prevent ineffective fit, I just use people aged above 12 since people below 12 have no permanent tooth.

data_all = data_all.loc[data_all['age'] > 12, :]
data_all['ohx01tc_y'] = data_all['ohx01tc'].apply(lambda x: 1 if x == 2 else 0)
X = data_all['age']
kmin = np.min(X)
kmax = np.max(X)
print('minimum age is :', kmin,'\n', 'maximum age is :', kmax)
# age [1, 80], test for a few sets of knots, make selection of best set by 'aic' index.
knot_dict = {}
# random seed to certain generated results
np.random.seed(10)
for i in range(10):
    knot_dict[str(i)] = np.random.randint(15, 75, 5)
knot_dict['10'] = [15, 25, 35, 45, 55]
aiclist = []
for j in knot_dict.keys():
    knots = knot_dict[j]
    # here I groupby age first to get average permanent tooth probability in each age group, so I use this as y and age as x
    # to build a ols regression, which is equal to logit.
    logitm = smf.ols('ohx01tc ~ patsy.bs(age, knots = knots, degree = 3)', data = data_all).fit()
    aiclist.append(logitm.aic)
minaic = min(aiclist)
min_index = aiclist.index(min(aiclist))
knots = knot_dict[str(min_index)] # knots of minimum aic model
logitmodel = smf.logit(' ohx01tc_y ~ patsy.bs(age, knots = knots, degree = 3)', data = data_all).fit()
logitmodel.summary()
print('Minimum Aic is :', min(aiclist),'\n')
print('knots are:', np.sort(knots))

# +
# Control for demographic data, add variable gender as a confounder
# I use 'race' for demographic data as warranted because different races may have different tooth conditions,
# adding race may have more information so that information loss can be less, which makes a smaller AIC value.
# After adding race, minimum AIC decreased from 58670 to only 23043, which shows a much better fit result.

logitinter = smf.logit('ohx01tc_y ~ patsy.bs(age, knots = knots, degree = 3) * race', data = data_all).fit()
print(logitinter.summary(),'\n')
print('AIC after adding race is:', logitinter.aic)

# +
## (b)
## To show fitted values in a more clear way, I use 'groupby' function to average fitted probability within each age.

knots = [20, 28, 66, 69] # best knots in (a)
tooth_variables = ['ohx' + str(i + 1).zfill(2) + 'tc' for i in range(32)]
fittedvalue = pd.DataFrame()
for i in range(len(tooth_variables)):
    groupname = tooth_variables[i] + str(i+1)
    data_all[groupname] = data_all[tooth_variables[i]].apply(lambda x: 1 if x == 2 else 0)  
    name = 'proby' + str(i + 1)
    log_data[name] = data_all.groupby(['age'])[groupname].mean()
    logitm = smf.logit(groupname + ' ~ patsy.bs(age, knots = knots, degree = 3)', data = data_all).fit()
    fitname = 'yhat' + str(i + 1)
    data_all['predict'+ str(i + 1)] = logitm.predict()
    fittedgroup = data_all.groupby(['age'])['predict'+ str(i + 1)].mean()
    fittedvalue[fitname] = fittedgroup

# display 
pd.options.display.float_format = '{:.5f}'.format
display(fittedvalue)


# -

## (c)
# Suppose there are 32 kinds of tooth count, so set up a 4 * 8 subplot set.
plt.figure(800 * 800)
plt.rcParams['figure.figsize'] = (40, 40)
plt.style.use('ggplot')
plt.style.context('dark')
col = ['red', 'black', 'blue', 'purple']
X = np.arange(13, 81, 1)
for i in range(32):
    ax = plt.subplot(8, 4, (i + 1))
    ax.scatter(X, fittedvalue.iloc[:, i], linewidth = 1, color = col[(i + 1) % 4])
    ax.set_ylabel('probability permanent present')
    ax.set_xlabel('age')
    ax.set_title('fitted value')
    

# Question 2
## Since in 1a I select tooth 'ohx01tc' as x, so I split this into 10 groups by quantile.
### 1
q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
probfit = logitmodel.predict()
quantiles = np.quantile(probfit, q)
bindict = {}
bindict['q1'] = probfit[probfit <= quantiles[0]] 
indexdict = {}
indexdict['q1'] = np.where(probfit <= quantiles[0])
for i in range(8):
    indexdict['q' + str(i + 2)] = np.where((probfit > quantiles[i]) & (probfit <= quantiles[i + 1]))
    bindict['q' + str(i + 2)] = probfit[(probfit > quantiles[i]) & (probfit <= quantiles[i + 1])]
indexdict['q10'] = np.where(probfit > quantiles[8])    
bindict['q10'] = probfit[probfit > quantiles[8]] 
bindict

### 2
observeddict = {}
expecteddict = {}
for j in range(10):
    count = 0
    length = len(indexdict['q' + str(j + 1)][0])
    for i in range(length):
        count += data_all['ohx01tc_y'].iloc[indexdict['q' + str(j + 1)][0][i]]
    observeddict['q' + str(j + 1)] = count / length
    expecteddict['q' + str(j + 1)] = np.mean(bindict['q' + str(j + 1)])
print('observed proportion:', observeddict)
print('expected proportion:', expecteddict)

### 3
# %matplotlib inline
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams["legend.loc"]
X = list(observeddict.values())
Y = list(expecteddict.values())
# y = x with slope 1
X1 = np.linspace(0, 0.4, 101)
Y1 = np.linspace(0, 0.4, 101)
plt.plot(X, Y, '--', color = 'red', linewidth = 5, label = 'prob points')
plt.plot(X1, Y1, color = 'blue', label = 'slope 1 line')
plt.xlabel('observed proportion')
plt.ylabel('fitted proportion')
plt.title('Hosmer-Lemeshow goodness-of-fit test')
plt.legend()
plt.show()
    

# +
### 4
## Comment: From figures drawn in question 3, we can see observed probabilities and fitted probabilities in each decile
## are almost shaped as a line with slope 1. So the fitted result is very good.

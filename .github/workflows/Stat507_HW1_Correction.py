# -*- coding: utf-8 -*-
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

# # Question 0
# This is *question 0* for <font color = "##dd0000">[problem set 1](https://jbhender.github.io/Stats507/F21/ps/ps1.html)
# </font> of <font color = "##dd0000">[Stats 507](https://jbhender.github.io/Stats507/F21/)</font>.
# > Qestion 0 is about Markdown.
#
# The next question is about the **Fibonnaci sequence**,$F_n = F_{n-2} + F_{n-1}.$
# In part **a** we will define a Python function `fib_rec()`.
#
# Below is a ...
#
# ### Level 3 Header
# Next,we can make a bulleted list:
#    
# * Item 1
#     * detail 1
#     * detail 2
# * Item 2
#
# Finally, we can make an enumerated list:
# 1. Item 1
# 2. Item 2
# 3. Item 3
#     

# +
# Question 0
## This is the correction of PS 1. 

'''
This is *question 0* for <font color = "##dd0000">[problem set 1](https://jbhender.github.io/Stats507/F21/ps/ps1.html)
</font> of <font color = "##dd0000">[Stats 507](https://jbhender.github.io/Stats507/F21/)</font>.
> Qestion 0 is about Markdown.

The next question is about the **Fibonnaci sequence**,$F_n = F_{n-2} + F_{n-1}.$
In part **a** we will define a Python function `fib_rec()`.

Below is a ...

### Level 3 Header

Next,we can make a bulleted list:
   
* Item 1
    * detail 1
    * detail 2
* Item 2

Finally, we can make an enumerated list:
1  Item 1
2. Item 2
3. Item 3
'''    

# +
import numpy as np
import pandas as pd
import time
from tabulate import tabulate #Tabulate used to draw table rather than calculation.
import math
import warnings # Warnings is used to raise warnings.
from scipy import stats
from scipy.stats import norm
from scipy.stats import beta

# Question 1
## (a)
def fib_rec(n):
    """
    This function aims to calculate Fibonnaci Sequence by recursive way.
    
    Parameters
    ----------
    n : Integer,
      This parameter represents the sequence number of Fibonnaci.The output of this function 
      is the value of Fibonnaci when sequence number is n.
    F_0 : Integer,
      This parameter represents the first value of Fibonnaci Sequence
    F_1 : Integer,
      This parameter represents the second value of Fibonnaci Sequence
      
     Returns 
     ----------
     Integer,
       the nth value of Fibonnaci Sequence
    """
    F_0 = 0
    F_1 = 1
    if n == 0 :
        return F_0
    elif n == 1 :
        return F_1
    else:
        return fib_rec(n-1)+fib_rec(n-2)
### test
print(fib_rec(7))
print(fib_rec(11))
print(fib_rec(13))


# -

## (b)
def fib_for(n):
    """
    This function aims to calculate Fibonnaci Sequence by summation using a 'for' loop.

    Parameters
    ----------
    n : Integer,
      This parameter represents the sequence number of Fibonnaci.The output of this function 
    is the value of Fibonnaci when sequence number is n.
    
    Returns
    ----------
    Integer,
      the nth value of Fibonnaci Sequence
    """ 
    fib_1 = 1 
    fib_0 = 0 #set initial value
    for i in range(2,n+1):
        if i%2 == 0:
            fib_0 = fib_0 + fib_1
        else:
            fib_1 = fib_0 + fib_1
    return max(fib_0,fib_1)
### test
print(fib_for(7))
print(fib_for(11))
print(fib_for(13))


## (c)
def fib_whl(n):
    """
    This function aims to calculate Fibonnaci Sequence by summation using a 'while' loop.

    Parameters
    ----------
    n : Integer,
      This parameter represents the sequence number of Fibonnaci.The output of this function 
    is the value of Fibonnaci when sequence number is n.
    
    Returns
    ----------
    Integer,
      the nth value of Fibonnaci Sequence
    """ 
    fib_1 = 1 
    fib_0 = 0 #set initial value  
    i=0 # a new growing order i to determine which fib to increase
    while n >= 2:
        if i%2 == 0:
            fib_0 = fib_0 + fib_1
        else:
            fib_1 = fib_0 + fib_1
        i = i+1
        n = n-1
    return max(fib_0,fib_1)
### test
print(fib_whl(7))
print(fib_whl(11))
print(fib_whl(13))


## (d)
def fib_rnd(n):
    """
    This function aims to calculate Fibonnaci Sequence by 'rounding' method related to golden ratio.

    Parameters
    ----------
    n : Integer,
      This parameter represents the sequence number of Fibonnaci.The output of this function 
    is the value of Fibonnaci when sequence number is n.
    
    Returns
    ----------
    Integer,
      the nth value of Fibonnaci Sequence
    """ 
    phi = (1+np.sqrt(5))/2 
    output = phi**n/np.sqrt(5)
    output = round(output)
    return output
### test
print(fib_rnd(7))
print(fib_rnd(11))
print(fib_rnd(13))


## (e)
def fib_flr(n):
    """
    This function aims to calculate Fibonnaci Sequence by 'truncation' method related to golden ratio.
    
    Parameters
    ----------
    n : Integer,
      This parameter represents the sequence number of Fibonnaci.The output of this function 
    is the value of Fibonnaci when sequence number is n.
    
    Returns
    ----------
    Integer,
      the nth value of Fibonnaci Sequence
    """ 
    phi = (1+np.sqrt(5))/2 
    output = phi**n/np.sqrt(5) + 1/2
    output = round(output)
    return output
### test
print(fib_flr(7))
print(fib_flr(11))
print(fib_flr(13))     

# +
## (f)
### The sequence of n is set as l:[5,10,15,20,25,30,35,40,45] 
### For each n,run 20 times and get median of each n,each method.
l = [i for i in range(5,41,5)]
rec_median = []
for_median = []
whl_median = []
rnd_median = []
flr_median = [] 
for i in l:
    rec_time = []
    for_time = []
    whl_time = []
    rnd_time = []
    flr_time = [] 
    for j in range(5):
        #rec
        start_rec_time = time.perf_counter() #start timing
        fib_rec(i)
        end_rec_time = time.perf_counter() # end timing
        re_t = end_rec_time - start_rec_time # run time
        rec_time.append(re_t)
        #for
        start_for_time = time.perf_counter()
        fib_for(i)
        end_for_time = time.perf_counter()
        f_t = end_for_time - start_for_time
        for_time.append(f_t)
        #whl
        start_whl_time = time.perf_counter()
        fib_whl(i)
        end_whl_time = time.perf_counter()
        w_t = end_whl_time - start_whl_time
        whl_time.append(w_t)
        #rnd
        start_rnd_time = time.perf_counter()
        fib_rnd(i)
        end_rnd_time = time.perf_counter()
        rn_t = end_rnd_time - start_rnd_time
        rnd_time.append(rn_t)
        #flr
        start_flr_time = time.perf_counter()
        fib_flr(i)
        end_flr_time = time.perf_counter()
        e_t = end_flr_time - start_flr_time
        flr_time.append(e_t)
        
    m_re = round(np.median(rec_time),6) #get median time of each n
    m_fo = round(np.median(for_time),6)
    m_w = round(np.median(whl_time),6)
    m_rn = round(np.median(rnd_time),6)
    m_fl = round(np.median(flr_time),6)
    
    rec_median.append(m_re)
    for_median.append(m_fo) 
    whl_median.append(m_w)
    rnd_median.append(m_rn)
    flr_median.append(m_fl)
    
result = [rec_median,for_median,whl_median,rnd_median,flr_median] 
index = ['rec','for','whl','rnd','flr']
df = pd.DataFrame(result,index = index)
columns = ['n_value','5','10','15','20','25','30','35','40']
print(tabulate(df,headers = columns))


# +
#  Question 2
## (a)
def Bino_coef(n,k):
    """
    This function aims to calculate Binomial coefficients.

    Parameters
    ----------
    n : Integer,
      The top number in Binomial coefficient.
    k : Integer
      The bottom number in Binomial coefficient.
    
    Returns
    ----------
    Integer,
      Binomial coefficients of (n,k)
    """ 
    if k==0:
        return 1
    else:
        result = int(Bino_coef(n,k-1)*(n+1-k)/k)
    return result

def Pascal_row(length):
    """
    This function aims to output a specified row of Pascal's triangle .

    Parameters
    ----------
    length : Integer,
      The length of the row,which is also the lth row of Pascal triangle
    
    Returns
    ----------
    A Sequence,
      the nth row of Pascal's triangle
    """ 
    list_p = []
    for i in range(0,length):
        pas_str = str(Bino_coef(length-1,i))
        list_p.append(pas_str)
    return ' '.join(list_p)
### test
Pascal_row(4) 


# -

## (b)
def triangle_paskal(nrow):
    """
    This function aims to output the first n rows of Pascal's triangle.

    Parameters
    ----------
    nrow : Integer,
      Row number of triangle_paskal 
    
    Returns
    ----------
    A triangle,
      the first n rows of Pascal’s triangle
    """ 
    # To gurantee enough space for triangle, I need to find the largest byte of all numbers when trasfer then into str.
    # In the former n rows, the largest number is Binomial coefficient(n,[n/2])
    max_n = Bino_coef(nrow,int(nrow/2))
    byte_len = len(str((max_n)))
    for i in range(1,nrow+1):
        a = str(Pascal_row(i))
        print(a.center(byte_len*nrow))
### test
print(triangle_paskal(15))


# Question 3
## (a)
def normal_estimate(data_array,c_level, CI_format = "{est}[{level}% CI: ({lwr}, {upr})]"):
    """
    This function aims to calculate the standard point and interval estimate
    for the populaiton mean based on Normal theory with sample data
    
    Parameters
    ----------
    data_array : array,
      This parameter represents the input data which is written as 1d array.
    c_level : int,
      This parameter represents the number of confidence interval
    CI_format : String,
      This parameter represents the format of point estimate and confidence interval.
      Especially,this parameter is configurable to dictionary format if its input value
      If the value of string is True,output string,othereise output dictionary.
     z : float,
      This parameter represents the confidence level quantile of standard normal distribution.
    
    Returns
    ----------
    A string or a dictionary,
      point and interval estimate for the mean
    """
    try:
        data_array = np.array(data_array)
    except:
        raise Exception('Input data is not an array !')
        
    if data_array.ndim != 1:
        raise Exception('Input data is not a 1d array !')
        
    est = np.mean(data_array)
    se = stats.sem(data_array)
    z = norm.ppf((1+c_level/100)/2,loc=0,scale=1)
    lwr = est - z * se
    lwr = round(lwr,4)
    upr = est + z * se
    upr = round(upr,4)
    level = c_level
    
    dic = {'est': round(est, 4), 'lwr': round(lwr, 4), 'upr': round(upr, 4), 
           'level': level}
    
    if CI_format is None:
        return dic
    else:
        return CI_format.format_map(dic)
### test
normal_estimate([1,2,3],90,None)


# +
## (b)
def proportion_normal_CI(data_array, c_level, method, CI_format = "{est}[{level}% CI: ({lwr}, {upr})]"):
    """
     This function aims to calculate the standard point and interval estimate 
     for the Bernoulli populaiton proportion by Normal approximation,
     Clopper-Pearson,Jeffrey's or Agresti-Coull interval.
     
     Parameter
     ----------
     data_array : array,
       This parameter represents the input data which is written as 1d array.
     c_level : int,
       This parameter represents the number of confidence interval
       If the value of string is True,output string,othereise output dictionary.
     method : string,
       This parameter represents the method of calcaulating confidence interval,which has 
       got 4 values:'Nor','Clo','Jef','Agr'.These 4 values represent Normal approximation,
       Clopper-Pearson,Jeffrey's and Agresti-Coull interval respectively.
     CI_format : String,
       This parameter represents the format of point estimate and confidence interval.
       Especially,this parameter is configurable to dictionary format if its input value 
       is None.
     x : int,
       This parameter represents the number of successes in Bernoulli trials.
     n : int,
       This parameter represents the number of Bernoulli trials. 
     
     Returns
     ---------
     A string or a dictionary,
       point and interval estimate for Bernoulli proportion
    """   
    try:
        data_array = np.array(data_array)
    except:
        raise Exception('Input data is not an array !')
        
    if data_array.ndim != 1:
        raise Exception('Input data is not a 1d array !')

    x = sum(data_array) # number of successful trials
    n = len(data_array) # number of trials
    p = x/n
    level = c_level
    ql = (1-level/100)/2 #lower quantile of interval
    qu = (1+level/100)/2 #upper quantile of interval
    est = p

    if method == 'Nor':
        if p * n <= 12 or (1 - p)* n <=12 :
            warnings.warn('The accuracy of this interval may be inadequate.')
        est = p #estimation of proportion
        se = np.sqrt(p*(1-p)/n)
        z = norm.ppf(qu,loc=0,scale=1)
        lwr = est - z * se
        upr = est + z * se
    elif method == 'Clo':
        lwr = beta.ppf(ql, x, n-x+1)
        upr = beta.ppf(qu, x+1, n-x)
    elif method == 'Jef':
        lwr = max(0,beta.ppf(ql, x+0.5, n-x+0.5))
        upr = min(beta.ppf(qu, x+0.5, n-x+0.5),1)
    elif  method == 'Agr':
        z = norm.ppf(qu,loc=0,scale=1)
        n_hat = n + z**2 
        p_hat = (x + z**2/2)/n_hat
        if p_hat * n_hat <= 12 or (1 - p_hat)* n_hat <=12 :
            warnings.warn('The accuracy of this interval may be inadequate.')
        se = np.sqrt(p_hat*(1-p_hat)/n_hat)
        z = norm.ppf(qu,loc=0,scale=1)
        lwr = p_hat - z * se
        upr = p_hat + z * se
        est = p_hat
    else:
        raise ValueError('The value of parameter method is wrong.')
    
    dic = {'est': round(est, 4), 'lwr': round(lwr, 4), 'upr': round(upr, 4), 
           'level': level}
    
    if CI_format is None:
        return dic
    else:
        return CI_format.format_map(dic)
    
### test
test_data = [1,0,0,1,0,0,1,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1]
print('Normal:',proportion_normal_CI(test_data,90,'Nor'))
print('Clopper:',proportion_normal_CI(test_data,95,'Clo'))
print('Jeffrey:',proportion_normal_CI(test_data,95,'Jef',None))
print('Agresti:',proportion_normal_CI(test_data,95,'Agr'))

# +
## (c)
l_0 = [0] 
l_1 = [1]
array_0 = [i for i in l_0 for j in range(0,48)]
array_1 = [i for i in l_1 for j in range(0,42)]
array_0.extend(array_1)
array = np.asarray(array_0) # create the required array

method = ['Nor', 'Clo', 'Jef', 'Agr','Nor_a']
level = [90, 95, 99]
str_level = ['90%','95%','99%']

# create a dataframe to load point and interval estimate
df = pd.DataFrame(np.zeros([5,3]),index=method,columns=str_level)
# create another dataframe to load width of interval
result = pd.DataFrame(np.zeros([5,3]),index=method,columns=str_level)
# fill result dataframe with cases of 3b
for i in range(0,4):
    for j in range(0,3):
        result_dict = proportion_normal_CI(array, level[j], method[i],None)
        df_str = proportion_normal_CI(array, level[j], method[i])
        
        result.iloc[i,j] = result_dict['upr'] - result_dict['lwr']
        df.iloc[i,j] = df_str
# fill result dataframe with case of 3a
for k in range(0,3):
    result_1_dict = normal_estimate(array,level[k],None)
    df_1_str = normal_estimate(array,level[k])
    
    result.iloc[4,k] = result_1_dict['upr'] - result_1_dict['lwr']   
    df.iloc[4,k] = df_1_str

### This table describes the width of confidence interval of each confidence
### level and interval method.From the table ,we can get that for each confidence
### level,the Agresti-Coull’s interval owns the smallest width. 

col_names = ['Confidence level','90%','95%','99%']
print('Table of interval width :\n',tabulate(result,headers = col_names),'\n')
print('Table of point and interval estimate :\n',tabulate(df,headers = col_names))
# -



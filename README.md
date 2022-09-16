
# Predicting Sale Price

Analyzing Askaan database to understand the possible patterns
that will affect real estate prices.

In this project, I used my data science skills and built an advanced
regression model to predict the selling price of real estate with high accuracy.

## Problem Statement
A national level investment company, called Askaan business co. is looking for data scientists to help them understand the possible patterns that will affect the real estate prices. 

Currently the company purchases and sells real estates across the country. The company is interested in estimating the price of real estate after 5 years from the date of purchase. Such prediction system will help the company to invest in potential estates that will generate substantial profit margins. 

The company has provided the relevant data that they have collected over the
years. Following table presents an overview of the given data:

## Metadata and Details

To perform the required analysis, there are 3 datasets with the following details and tasks:

Table 1: Data Description (Any non-applicable value is set to NA)

| Field   |      Description      |
|----------|:-------------:|
| Sale-Price |  Sale Price of the property after 5 years from the date of purchase in millions of SAR |
| Purchase-Date | Month and year, when the property was purchased.   |
| Purchase-Price | Property's price at the time of purchase in millions of SAR. |
| Type   |  Type of the property. The property could be open-land, villa, duplex, flat.   |
| Class | Legal classification of the property, could be one of the following options: residential, industrial, or commercial.|
| Location |  Where the property is located w.r.t nearby city. 'Center' (of the city), 'Border' (at entry/exit of city), 'Outskirts' implies on the outskirts of the city. |
| Shape |  Shape of the property. It could be rectangle, trapezoid, irregular.|
| U-Index | Index based on number of utilities available on a scale of 1 to 5. A value of 5 indicates all utilities are available. |
| Proximity | Proximity to the nearest metro station in meters. |
| N-Rank | Rank based on neighborhood facilities that will make the property attractive on a scale of 1 to 10. 1 indicates best neighborhood. |
| P-Chance | Probability of finding parking space on adjacent roads at a given time. It is a value between 0 and 1, where 1 indicates sure availability of parking space. |
| Built | Original year of construction. Applicable for villa, duplex, flat.  |
| Renovate | Latest renovation year. Applicable for villa, duplex, flat. A value of 0 implies no renovation done so far or renovation not applicable. |
| Access |  Type of direct access to the property, which could be street, alley or highway.  |
| Crime-Rate | Average number of crimes reported per year in the neighborhood. |
| C-Rating |  Pleasantness of the climate throughout the year on a scale of 1 to 5. A value of 5 indicates pleasant climate. |
| Gov-Index |  Expected level of government infrastructure project and/or developments in the neighborhood on a scale of 1 to 10. A value of 10 indicates that there are huge developments planned by the government.|
| Contour | Flatness of the property. Applicable only for the open land type property. A value of C indicates the slope of the property is irregular. A value of F indicates the property has a smooth slope. |
| Garage | Is there a private parking garage? Yes or No. Applicable to the flat or duplex type. All villas have private garage. |
| Swimming | Is there a swimming pool? Yes or No. Applicable to the villa type. |


## 🛠 Project Aim
The aim of this project is to explore the data, and find possible patterns/relationships in the data. The key variable of interest to askaan business co. is Sale-Price. 

Any patterns that shows connections of input variables to the output variable (Sale-Price) will be considered fruitful
by askaan. Assume that the properties that appreciate by 100% or less over the five years are low potential estates, and those that appreciate by 400% or more are high potential estates. The percentage increase or decrease is defined as:
([Sale−Price]−[Purchase−Price]) / [Purchase−Price] ∗ 100.


## Code

```python
# cell for Importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
```

## Task-1
Prepare the data file (A) i.e., handle the missing values, remove outliers, and fix inconsistencies. 
You can pick any set of methods, but clearly justify your approach.

```python
df = pd.read_csv('HousingPrices_A.csv', delimiter = ',') # Read
df.info() #Identify the fields of the data
# Count the number of rows and columns in the data
print(f'The number of rows are {len(df.index)}, and the number of columns are {len(df.columns)}')
null =df.columns[df.isna().any()]
print(f'The columns containing missing data are:\n{null}') 
print(f'The statistical summaries for the numerical data are:')
display(df.describe(include="number"))# only numerical
print(f'The statistical summaries for the categorical data are:')
display(df.describe(include="object")) # only categorical
#--------------------------------------------------------------------------------------------
# Handling the missing values:

# Renovate: is numerical and has some NaN values, so I must fill the NaN. However, the numbers represent dates,
# so I should fill the NaN values with the mode NOT the mean. (The NaN values correspond to land in "Type" column)
df["Renovate"].fillna(df["Renovate"].mode()[0],inplace = True)

# Contour: (The values are for open land in *Type* column, the rest is NaN)
display(df["Contour"].unique()) # to check the unique values in the column to fill appropriately
# To fix this, I filled each NaN value with Not Applicable (NA)
# Each NA represent the values of other "Type" (e.g., villa, duplex, flat), where it isn't applicable to have a value
df["Contour"].fillna("NA",inplace = True)

# Garage: is onle applicable to flat or duplex *Type* (villa always has garage = Yes)
display(df["Garage"].unique()) # to check the unique values in the column to fill appropriately
# Also, Garage column has some NaN values. Since private parking garage can only exist in flat, duplex, or villa,
# I filled each NaN value (corresponding to the remaining types) with "NA" >> can't have a private parking garage.
df["Garage"].fillna("NA",inplace = True) 
display(df["Swimming"].unique())

# Swimming is only applicable to villa *Type*, and has many NaN values (corresponding to other *Types*)
display(df["Swimming"].unique()) # to check the unique values in the column to fill appropriately
# Since a swimming pool can only exist in villas, I fill the NaN values with "NA" >> can't have a Swimming pool
df["Swimming"].fillna("NA",inplace = True) 
display(df["Swimming"].unique())
# # After that, the " No" is displayed with an additional space in the begining, to fix this:
# df["Swimming"] = df["Swimming"].apply(lambda x: str(x).strip()) # to remove the space before " No"
# display(df["Swimming"].unique())
null = df.columns[df.isna().any()]
print(f'The columns containing missing data are:\n{null}') 
print()
#-------------------------------------------------------------
# Handling incosistency: to apply further analysis, I need to make sure that all data are consistent
# Also, the categorical columns should not be deleted. Rather, it should be converted to numerical and nominal 
# to be used in further analysis (regression analysis and PCA).

# *Type* column has 5 unique values, where it should be 4 (land might have been entered wrong, it should be open land)
# to fix this, temp is created to save the "open land". Then, land is converted to open land to ensure consistency
print(df["Type"].unique())
df["Type"] = df["Type"].apply(lambda x: x.replace('open land','temp').replace('land','open land')
                             .replace('temp','open land'))
print(df["Type"].unique())

# Proximity is categorical, while it should be numerical; so I converted it:
df["Proximity"] = df["Proximity"].apply((lambda x: x.replace("mts",""))).apply(pd.to_numeric)

# Since the data in *Purchase-Date* column are not consistent, 
# I must make them all appear with the (full month name-full year) 
# Fixing the full month name rather than the three letters abbreviation and also fixing the full year: 
df["Purchase-Date"] = df["Purchase-Date"].apply(lambda x: x[0:3]+"-19"+x[-2:] if int(x[-2:])>20 else  
                                                x[0:3]+"-20"+x[-2:]).apply(lambda x: x.replace("Jan-","January-")
                                                .replace("Feb-","February-")
                                                .replace("Mar-","March-").replace("Apr-","April-")
                                                .replace("Jun-","June-").replace("Jul-","July-")
                                                .replace("Aug-","August-")
                                                .replace("Sep-","September-").replace("Oct-","October-")
                                                .replace("Nov-","November-").replace("Dec-","December-"))
#---------------------------------------------------------------------------------------------------------
# Handling outliers: 
# To provide precise details, outliers must be removed.
# 1- The two standard deviations approach: (Removes the values that are far from the mean and it keeps many values to be processed)

# scaled_values = StandardScaler().fit_transform(df.select_dtypes(exclude='object'))
# df2 = pd.DataFrame(scaled_values,columns=df.select_dtypes(exclude='object').columns)
# threshold = 2
# selected_rows= (df2<threshold).all(axis=1)  & (df2>-threshold).all(axis=1)
# selected_index=df[~selected_rows].index
# df2.drop(index=selected_index,inplace=True) 
# ndf = df.drop(index=selected_index)  
# ndf.reset_index(inplace = True, drop = True) # to reset the index 
# df = ndf

#2- Drawing boxplots to identify the outliers (above or below the whiskers), and then removing the outliers
# The whiskers approach was preferred
Q3, Q1 = np.quantile(df["Sale-Price"], [0.75 ,0.25])
IQR = Q3 - Q1
upperWhisker = Q3 + 1.5 * IQR
lowerWhisker = Q1 - 1.5 * IQR
mask = (df["Sale-Price"] > upperWhisker) | (df["Sale-Price"] < lowerWhisker) # identifing the outliers
indexes = df[mask].index # getting the indecies for outliers (to drop them next)
df.drop(indexes, inplace=True)
# df[mask]   # to make sure there are no more outliers
df.info()
df
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3000 entries, 0 to 2999
    Data columns (total 20 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   Sale-Price      3000 non-null   float64
     1   Purchase-Date   3000 non-null   object 
     2   Purchase-Price  3000 non-null   float64
     3   Type            3000 non-null   object 
     4   Class           3000 non-null   object 
     5   Location        3000 non-null   object 
     6   Shape           3000 non-null   object 
     7   U-Index         3000 non-null   int64  
     8   Proximity       3000 non-null   object 
     9   N-Rank          3000 non-null   int64  
     10  P-Chance        3000 non-null   float64
     11  Built           3000 non-null   int64  
     12  Renovate        2766 non-null   float64
     13  Access          3000 non-null   object 
     14  Crime-Rate      3000 non-null   int64  
     15  C-Rating        3000 non-null   int64  
     16  Gov-Index       3000 non-null   int64  
     17  Contour         768 non-null    object 
     18  Garage          2232 non-null   object 
     19  Swimming        739 non-null    object 
    dtypes: float64(4), int64(6), object(10)
    memory usage: 468.9+ KB
    The number of rows are 3000, and the number of columns are 20
    The columns containing missing data are:
    Index(['Renovate', 'Contour', 'Garage', 'Swimming'], dtype='object')
    The statistical summaries for the numerical data are:
    



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sale-Price</th>
      <th>Purchase-Price</th>
      <th>U-Index</th>
      <th>N-Rank</th>
      <th>P-Chance</th>
      <th>Built</th>
      <th>Renovate</th>
      <th>Crime-Rate</th>
      <th>C-Rating</th>
      <th>Gov-Index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3000.000000</td>
      <td>3000.000000</td>
      <td>3000.000000</td>
      <td>3000.000000</td>
      <td>3000.000000</td>
      <td>3000.000000</td>
      <td>2766.000000</td>
      <td>3000.000000</td>
      <td>3000.000000</td>
      <td>3000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>14919.682000</td>
      <td>6420.645933</td>
      <td>3.048667</td>
      <td>5.545667</td>
      <td>0.503161</td>
      <td>1979.846000</td>
      <td>815.838756</td>
      <td>49.891333</td>
      <td>3.037333</td>
      <td>5.392333</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3405.722055</td>
      <td>3207.601791</td>
      <td>1.423016</td>
      <td>2.865773</td>
      <td>0.288068</td>
      <td>12.964193</td>
      <td>976.735817</td>
      <td>28.988118</td>
      <td>1.434094</td>
      <td>2.853499</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5392.000000</td>
      <td>801.100000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000408</td>
      <td>1951.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>12541.100000</td>
      <td>3608.925000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>0.255795</td>
      <td>1970.000000</td>
      <td>0.000000</td>
      <td>25.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>14787.200000</td>
      <td>6566.050000</td>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>0.497203</td>
      <td>1980.000000</td>
      <td>0.000000</td>
      <td>50.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>17134.925000</td>
      <td>9202.625000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>0.755137</td>
      <td>1990.000000</td>
      <td>1980.000000</td>
      <td>75.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>27513.300000</td>
      <td>11999.100000</td>
      <td>5.000000</td>
      <td>10.000000</td>
      <td>0.999984</td>
      <td>2009.000000</td>
      <td>2009.000000</td>
      <td>99.000000</td>
      <td>5.000000</td>
      <td>10.000000</td>
    </tr>
  </tbody>
</table>
</div>


    The statistical summaries for the categorical data are:
    


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Purchase-Date</th>
      <th>Type</th>
      <th>Class</th>
      <th>Location</th>
      <th>Shape</th>
      <th>Proximity</th>
      <th>Access</th>
      <th>Contour</th>
      <th>Garage</th>
      <th>Swimming</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3000</td>
      <td>3000</td>
      <td>3000</td>
      <td>3000</td>
      <td>3000</td>
      <td>3000</td>
      <td>3000</td>
      <td>768</td>
      <td>2232</td>
      <td>739</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>1064</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2957</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>February-1998</td>
      <td>flat</td>
      <td>residential</td>
      <td>Center</td>
      <td>trapezoid</td>
      <td>4852</td>
      <td>highway</td>
      <td>F</td>
      <td>Yes</td>
      <td>No</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>11</td>
      <td>759</td>
      <td>1696</td>
      <td>1029</td>
      <td>1005</td>
      <td>3</td>
      <td>1466</td>
      <td>399</td>
      <td>1492</td>
      <td>390</td>
    </tr>
  </tbody>
</table>
</div>



    array([nan, 'F', 'C'], dtype=object)



    array(['Yes', nan, 'No'], dtype=object)



    array([' No', nan, 'Yes'], dtype=object)



    array([' No', nan, 'Yes'], dtype=object)



    array([' No', 'NA', 'Yes'], dtype=object)


    The columns containing missing data are:
    Index([], dtype='object')
    
    ['villa' 'land' 'duplex' 'open land' 'flat']
    ['villa' 'open land' 'duplex' 'flat']
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2973 entries, 0 to 2999
    Data columns (total 20 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   Sale-Price      2973 non-null   float64
     1   Purchase-Date   2973 non-null   object 
     2   Purchase-Price  2973 non-null   float64
     3   Type            2973 non-null   object 
     4   Class           2973 non-null   object 
     5   Location        2973 non-null   object 
     6   Shape           2973 non-null   object 
     7   U-Index         2973 non-null   int64  
     8   Proximity       2973 non-null   int64  
     9   N-Rank          2973 non-null   int64  
     10  P-Chance        2973 non-null   float64
     11  Built           2973 non-null   int64  
     12  Renovate        2973 non-null   float64
     13  Access          2973 non-null   object 
     14  Crime-Rate      2973 non-null   int64  
     15  C-Rating        2973 non-null   int64  
     16  Gov-Index       2973 non-null   int64  
     17  Contour         2973 non-null   object 
     18  Garage          2973 non-null   object 
     19  Swimming        2973 non-null   object 
    dtypes: float64(4), int64(7), object(9)
    memory usage: 487.8+ KB
    





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sale-Price</th>
      <th>Purchase-Date</th>
      <th>Purchase-Price</th>
      <th>Type</th>
      <th>Class</th>
      <th>Location</th>
      <th>Shape</th>
      <th>U-Index</th>
      <th>Proximity</th>
      <th>N-Rank</th>
      <th>P-Chance</th>
      <th>Built</th>
      <th>Renovate</th>
      <th>Access</th>
      <th>Crime-Rate</th>
      <th>C-Rating</th>
      <th>Gov-Index</th>
      <th>Contour</th>
      <th>Garage</th>
      <th>Swimming</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9545.3</td>
      <td>June-1980</td>
      <td>2156.8</td>
      <td>villa</td>
      <td>residential</td>
      <td>Border</td>
      <td>rectangle</td>
      <td>1</td>
      <td>26608</td>
      <td>9</td>
      <td>0.455932</td>
      <td>1967</td>
      <td>1975.0</td>
      <td>highway</td>
      <td>13</td>
      <td>2</td>
      <td>7</td>
      <td>NA</td>
      <td>Yes</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16653.6</td>
      <td>February-2006</td>
      <td>5949.5</td>
      <td>villa</td>
      <td>residential</td>
      <td>Center</td>
      <td>rectangle</td>
      <td>2</td>
      <td>17132</td>
      <td>10</td>
      <td>0.937587</td>
      <td>1994</td>
      <td>2003.0</td>
      <td>alley</td>
      <td>28</td>
      <td>1</td>
      <td>6</td>
      <td>NA</td>
      <td>Yes</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17885.3</td>
      <td>February-1978</td>
      <td>11751.4</td>
      <td>open land</td>
      <td>commercial</td>
      <td>Border</td>
      <td>rectangle</td>
      <td>2</td>
      <td>37824</td>
      <td>9</td>
      <td>0.162595</td>
      <td>1960</td>
      <td>0.0</td>
      <td>street</td>
      <td>64</td>
      <td>3</td>
      <td>1</td>
      <td>F</td>
      <td>NA</td>
      <td>NA</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14053.0</td>
      <td>March-1975</td>
      <td>6829.6</td>
      <td>villa</td>
      <td>residential</td>
      <td>Center</td>
      <td>rectangle</td>
      <td>1</td>
      <td>19872</td>
      <td>5</td>
      <td>0.070315</td>
      <td>1968</td>
      <td>0.0</td>
      <td>alley</td>
      <td>68</td>
      <td>1</td>
      <td>6</td>
      <td>NA</td>
      <td>Yes</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15741.2</td>
      <td>November-1990</td>
      <td>4469.3</td>
      <td>duplex</td>
      <td>residential</td>
      <td>Center</td>
      <td>irregular</td>
      <td>2</td>
      <td>2344</td>
      <td>4</td>
      <td>0.914535</td>
      <td>1990</td>
      <td>0.0</td>
      <td>alley</td>
      <td>26</td>
      <td>1</td>
      <td>7</td>
      <td>NA</td>
      <td>No</td>
      <td>NA</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2995</th>
      <td>14464.0</td>
      <td>March-1995</td>
      <td>10105.9</td>
      <td>flat</td>
      <td>commercial</td>
      <td>Outskirts</td>
      <td>irregular</td>
      <td>2</td>
      <td>80129</td>
      <td>2</td>
      <td>0.733165</td>
      <td>1986</td>
      <td>0.0</td>
      <td>highway</td>
      <td>10</td>
      <td>5</td>
      <td>10</td>
      <td>NA</td>
      <td>No</td>
      <td>NA</td>
    </tr>
    <tr>
      <th>2996</th>
      <td>14830.1</td>
      <td>October-1995</td>
      <td>6383.1</td>
      <td>flat</td>
      <td>residential</td>
      <td>Outskirts</td>
      <td>rectangle</td>
      <td>5</td>
      <td>61085</td>
      <td>5</td>
      <td>0.866005</td>
      <td>1985</td>
      <td>1994.0</td>
      <td>highway</td>
      <td>90</td>
      <td>1</td>
      <td>9</td>
      <td>NA</td>
      <td>No</td>
      <td>NA</td>
    </tr>
    <tr>
      <th>2997</th>
      <td>8654.2</td>
      <td>January-1972</td>
      <td>6998.6</td>
      <td>villa</td>
      <td>residential</td>
      <td>Border</td>
      <td>irregular</td>
      <td>5</td>
      <td>30725</td>
      <td>10</td>
      <td>0.073880</td>
      <td>1953</td>
      <td>1961.0</td>
      <td>street</td>
      <td>46</td>
      <td>1</td>
      <td>6</td>
      <td>NA</td>
      <td>Yes</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2998</th>
      <td>17322.1</td>
      <td>February-2007</td>
      <td>2555.8</td>
      <td>flat</td>
      <td>residential</td>
      <td>Center</td>
      <td>rectangle</td>
      <td>5</td>
      <td>17731</td>
      <td>8</td>
      <td>0.421831</td>
      <td>1992</td>
      <td>2003.0</td>
      <td>alley</td>
      <td>90</td>
      <td>3</td>
      <td>5</td>
      <td>NA</td>
      <td>No</td>
      <td>NA</td>
    </tr>
    <tr>
      <th>2999</th>
      <td>15057.4</td>
      <td>April-1973</td>
      <td>3400.4</td>
      <td>flat</td>
      <td>commercial</td>
      <td>Outskirts</td>
      <td>irregular</td>
      <td>4</td>
      <td>79508</td>
      <td>9</td>
      <td>0.897812</td>
      <td>1973</td>
      <td>0.0</td>
      <td>highway</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>NA</td>
      <td>Yes</td>
      <td>NA</td>
    </tr>
  </tbody>
</table>
<p>2973 rows × 20 columns</p>
</div>



## Task-2
Using data (B), draw the pair-wise plots between all the input variables and the output variable (Sale-Price).

```python
df = pd.read_csv('HousingPrices_B.csv', delimiter = ',',keep_default_na=False) # Read and keep NA in file as values non NaNs
# Assuming there are no outliers in dataset B 
# To apply Exploratory Data Analysis, all columns used must be numerical. To fix this:
# *Purchase-Date* is converted to numerical for graphing:
df["Purchase-Date"] = df["Purchase-Date"].apply(lambda x: x.replace("-","").lstrip("Jan").lstrip("January")
                                               .lstrip("Feb").lstrip("February").lstrip("Mar").lstrip("March")
                                               .lstrip("Apr").lstrip("April").lstrip("Jun").lstrip("June")
                                               .lstrip("Jul").lstrip("July").lstrip("Aug").lstrip("August")
                                               .lstrip("Sep").lstrip("September").lstrip("Oct").lstrip("October")
                                               .lstrip("Nov").lstrip("November").lstrip("Dec")
                                                .lstrip("December")).apply(pd.to_numeric)

selected_columns = df.drop(columns=["Sale-Price"]) # dropping the output 
# Divide the inputs to seperate them (to look better when graphed)
X1 = selected_columns.iloc[:,:3]
X2 = selected_columns.iloc[:,3:6]
X3 = selected_columns.iloc[:,6:9]
X4 = selected_columns.iloc[:,9:12]
X5 = selected_columns.iloc[:,12:15]
X6 = selected_columns.iloc[:,15:17] 
X7 = selected_columns.iloc[:,17:] 

# pairplot with histograms (3d bivariate) are used to depict the relationship between each input and the output
# 3d bivariate histogram was chosen since it is aesthetically pleasing and can be used for both num/catg
plt.figure()
sns.pairplot(x_vars=X1,y_vars="Sale-Price",kind="hist",height=3, aspect=2,data=df)
sns.pairplot(x_vars=X2,y_vars="Sale-Price",kind="hist",height=3, aspect=2,data=df)
sns.pairplot(x_vars=X3,y_vars="Sale-Price",kind="hist",height=3, aspect=2,data=df)
sns.pairplot(x_vars=X4,y_vars="Sale-Price",kind="hist",height=3, aspect=2,data=df)
sns.pairplot(x_vars=X5,y_vars="Sale-Price",kind="hist",height=3, aspect=2,data=df)
sns.pairplot(x_vars=X6,y_vars="Sale-Price",kind="hist",height=3, aspect=2,data=df)
sns.pairplot(x_vars=X7,y_vars="Sale-Price",kind="hist",height=3, aspect=2,data=df)
plt.show()
```


    <Figure size 432x288 with 0 Axes>



    
![png](PredictingSalePrice_files/PredictingSalePrice_4_1.png)
    



    
![png](PredictingSalePrice_files/PredictingSalePrice_4_2.png)
    



    
![png](PredictingSalePrice_files/PredictingSalePrice_4_3.png)
    



    
![png](PredictingSalePrice_files/PredictingSalePrice_4_4.png)
    



    
![png](PredictingSalePrice_files/PredictingSalePrice_4_5.png)
    



    
![png](PredictingSalePrice_files/PredictingSalePrice_4_6.png)
    



    
![png](PredictingSalePrice_files/PredictingSalePrice_4_7.png)
    


## Task-3
Identify top and bottom three numerical variables that are strongly related to the output variable (Sale-Price)? 
Use the relevant analysis approach.

```python
## Before displaying the correlation, I need to consider the categorical columns by encoding them
# The categorical columns should not be dropped. Rather, it should be converted to numerical and nominal
# The categorical columns are converted below:

# Encoding
# Below: mapping, label encoder are used for conversion

# Type column should be converted to numericals for further analysis. 
print(df["Type"].unique())
# Using custom encoding (since the types imply an order, assume that villa is most luxurious, open land is least)
Type_mapping = {'villa':1,'duplex':2,'flat':3,'open land':4}
df["Type"] = df["Type"].map(Type_mapping)
# Using label encoder
encoder = LabelEncoder()
print(df["Class"].unique()) # 0: commercial, 1: industrial, 2: residential (alphabetically)
df["Class"] = encoder.fit_transform(df["Class"]) 
print(df["Location"].unique()) # Border:0, Center:1, Outskirts:2 (alphabetically)
df["Location"] = encoder.fit_transform(df["Location"]) 
print(df["Shape"].unique()) # irregular:0, rectangle:1, trapezoid:2 (alphabetically)
df["Shape"] = encoder.fit_transform(df["Shape"]) 
print(df["Access"].unique()) # alley:0, highway:1, street:2 (alphabetically)
df["Access"] = encoder.fit_transform(df["Access"]) 

print(df["Contour"].unique())
Contour_mapping = {'NA':0,'F':1,'C':2} # 0 means it is not applicable to have a value here
df["Contour"] = df["Contour"].map(Contour_mapping)
print(df["Garage"].unique())
Garage_mapping = {'NA':0,'Yes':1,'No':2} # 0 means it is not applicable to have a value here
df["Garage"] = df["Garage"].map(Garage_mapping)
print(df["Swimming"].unique())
Swimming_mapping = {'NA':0,'Yes':1,' No':2} # 0 means it is not applicable to have a value here
df["Swimming"] = df["Swimming"].map(Swimming_mapping)

# to put the sale-price as the last column (output)
df["SP"] = df["Sale-Price"]
df.drop(columns="Sale-Price", inplace=True)
df["Sale-Price"] =df["SP"]
df.drop(columns="SP", inplace=True)

# Now all columns are included as numerical to be analyzed in the correlation analysis
corr = df.corr()
display(corr.style.background_gradient(cmap='coolwarm').set_precision(4))
# display(corr) #if the style doesn't work 
corr=corr.apply(lambda x: np.abs(x)) # to get absolutes
sorted_corr = corr.sort_values(by=['Sale-Price'], ascending=False) # negative values will be considered using abs.
# sort by Sale-Price to get what variables are correlated with Sale-Price
sorted_corr = sorted_corr['Sale-Price'].index # store the indecies here to sort them next
print('The top three input variables correlated with the Sale-Price are: ',sorted_corr[1:4].tolist())
print('The least three input variables correlated with the Sale-Price are: ',sorted_corr[17::].tolist())
df
#---
## The following 3 columns *Contour*,*Garage*,*Swimming* are only applicable for certain *Type*. 
# For example, *Swimming* is only applicable for "Villa" *Type*. This might cause an issue in the upcoming analysis 
# because many rows will be with the value NA. As a result, the analysis might not produce weak results 
# The correlation might show innacurate results. Also, the PCA might be affected as well
# For those reasons, these 3 columns will be dropped from further analysis
# For now, I will keep them as categorical columns
```

    ['duplex' 'open land' 'flat' 'villa']
    ['residential' 'industrial' 'commercial']
    ['Center' 'Outskirts' 'Border']
    ['irregular' 'trapezoid' 'rectangle']
    ['street' 'alley' 'highway']
    ['NA' 'F' 'C']
    ['No' 'NA' 'Yes']
    ['NA' 'Yes' ' No']
    

![png](PredictingSalePrice_files/PredictingSalePrice_4_0.png)


    The top three input variables correlated with the Sale-Price are:  ['Type', 'Contour', 'Swimming']
    The least three input variables correlated with the Sale-Price are:  ['C-Rating', 'P-Chance', 'N-Rank']
    





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Purchase-Date</th>
      <th>Purchase-Price</th>
      <th>Type</th>
      <th>Class</th>
      <th>Location</th>
      <th>Shape</th>
      <th>U-Index</th>
      <th>Proximity</th>
      <th>N-Rank</th>
      <th>P-Chance</th>
      <th>Built</th>
      <th>Renovate</th>
      <th>Access</th>
      <th>Crime-Rate</th>
      <th>C-Rating</th>
      <th>Gov-Index</th>
      <th>Contour</th>
      <th>Garage</th>
      <th>Swimming</th>
      <th>Sale-Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1975</td>
      <td>2474.5</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>8171</td>
      <td>10</td>
      <td>0.473210</td>
      <td>1964</td>
      <td>1973</td>
      <td>2</td>
      <td>59</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>11896.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1991</td>
      <td>2414.2</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>10290</td>
      <td>1</td>
      <td>0.064744</td>
      <td>1979</td>
      <td>0</td>
      <td>0</td>
      <td>24</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>17187.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1989</td>
      <td>2453.3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>9940</td>
      <td>8</td>
      <td>0.115741</td>
      <td>1983</td>
      <td>0</td>
      <td>2</td>
      <td>94</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>12756.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1973</td>
      <td>2513.7</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>69542</td>
      <td>4</td>
      <td>0.300323</td>
      <td>1967</td>
      <td>0</td>
      <td>1</td>
      <td>68</td>
      <td>1</td>
      <td>9</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>14722.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981</td>
      <td>8691.7</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3328</td>
      <td>5</td>
      <td>0.017304</td>
      <td>1962</td>
      <td>0</td>
      <td>2</td>
      <td>88</td>
      <td>4</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>17694.9</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4995</th>
      <td>1983</td>
      <td>6337.5</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>9507</td>
      <td>8</td>
      <td>0.432669</td>
      <td>1968</td>
      <td>1977</td>
      <td>2</td>
      <td>88</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>15072.4</td>
    </tr>
    <tr>
      <th>4996</th>
      <td>2004</td>
      <td>11209.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>20864</td>
      <td>8</td>
      <td>0.009118</td>
      <td>1988</td>
      <td>1998</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>6</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>13823.7</td>
    </tr>
    <tr>
      <th>4997</th>
      <td>1997</td>
      <td>1604.4</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>78957</td>
      <td>7</td>
      <td>0.561522</td>
      <td>1987</td>
      <td>1997</td>
      <td>1</td>
      <td>83</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>11828.1</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>2006</td>
      <td>7111.0</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>11546</td>
      <td>8</td>
      <td>0.011843</td>
      <td>2006</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>19046.1</td>
    </tr>
    <tr>
      <th>4999</th>
      <td>1981</td>
      <td>4571.9</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>75949</td>
      <td>1</td>
      <td>0.658805</td>
      <td>1978</td>
      <td>0</td>
      <td>1</td>
      <td>86</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>11967.6</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 20 columns</p>
</div>




```python
# Before doing Task 4
df = pd.read_csv('HousingPrices_B.csv', delimiter = ',',keep_default_na=False) # Read and keep NA in file as values non NaNs
# rereading the original df. Mainly, the PCA showed week seperation when I included the
# nominal columns that were converted using label encoding/one hot encoding. 
# To overcome this, even though I will be sacrificing some information, more accurate information will be gained

## The following 3 columns *Contour*,*Garage*,*Swimming* are only applicable for certain *Type*. 
# For example, *Swimming* is only applicable for "Villa" *Type*. This have caused an issue in PC analysis
# because many rows will be with the value NA. As a result, the PC analysis have produced weak results 
# For those reasons, these 3 columns will be ignored in the PC analysis
#------------------------------------------------------------------------------------------------------
df["Purchase-Date"] = df["Purchase-Date"].apply(lambda x: x.replace("-","").lstrip("Jan").lstrip("January")
                                               .lstrip("Feb").lstrip("February").lstrip("Mar").lstrip("March")
                                               .lstrip("Apr").lstrip("April").lstrip("Jun").lstrip("June")
                                               .lstrip("Jul").lstrip("July").lstrip("Aug").lstrip("August")
                                               .lstrip("Sep").lstrip("September").lstrip("Oct").lstrip("October")
                                               .lstrip("Nov").lstrip("November").lstrip("Dec")
                                                .lstrip("December")).apply(pd.to_numeric)

# Type column should be converted to numericals for PCA analysis. 
# Using custom encoding (since the types imply an order, assume that villa is most luxurious, open land is least)
Type_mapping = {'villa':1,'duplex':2,'flat':3,'open land':4}
df["Type"] = df["Type"].map(Type_mapping)
# Using label encoder
encoder = LabelEncoder()
df["Class"] = encoder.fit_transform(df["Class"]) # 0: commercial, 1: industrial, 2: residential (alphabetically)
df["Location"] = encoder.fit_transform(df["Location"]) # Border:0, Center:1, Outskirts:2 (alphabetically)
df["Shape"] = encoder.fit_transform(df["Shape"]) # irregular:0, rectangle:1, trapezoid:2 (alphabetically)
df["Access"] = encoder.fit_transform(df["Access"]) # alley:0, highway:1, street:2 (alphabetically)
# to put the sale-price as the last column (output)
df["SP"] = df["Sale-Price"]
df.drop(columns="Sale-Price", inplace=True)
df["Sale-Price"] =df["SP"]
df.drop(columns="SP", inplace=True)
df
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Purchase-Date</th>
      <th>Purchase-Price</th>
      <th>Type</th>
      <th>Class</th>
      <th>Location</th>
      <th>Shape</th>
      <th>U-Index</th>
      <th>Proximity</th>
      <th>N-Rank</th>
      <th>P-Chance</th>
      <th>Built</th>
      <th>Renovate</th>
      <th>Access</th>
      <th>Crime-Rate</th>
      <th>C-Rating</th>
      <th>Gov-Index</th>
      <th>Contour</th>
      <th>Garage</th>
      <th>Swimming</th>
      <th>Sale-Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1975</td>
      <td>2474.5</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>8171</td>
      <td>10</td>
      <td>0.473210</td>
      <td>1964</td>
      <td>1973</td>
      <td>2</td>
      <td>59</td>
      <td>3</td>
      <td>4</td>
      <td>NA</td>
      <td>No</td>
      <td>NA</td>
      <td>11896.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1991</td>
      <td>2414.2</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>10290</td>
      <td>1</td>
      <td>0.064744</td>
      <td>1979</td>
      <td>0</td>
      <td>0</td>
      <td>24</td>
      <td>3</td>
      <td>1</td>
      <td>F</td>
      <td>NA</td>
      <td>NA</td>
      <td>17187.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1989</td>
      <td>2453.3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>9940</td>
      <td>8</td>
      <td>0.115741</td>
      <td>1983</td>
      <td>0</td>
      <td>2</td>
      <td>94</td>
      <td>3</td>
      <td>5</td>
      <td>NA</td>
      <td>No</td>
      <td>NA</td>
      <td>12756.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1973</td>
      <td>2513.7</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>69542</td>
      <td>4</td>
      <td>0.300323</td>
      <td>1967</td>
      <td>0</td>
      <td>1</td>
      <td>68</td>
      <td>1</td>
      <td>9</td>
      <td>C</td>
      <td>NA</td>
      <td>NA</td>
      <td>14722.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981</td>
      <td>8691.7</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3328</td>
      <td>5</td>
      <td>0.017304</td>
      <td>1962</td>
      <td>0</td>
      <td>2</td>
      <td>88</td>
      <td>4</td>
      <td>8</td>
      <td>F</td>
      <td>NA</td>
      <td>NA</td>
      <td>17694.9</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4995</th>
      <td>1983</td>
      <td>6337.5</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>9507</td>
      <td>8</td>
      <td>0.432669</td>
      <td>1968</td>
      <td>1977</td>
      <td>2</td>
      <td>88</td>
      <td>2</td>
      <td>2</td>
      <td>NA</td>
      <td>No</td>
      <td>NA</td>
      <td>15072.4</td>
    </tr>
    <tr>
      <th>4996</th>
      <td>2004</td>
      <td>11209.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>20864</td>
      <td>8</td>
      <td>0.009118</td>
      <td>1988</td>
      <td>1998</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>6</td>
      <td>NA</td>
      <td>No</td>
      <td>NA</td>
      <td>13823.7</td>
    </tr>
    <tr>
      <th>4997</th>
      <td>1997</td>
      <td>1604.4</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>78957</td>
      <td>7</td>
      <td>0.561522</td>
      <td>1987</td>
      <td>1997</td>
      <td>1</td>
      <td>83</td>
      <td>3</td>
      <td>8</td>
      <td>NA</td>
      <td>No</td>
      <td>NA</td>
      <td>11828.1</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>2006</td>
      <td>7111.0</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>11546</td>
      <td>8</td>
      <td>0.011843</td>
      <td>2006</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
      <td>4</td>
      <td>5</td>
      <td>NA</td>
      <td>Yes</td>
      <td>NA</td>
      <td>19046.1</td>
    </tr>
    <tr>
      <th>4999</th>
      <td>1981</td>
      <td>4571.9</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>75949</td>
      <td>1</td>
      <td>0.658805</td>
      <td>1978</td>
      <td>0</td>
      <td>1</td>
      <td>86</td>
      <td>3</td>
      <td>2</td>
      <td>NA</td>
      <td>Yes</td>
      <td>NA</td>
      <td>11967.6</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 20 columns</p>
</div>



## Task-4
Show if the input variables have the information to separate low and high performing estates? Use plots to justify.

```python
# To apply PCA, all inputs must be numerical. 
# The categorical columns should not be dropped. Rather, it should be converted to numerical and nominal
# The categorical columns that are used were converted above, the rest were ignored ONLY here (last 3 columns):
num_columns = df.select_dtypes(exclude='object').columns # numerical columns
pca_input_columns = num_columns.drop(["Sale-Price"]) # only numerical and drop the output

# Before doing PCA, the data appear to have different numerical scales. To fix this, normalization is needed
#--------
pca = PCA(n_components=2) # create object (2 is how many principal components)
principalComponents = pca.fit_transform(StandardScaler().fit_transform(df[pca_input_columns])) # Standardize
# by using standard scaler, I optained more accurate results
df['PC1'] = principalComponents[:,0] 
df['PC2'] = principalComponents[:,1] 

# Classify the performance based on the (appreciation and depreciation) level
df["Appreciation/Depreciation"] = (df["Sale-Price"] - df["Purchase-Price"]) / df["Purchase-Price"] # formula
df["Potential"] = df["Appreciation/Depreciation"].apply(lambda x: "Low" if x<= 1 else 
                                                          ("High" if x >= 4 else "Average"))
Potential = df["Potential"] # keeping a copy for graphing next (in task 5-6)
plt.figure()
# This scatter plot shows the efficency of the two PCs clearly.
sns.relplot(x='PC1',y='PC2', # (NOTE: if the graph lags run this cell twice)
            hue="Potential", palette=["b","g","r"],  
            kind='scatter',alpha=0.75,
            height=5, aspect=1,
            data=df)
plt.title('PCs vs Sale-Price', fontsize = 20)
plt.show()
print(f'''The two PCs show that the inputs provide a good separation of high and low performing estates, 
with some overlap. even without using Sale-Price as an input. 
In general, the more you go up, the more likely the estate has high potential''')
df.drop(columns=['PC1','PC2'],inplace=True)
display(df)
```


    <Figure size 432x288 with 0 Axes>



    
![png](PredictingSalePrice_files/PredictingSalePrice_9_1.png)
    


    The two PCs show that the inputs provide a good separation of high and low performing estates, 
    with some overlap. even without using Sale-Price as an input. 
    In general, the more you go up, the more likely the estate has high potential
    



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Purchase-Date</th>
      <th>Purchase-Price</th>
      <th>Type</th>
      <th>Class</th>
      <th>Location</th>
      <th>Shape</th>
      <th>U-Index</th>
      <th>Proximity</th>
      <th>N-Rank</th>
      <th>P-Chance</th>
      <th>...</th>
      <th>Access</th>
      <th>Crime-Rate</th>
      <th>C-Rating</th>
      <th>Gov-Index</th>
      <th>Contour</th>
      <th>Garage</th>
      <th>Swimming</th>
      <th>Sale-Price</th>
      <th>Appreciation/Depreciation</th>
      <th>Potential</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1975</td>
      <td>2474.5</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>8171</td>
      <td>10</td>
      <td>0.473210</td>
      <td>...</td>
      <td>2</td>
      <td>59</td>
      <td>3</td>
      <td>4</td>
      <td>NA</td>
      <td>No</td>
      <td>NA</td>
      <td>11896.0</td>
      <td>3.807436</td>
      <td>Average</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1991</td>
      <td>2414.2</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>10290</td>
      <td>1</td>
      <td>0.064744</td>
      <td>...</td>
      <td>0</td>
      <td>24</td>
      <td>3</td>
      <td>1</td>
      <td>F</td>
      <td>NA</td>
      <td>NA</td>
      <td>17187.2</td>
      <td>6.119211</td>
      <td>High</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1989</td>
      <td>2453.3</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>9940</td>
      <td>8</td>
      <td>0.115741</td>
      <td>...</td>
      <td>2</td>
      <td>94</td>
      <td>3</td>
      <td>5</td>
      <td>NA</td>
      <td>No</td>
      <td>NA</td>
      <td>12756.4</td>
      <td>4.199690</td>
      <td>High</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1973</td>
      <td>2513.7</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>5</td>
      <td>69542</td>
      <td>4</td>
      <td>0.300323</td>
      <td>...</td>
      <td>1</td>
      <td>68</td>
      <td>1</td>
      <td>9</td>
      <td>C</td>
      <td>NA</td>
      <td>NA</td>
      <td>14722.9</td>
      <td>4.857063</td>
      <td>High</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981</td>
      <td>8691.7</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3328</td>
      <td>5</td>
      <td>0.017304</td>
      <td>...</td>
      <td>2</td>
      <td>88</td>
      <td>4</td>
      <td>8</td>
      <td>F</td>
      <td>NA</td>
      <td>NA</td>
      <td>17694.9</td>
      <td>1.035839</td>
      <td>Average</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4995</th>
      <td>1983</td>
      <td>6337.5</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>9507</td>
      <td>8</td>
      <td>0.432669</td>
      <td>...</td>
      <td>2</td>
      <td>88</td>
      <td>2</td>
      <td>2</td>
      <td>NA</td>
      <td>No</td>
      <td>NA</td>
      <td>15072.4</td>
      <td>1.378288</td>
      <td>Average</td>
    </tr>
    <tr>
      <th>4996</th>
      <td>2004</td>
      <td>11209.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>20864</td>
      <td>8</td>
      <td>0.009118</td>
      <td>...</td>
      <td>1</td>
      <td>6</td>
      <td>4</td>
      <td>6</td>
      <td>NA</td>
      <td>No</td>
      <td>NA</td>
      <td>13823.7</td>
      <td>0.233268</td>
      <td>Low</td>
    </tr>
    <tr>
      <th>4997</th>
      <td>1997</td>
      <td>1604.4</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>78957</td>
      <td>7</td>
      <td>0.561522</td>
      <td>...</td>
      <td>1</td>
      <td>83</td>
      <td>3</td>
      <td>8</td>
      <td>NA</td>
      <td>No</td>
      <td>NA</td>
      <td>11828.1</td>
      <td>6.372289</td>
      <td>High</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>2006</td>
      <td>7111.0</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>11546</td>
      <td>8</td>
      <td>0.011843</td>
      <td>...</td>
      <td>0</td>
      <td>16</td>
      <td>4</td>
      <td>5</td>
      <td>NA</td>
      <td>Yes</td>
      <td>NA</td>
      <td>19046.1</td>
      <td>1.678400</td>
      <td>Average</td>
    </tr>
    <tr>
      <th>4999</th>
      <td>1981</td>
      <td>4571.9</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>75949</td>
      <td>1</td>
      <td>0.658805</td>
      <td>...</td>
      <td>1</td>
      <td>86</td>
      <td>3</td>
      <td>2</td>
      <td>NA</td>
      <td>Yes</td>
      <td>NA</td>
      <td>11967.6</td>
      <td>1.617643</td>
      <td>Average</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 22 columns</p>
</div>


## Task-5
What are the common patterns for the low performance of the estates? Use plots to justify.

```python
# Rather than creating a new df, (reread th the file for graphing) and do the same as previous steps
# reread the file for graphing
df = pd.read_csv('HousingPrices_B.csv', delimiter = ',',keep_default_na=False) # Read and keep NA in file as values non NaNs
selected_columns = df.drop(columns=["Sale-Price"]) # dropping the output 
df["Purchase-Date"] = df["Purchase-Date"].apply(lambda x: x.replace("-","").lstrip("Jan").lstrip("January")
                                               .lstrip("Feb").lstrip("February").lstrip("Mar").lstrip("March")
                                               .lstrip("Apr").lstrip("April").lstrip("Jun").lstrip("June")
                                               .lstrip("Jul").lstrip("July").lstrip("Aug").lstrip("August")
                                               .lstrip("Sep").lstrip("September").lstrip("Oct").lstrip("October")
                                               .lstrip("Nov").lstrip("November").lstrip("Dec")
                                                .lstrip("December")).apply(pd.to_numeric)
# Divide the inputs to seperate them (to look better when graphed)
X1 = selected_columns.iloc[:,:3]
X2 = selected_columns.iloc[:,3:6]
X3 = selected_columns.iloc[:,6:9]
X4 = selected_columns.iloc[:,9:12]
X5 = selected_columns.iloc[:,12:15]
X6 = selected_columns.iloc[:,15:17] 
X7 = selected_columns.iloc[:,17:] 
#----------------------------------------------------------------------
# I created 'performance' column for seperating low performance estates
df['performance'] = Potential == "Low"
df['performance']= df['performance'].apply(lambda x: "Low" if x else "not Low")

# I choosed histograms (3d bivariate) because it clearly shows where tha data is concentrated. 
# This helped my to identify the pattern easily.
plt.figure()
sns.pairplot(x_vars=X1,y_vars="Sale-Price",kind="hist",height=3, aspect=2,data=df,hue="performance")
sns.pairplot(x_vars=X2,y_vars="Sale-Price",kind="hist",height=3, aspect=2,data=df,hue="performance")
sns.pairplot(x_vars=X3,y_vars="Sale-Price",kind="hist",height=3, aspect=2,data=df,hue="performance")
sns.pairplot(x_vars=X4,y_vars="Sale-Price",kind="hist",height=3, aspect=2,data=df,hue="performance")
sns.pairplot(x_vars=X5,y_vars="Sale-Price",kind="hist",height=3, aspect=2,data=df,hue="performance")
sns.pairplot(x_vars=X6,y_vars="Sale-Price",kind="hist",height=3, aspect=2,data=df,hue="performance")
sns.pairplot(x_vars=X7,y_vars="Sale-Price",kind="hist",height=3, aspect=2,data=df,hue="performance")
plt.show()

df.drop(columns='performance',inplace=True)
#-----------------------------------------------------------------------------------
# The patterns of Low performance are explained in extensive detail below:
### Sale-price Vs Purchase-date:
print(f'The purchase date does not seem to show a certain pattern for low performance.')
# the low performance is more common at the sale price between 10-15 thousand.
# and the concentration of not low gets higher the more the sale-price moves away from this value.

### Sale-price Vs Purchase-price:
print(f'Low performance is more common when the purchase price is above 6000.') 
# The low performance is more common when the purchase-price increases. 
# At the sale price between 10-15 thousand and purchase price between 4-8 thousand, 
# there is a high fluctuation in the performance.

### Sale-price Vs Type:
print(f'''Low performance is more common in Duplex, Flats, and Villas, but the type alone does not seem to determine
      the performance of the proerty''')
# Especially with sale price of 10,000 - 15,000
# Even though this is not expected, but the estates with low performance are clustered in these types  
# The duplexes-open lands-flats-villas. Have a not low performance mostly at lower sale price.
# And have high concentration of low performance at medium sale price.
# And at a low sale price, the vila type has low performance.

### Sale-price Vs Class:
print(f'Low performance is more common in residential (legal classification of the property).')
# This showes that the class can infact affect the performance of the property
# The residential class has a high concentration of low performance at intermediate sale price. 
# Low performance at low sale price except for the residential. 
# The industrial class has a weak concentration of low performance at different sale prices. 
# The commercial class has approximately the same distribution as the residential class.

### Sale-price Vs Location:
print(f'Low performance is more common in properties nearby the border of the city.')
# low performance is very common at Intermediate sale price despite the location.

### Sale-price Vs Shape:
print(f'Low performance is more common in irregular property shape.')
# At the intermediate sale price, low performance is very common despite the shape.

### Sale-price Vs U-index:
print(f'Utilities Index does not seem to show a certain pattern for low performance.')
# At the sale price from 10000 to 20000 the low performance is common among U Indecies from 1 to 5

### Sale-price Vs Proximity:
print(f"""Low performance is more common in Proximity to metro stationof approximately 20,000, but doesn't seem to have a
    strong pattern""")
# The low performance is common in the intermediate sale-price and less common.

### Sale-price Vs N-Rank:
print(f"""Low performance is more common in neighborhood facilities of rank 5, but doesn't seem to show a
    strong pattern""")
#Regardless of the rank, low performance is common at intermediate price.

### Sale-price Vs P-Chance:
print(f'''Probability of finding parking space on adjacent roads
      does not seem to show a certain pattern for low performance.''')
# Regardless of the P-Chance, low performance is common at intermediate price. 

### Sale-price Vs The year of construction(Built):
print(f'Year of construction does not seem to show a certain pattern for low performance.')
# Low performance is common at intermediate pice in the years from 1965 to 1995. 

### Sale-price Vs Renovate:
print(f"Low performance is equally common in properties that were renovated and those that weren't.")
# Renovation does not tell much about the performance

### Sale-price Vs Access:
print(f'Low performance is more common in Highway (direct access to the property)')

### Sale-price Vs Crime-Rate:
print(f'''Average number of crimes reported per year in the neighborhood does not seem to affect the performance
      of the property.''')
# At high price the low performance is less common despite the crime-rate.
# the crime rate does not tell much about the performance.
# low performance is more common among intermediate prices.

### Sale-price Vs C-Rating:
print(f'Pleasantness of the climate throughout the year does not seem to affect the performance much.')
# At high and low price low performance is uncommon, and in intermediate price the low performance is very common. 

### Sale-price Vs Gov-Index:
print(f'''Low performace is more common in properties with Gov-Index of 1-2
      (Expected level of government infrastructure project).''')

### Sale-price Vs Contour:
print(f'''Low performance is more common in properties at which it is not applicable to to have flatness of property
      (i.e. Villa, Flat, and Duplex.''')
# for F and C, the low performance exists in low concentration from approximately 13 to 21 thousands.
# otherwise there is no to little low performance in open land

### Sale-price Vs Garage:
print(f'Low performance is more common in properties that have a garage.')
# for no garage, there is no low performance above 22500 and below 8000 and the low concentration is common at 15000
# for NA there is no low performance at more than 23500 and below 11000 and low performance is from 13000-23000
# for properties with Garages, the low performance is very high from 10-15 thosand and doesn't exist above 24000

### Sale-price Vs Swimming:
print(f'Low performance is more common in properties at which it is not applicable to to have a swimming pool.')
#with NA at intermediate price, low is very common and it gets less common the more it moved from intermediate price
#It seems like with swimming or without there will be no major effect on the performance. 
```


    <Figure size 432x288 with 0 Axes>



    
![png](PredictingSalePrice_files/PredictingSalePrice_11_1.png)
    



    
![png](PredictingSalePrice_files/PredictingSalePrice_11_2.png)
    



    
![png](PredictingSalePrice_files/PredictingSalePrice_11_3.png)
    



    
![png](PredictingSalePrice_files/PredictingSalePrice_11_4.png)
    



    
![png](PredictingSalePrice_files/PredictingSalePrice_11_5.png)
    



    
![png](PredictingSalePrice_files/PredictingSalePrice_11_6.png)
    



    
![png](PredictingSalePrice_files/PredictingSalePrice_11_7.png)
    


    The purchase date does not seem to show a certain pattern for low performance.
    Low performance is more common when the purchase price is above 6000.
    Low performance is more common in Duplex, Flats, and Villas, but the type alone does not seem to determine
          the performance of the proerty
    Low performance is more common in residential (legal classification of the property).
    Low performance is more common in properties nearby the border of the city.
    Low performance is more common in irregular property shape.
    Utilities Index does not seem to show a certain pattern for low performance.
    Low performance is more common in Proximity to metro stationof approximately 20,000, but doesn't seem to have a
        strong pattern
    Low performance is more common in neighborhood facilities of rank 5, but doesn't seem to show a
        strong pattern
    Probability of finding parking space on adjacent roads
          does not seem to show a certain pattern for low performance.
    Year of construction does not seem to show a certain pattern for low performance.
    Low performance is equally common in properties that were renovated and those that weren't.
    Low performance is more common in Highway (direct access to the property)
    Average number of crimes reported per year in the neighborhood does not seem to affect the performance
          of the property.
    Pleasantness of the climate throughout the year does not seem to affect the performance much.
    Low performace is more common in properties with Gov-Index of 1-2
          (Expected level of government infrastructure project).
    Low performance is more common in properties at which it is not applicable to to have flatness of property
          (i.e. Villa, Flat, and Duplex.
    Low performance is more common in properties that have a garage.
    Low performance is more common in properties at which it is not applicable to to have a swimming pool.
    

## Task-6
What are the common patterns for the high performance of the estates? Use plots to justify.

```python
# I created 'performance' column for seperating high performance
df['performance'] = Potential == "High"
df['performance']=df['performance'].apply(lambda x: "High" if x else "not High")

# I choosed histograms (3d bivariate) because it clearly shows where tha data is concentrated. 
# This helped my to identify the pattern easily.
plt.figure()
sns.pairplot(x_vars=X1,y_vars="Sale-Price",kind="hist",height=3, aspect=2,data=df,hue="performance")
sns.pairplot(x_vars=X2,y_vars="Sale-Price",kind="hist",height=3, aspect=2,data=df,hue="performance")
sns.pairplot(x_vars=X3,y_vars="Sale-Price",kind="hist",height=3, aspect=2,data=df,hue="performance")
sns.pairplot(x_vars=X4,y_vars="Sale-Price",kind="hist",height=3, aspect=2,data=df,hue="performance")
sns.pairplot(x_vars=X5,y_vars="Sale-Price",kind="hist",height=3, aspect=2,data=df,hue="performance")
sns.pairplot(x_vars=X6,y_vars="Sale-Price",kind="hist",height=3, aspect=2,data=df,hue="performance")
sns.pairplot(x_vars=X7,y_vars="Sale-Price",kind="hist",height=3, aspect=2,data=df,hue="performance")
plt.show()

df.drop(columns='performance',inplace=True)
#-----------------------------------------------------------------------------------
# The patterns of High performance are explained in extensive detail below:
### Sale-price Vs Purchase-Date:
print(f'The purchase date does not seem to show a certain pattern for high performance.')

### Sale-price Vs Purchase-Price:
print(f'High performance is more common in properties with purchase price below 4000')
#After 4000 There is no high performance. concentration of high and not high increase in the intermediate sale-price.

### Sale-price Vs Type:
print(f'''High performance is more common in open land, but the type alone does not seem to determine
      the performance of the proerty''')
# for duplexes-open lands-vilas-flats the high performance is in common in the high sale-price region.
# The high performance is more common at the sale price between 10-20 thousands for all types.

### Sale-price Vs Class:
print(f'High performance is also more common in residential (legal classification of the property).')
# This showes that the class can infact affect the performance of the property
# Despite the class, high performance is more common below 20000 sale-price.

### Sale-price Vs location:
print(f'High performance is more common in properties nearby the center of the city.')
#Despite the location, the high performance is  more common below 20000 sale price.
#and between 10 -15 thousands the high tends to be more common

### Sale-price Vs Shape:
print(f'High performance is more common in rectangle property shape.')
#the high performance is concentrated more at sale price between 10-15 thousands.
#and gets less common the more the sale price diverges grom this value, above 20000 sale price.

### Sale-price Vs U-Index:
print(f'''High performance is somewhat more common in U-Index of 5, but in general,
      Utilities Index does not seem to show a certain pattern for high performance.''')
#The high performance is common at price from 7000 to 22000 for all U-inexes.
#for 1 2 3 4 indexes there is some non existence of high performance in around 5000 sale price.

### Sale-price Vs Proximity:
print(f"Proximity to metro station does not seem to show a certain patter for high performing properties")

### Sale-price Vs N-Rank:
print(f'High performance is more common in neighborhood facilities of rank 1')

### Sale-price Vs P-Chance:
print(f'''Probability of finding parking space on adjacent roads does not seem to show a certain pattern
for high performance. Yet, when there is a guaranteed availability of parking space, it's usally a high performing''')
#The high performance is common between 10-20 thousands with some differences in some areas

### Sale-price Vs Construction year(Built):
print(f'Year of construction does not seem to show a certain pattern for high performance.')
#At years before the 1960s the high performance seems uncommon despite the sale price.
#between the year 1970 and 2005 the high performance is more common between 9000 to 18000 price
#After the year 2005 there is a weak concentration of high performance.

### Sale-price Vs Renovate:
print(f"High performance is more common in properties that were not renovated.")
# Renovation does not tell much about the performance
# At high price the high performnce is uncommon, and in the 
# Unrenovated estates seem to have approximately the same distribution of high and not high performance 

### Sale-price Vs Access:
print(f'High performance is somewhat more common in street (direct access to the property)')
# In Street, below 6000 sale price the high performance does not exist.
# While between 6000 and 20000 the high performance is common for all the access ways 

### Sale-price Vs Crime-Rate:
print(f'''Average number of crimes reported per year in the neighborhood does not seem to affect the performance
      of the property.''')
#At low and high sale price the high performance is uncommon despite the crime rate, and between 10 to 20000, 
#the high performance is more common over all the rates with some exceptions.

### Sale-price Vs C-Rating:
print(f'High performance is more common when pleasantness of the climate is at C-Rating of (4-5)') 
#At low price  and high price, high performance is uncommon at all ratings. And in intermediate price the high 

### Sale-price Vs Gov-Index:
print(f'''High performace is more common in properties with Gov-Index of 9-10
      (Expected level of government infrastructure project).''')

### Sale-price Vs Contour:
print(f'''High performance is more common in properties at which it is not applicable to to have flatness of property
      (i.e. Villa, Flat, and Duplex.''')
# This shows that the Contour does not tell much alone

### Sale-price Vs Garage:
print(f'High performance is somewhat common in properties that do not have a garage.')
# for no garage, the performance tend to be not high t more than 22500 sale price. And the same goes for estates 
#with garages. and for NA values the performance fluctuates between high and not high above 20000 sale price. 

### Sale-price Vs Swimming:
print(f'High performance is more common in properties at which it is not applicable to to have a swimming pool.')
# For NA, the performance fluctuates between high and not high above 23000 and below this value the performance
# tends to be high. For Swimming, there is no data at high price but above 15 -18 thousand the price tends
#not to be high. For No swimming also, there is no data at high price. However,the performance fluctuates between high 
# Otherwise the performance tend to be high with low concentration. 
```


    <Figure size 432x288 with 0 Axes>



    
![png](PredictingSalePrice_files/PredictingSalePrice_13_1.png)
    



    
![png](PredictingSalePrice_files/PredictingSalePrice_13_2.png)
    



    
![png](PredictingSalePrice_files/PredictingSalePrice_13_3.png)
    



    
![png](PredictingSalePrice_files/PredictingSalePrice_13_4.png)
    



    
![png](PredictingSalePrice_files/PredictingSalePrice_13_5.png)
    



    
![png](PredictingSalePrice_files/PredictingSalePrice_13_6.png)
    



    
![png](PredictingSalePrice_files/PredictingSalePrice_13_7.png)
    


    The purchase date does not seem to show a certain pattern for high performance.
    High performance is more common in properties with purchase price below 4000
    High performance is more common in open land, but the type alone does not seem to determine
          the performance of the proerty
    High performance is also more common in residential (legal classification of the property).
    High performance is more common in properties nearby the center of the city.
    High performance is more common in rectangle property shape.
    High performance is somewhat more common in U-Index of 5, but in general,
          Utilities Index does not seem to show a certain pattern for high performance.
    Proximity to metro station does not seem to show a certain patter for high performing properties
    High performance is more common in neighborhood facilities of rank 1
    Probability of finding parking space on adjacent roads does not seem to show a certain pattern
    for high performance. Yet, when there is a guaranteed availability of parking space, it's usally a high performing
    Year of construction does not seem to show a certain pattern for high performance.
    High performance is more common in properties that were not renovated.
    High performance is somewhat more common in street (direct access to the property)
    Average number of crimes reported per year in the neighborhood does not seem to affect the performance
          of the property.
    High performance is more common when pleasantness of the climate is at C-Rating of (4-5)
    High performace is more common in properties with Gov-Index of 9-10
          (Expected level of government infrastructure project).
    High performance is more common in properties at which it is not applicable to to have flatness of property
          (i.e. Villa, Flat, and Duplex.
    High performance is somewhat common in properties that do not have a garage.
    High performance is more common in properties at which it is not applicable to to have a swimming pool.
    


```python
# Before doing (Task 7) the regression analysis:
df = pd.read_csv('HousingPrices_B.csv', delimiter = ',',keep_default_na=False)
# rereading last time to encode using one hot encoding to include all inputs in the regression (train file)
df["Purchase-Date"] = df["Purchase-Date"].apply(lambda x: x.replace("-","").lstrip("Jan").lstrip("January")
                                               .lstrip("Feb").lstrip("February").lstrip("Mar").lstrip("March")
                                               .lstrip("Apr").lstrip("April").lstrip("Jun").lstrip("June")
                                               .lstrip("Jul").lstrip("July").lstrip("Aug").lstrip("August")
                                               .lstrip("Sep").lstrip("September").lstrip("Oct").lstrip("October")
                                               .lstrip("Nov").lstrip("November").lstrip("Dec")
                                                .lstrip("December")).apply(pd.to_numeric)
hot_df = pd.get_dummies(df, columns=['Class','Location','Shape','Access','Contour','Garage','Swimming']
                        ,drop_first=True)
Type_mapping = {'villa':1,'duplex':2,'flat':3,'open land':4}
hot_df["Type"] = df["Type"].map(Type_mapping)
hot_df["SP"] = hot_df["Sale-Price"]
hot_df.drop(columns="Sale-Price", inplace=True)
hot_df["Sale-Price"] = hot_df["SP"]
hot_df.drop(columns="SP", inplace=True)
#---------------------------------------------------------------------------------------------------------------------
# Reading file C (the test file) and applying all previous steps to ensure consistency to predict values 
idf = pd.read_csv('HousingPrices_C.csv', delimiter = ',',keep_default_na=False)# Read and keep NA in file as values non NaNs
idf["Purchase-Date"] = idf["Purchase-Date"].apply(lambda x: x.replace("-","").lstrip("Jan").lstrip("January")
                                               .lstrip("Feb").lstrip("February").lstrip("Mar").lstrip("March")
                                               .lstrip("Apr").lstrip("April").lstrip("Jun").lstrip("June")
                                               .lstrip("Jul").lstrip("July").lstrip("Aug").lstrip("August")
                                               .lstrip("Sep").lstrip("September").lstrip("Oct").lstrip("October")
                                               .lstrip("Nov").lstrip("November").lstrip("Dec")
                                                .lstrip("December")).apply(pd.to_numeric)
idf = pd.get_dummies(idf, columns=['Class','Location','Shape','Access','Contour','Garage','Swimming']
                        ,drop_first=True)
idf["Type"] = idf["Type"].map(Type_mapping)
idf["SP"] = idf["Sale-Price"]
idf.drop(columns="Sale-Price", inplace=True)
idf["Sale-Price"] =idf["SP"]
idf.drop(columns="SP", inplace=True)
#-----------------------------------------------------------------------------------------------------------
# First step in doing regression analysis is to split the Train/Test datasets
# Trian/Test spliting using Sample (DataB is the train, DataC is the test)

# Using all the inputs in the X. 
X_train = hot_df.drop(columns=['Sale-Price']).values # using B file and including all inputs (after encoding)
# Using the output (Sale-Price) in the Y
y_train = hot_df['Sale-Price'].values # the values of Sale-Price in the training dataset

X_test = idf.drop(columns=['Sale-Price']).values # using C file and including all inputs (after encoding)
y_test = idf['Sale-Price'].values # the values of Sale-Price in the training dataset
#--------------------------------------------------------------------------------------------------
# After splitting the datasets, I need to normalize the values to ensure accurate results
# Scaling the Train - Test splits 
scaler = StandardScaler() # Standard scaling helped yield better regression results.
scaler.fit(np.c_[X_train,y_train])

A_train = scaler.transform(np.c_[X_train,y_train])
X_train = A_train[:,:-1]
y_train = A_train[:,-1]

A_test = scaler.transform(np.c_[X_test,y_test])
X_test = A_test[:,:-1]
y_test = A_test[:,-1]
#----------------------------------------------------------------------------------
# Finally, Testing all regression methods to find which one is more suitable for predicting (Continue below)
```

## Task-7
From the input and output columns, identify how the input variables together are related to the output. Assume that all the input variables are relevant to output variable (Sale-Price).

```python
# To identify how all variables are related to the output, I firstly used 
## OLS (Ordinary Least Squares Linear Regression)
from sklearn.linear_model import LinearRegression
reg1 = LinearRegression().fit(X_train, y_train)
y_pred1 = reg1.predict(X_test)
print('The MSE using OLS is:', mean_squared_error(y_test, y_pred1))
MSE1= mean_squared_error(y_test, y_pred1)
```

    The MSE using OLS is: 0.1622301783308531
    

## Task-8
It was observed that some of the input columns are correlated, and this may make the above analysis unreliable. 
Redo Task-(7), with the consideration of correlation issue between input variables.

```python
# Since some inputs might be correlated with each other, I need to use Ridge regression
## Ridge
from sklearn.linear_model import RidgeCV
reg2 = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2], cv=10).fit(X_train, y_train)
y_pred2 = reg2.predict(X_test)
print('The MSE using Ridge is:', mean_squared_error(y_test, y_pred2))
MSE2 = mean_squared_error(y_test, y_pred2)
```

    The MSE using Ridge is: 0.16221229262980064
    

## Task-9
It was observed that some of the input columns may not be relevant to the output variable,and this may make the above analysis unreliable. 
Redo Task-(7), with the consideration of possible unrelated input variables.

```python
# Because some inputs might not be really related to the output, I need to consider using Lasso regression
## Lasso
from sklearn.linear_model import LassoCV
reg3 = LassoCV(cv=10, random_state=0).fit(X_train, y_train)
y_pred3 = reg3.predict(X_test)
print('The MSE using Lasso is:', mean_squared_error(y_test, y_pred3))
MSE3 = mean_squared_error(y_test, y_pred3)
#----------------------------------------------------------------------
# To evaluate the best regression model, I selected the one with smallest mean squared error (MSE).
# I also tried using R^2 but MSE is a better measure with this data
# This will provide the most accurate prediction
MSE = {"OLS":MSE1,"Ridge":MSE2,"Lasso":MSE3}
print(f'\nThe best regrssion is {min(MSE, key=MSE.get)} with MSE of: {min(MSE.values())}')
print(f'Lasso performs better than OLS and Ridge for this data.')
#--------------------------------------------------------------------
# Printing the regression coefficients:
best_beta =  np.round(reg3.coef_,2)
best_beta_0 = np.round(reg3.intercept_,2)
print(f'The best values for the estimates are :', best_beta_0, best_beta.tolist())
```

    The MSE using Lasso is: 0.1621181495723582
    
    The best regrssion is Lasso with MSE of: 0.1621181495723582
    Lasso performs better than OLS and Ridge for this data.
    The best values for the estimates are : 0.0 [0.17, 0.34, -0.35, 0.11, 0.16, -0.04, 0.07, 0.02, -0.01, -0.17, 0.08, 0.08, -0.03, 0.11, 0.36, 0.0, 0.4, 0.3, -0.32, -0.21, 0.04, -0.57, -0.02, 0.0, 0.56, -0.04]
    

## Task-10
Predict the estimated Sale-Price values given in data (C) file. 
Consider all the numerical and categorical variables for the analysis. If you skip any column, then provide strong justification. 


```python
# In predicting, the scaled values were used. Therefore, the inverse scale must be used 
# to get an approximation to the actual predicted value.

# The inverse scaling succeeded to provude values close to the real ones.
y_pred = scaler.inverse_transform(np.c_[X_test,y_pred3])[:,-1]

idf['Predicted_Sale-Price']= y_pred.round().astype(int) # (see the last column)
# Displaying sample results of the predicted prices regression model compared to actualy prices
display(idf.sample(10).loc[:, 'Sale-Price':'Predicted_Sale-Price'])
```

|      | Sale-Price | Predicted_Sale-Price |
|------|------------|----------------------|
| 99   | 11326.2    | 11741                |
| 1833 | 23211.4    | 23481                |
| 1377 | 13376.2    | 13606                |
| 577  | 13776.9    | 14242                |
| 637  | 19497.8    | 15248                |
| 956  | 13943.2    | 14298                |
| 695  | 9024.4     | 9520                 |
| 518  | 14281.4    | 14699                |
| 848  | 14891.1    | 15082                |
| 368  | 9361.9     | 9766                 |



## 🛠 Skills Used
Python, Correlation Analysis, Data Cleaning, Data Visualization, Data Wrangling, 

Descriptive Statistical Analysis, Exploratory Data Analysis, Filtering Noisy Data, Handling Missing Values, Label Encoding ,

Lasso Regression, Linear Regression, Machine Learning, Normalization, One-Hot Encoding, Principal Component Analysis, Ridge Regression
## 🚀 About Me
👋 Hi, I’m @Raed-Alshehri

👀 I’m interested in data science, machine learning, and statistics.

🌱 I’m applying my skills in the data analytics field using Python, R, and SQL


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://raed-alshehri.github.io/RaedAlshehri.github.io/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/raedalshehri/)


## Feedback

If you have any feedback, please reach out to me at alshehri.raeda@gmail.com


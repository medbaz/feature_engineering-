import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "Salary Data.csv"

# Load the latest version
salary_data = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "rkiattisak/salaly-prediction-for-beginer",
  file_path
)





# NOTE UNIVARIANT PLOTS 

"""
sns.barplot(x='Gender', y='Salary' , estimator=np.max, data=salary_data )
print(salary_data["Salary"].mean())
? Add a title
plt.title("Average Salary by Job Title")
? Show the plot
plt.show()
"""


"""
NOTE countplot compte le nombre de chaque category dans un coloumn 
sns.countplot(x='Education Level',data=salary_data)
plt.show()
"""

"""
NOTE histplot instead of distplot visuliser la repartition des données
sns.histplot(salary_data,x='Salary',binwidth=10000)
plt.show()
"""

""" 
NOTE : kernel density estimation (KDE) 
NOTE : montre l'estimation de chaque category d'un column dans un courb continue
sns.kdeplot(data=salary_data , x='Salary',fill=True)
plt.show()
"""





# ? BIVARIANT PLOTS

"""
NOTE shows
NOTE A bivariate relationship (scatter plot, hexbin, KDE, etc.) between two variables. AT THE MIDDLE
NOTE AND
NOTE Univariate distributions (histograms like showing the count of each variable category or KDEs) for each variable on the margins AT THE SIDES

sns.jointplot(x='Age', y='Salary', data=salary_data, )
plt.title('Income Density by Education Level')
plt.show()
"""


"""
NOTE shows bivariant plot with an index of correlation between the two vars
sns.regplot(data=salary_data,x='Salary',y='Age')
plt.xlabel("the salary ")
plt.ylabel("the age ")

sns.lmplot(data=salary_data,x='Salary',y='Age')
plt.xlabel("the salary ")
plt.ylabel("the age ")

plt.show()
"""


"""
NOTE shows 6 plots , univariants in the diagonal and bivar on the rest 
sns.pairplot(data=salary_data[['Age','Salary','Years of Experience']])
plt.show()
"""


"""
NOTE MULTIVARIANT PLOTS
g=sns.FacetGrid(data=salary_data,col="Gender")
g.map(plt.scatter,'Age','Salary')
plt.show()

"""




import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "titanic.csv"

titanicDF = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "markmedhat/titanic",
  file_path,
)

"""




NOTE  Drive Information 

print("\n \n")
print(" EXECUTER   titanicDF.info ")
print(titanicDF.info())

print("\n \n")
print(" EXECUTER   titanicDF.describe() ")
print(titanicDF.describe())

print("\n \n")
print(" EXECUTER   titanicDF.isnull().sum() ")
print(titanicDF.isnull().sum())




NOTE  dropping Embarked column 
print("\n \n")
print(" EXECUTER   titanDFdrop.isnull().sum() ")
titanDFdrop = titanicDF.dropna(subset=["Embarked"])
print(titanDFdrop.isnull().sum())


print("\n \n")
print(" EXECUTER   titanDFdrop['Sex'].value_counts() ")
print(titanDFdrop["Sex"].value_counts())


print("\n \n")
# sns.histplot(x='Age' , data=titanDFdrop[(titanDFdrop['Sex'] == 'female')], binwidth=10)
# plt.xlabel = 'female ages'
# sns.histplot(x='Age' , data=titanDFdrop[(titanDFdrop['Sex'] == 'male')], binwidth=10)
# plt.xlabel = 'male ages'
# plt.show()


NOTE: REPLACE AGE VALUES
themean = titanDFdrop['Age'].mean() 
titanDFdrop.loc[(titanDFdrop['Age'].isnull() == True) , 'Age'] = themean



NOTE: DROPE COLUMN CABIN 
del titanDFdrop["Cabin"]

classAGE = titanDFdrop.groupby(['Pclass','Sex'])['Age'].mean()
print(classAGE)

for (Pclass , Sex ) , Age in classAGE.items():
    titanDFdrop.loc[(titanDFdrop['Pclass'] == Pclass) & (titanDFdrop["Sex"] == Sex) & (titanDFdrop['Age'].isnull() ),'Age'] = Age


sumOFpreviousNULLvalues = 0
for (Pclass , Sex) , Age in classAGE.items():
    nullCOUNT = titanDFdrop.loc[(titanDFdrop['Age'] == Age),'Age'].value_counts()
    sumOFpreviousNULLvalues += nullCOUNT[Age]

print("the count of previous null values of each case from classAGE " , sumOFpreviousNULLvalues )
print(titanDFdrop.head())
print(  titanDFdrop.isnull().sum())

"""



"""
! imputation par la moyenne / mediane
  NOTE : EXEMPLE REPLACING MISSING AGE VALUES
..  themean = titanDFdrop['Age'].mean() 
..  titanDFdrop.loc[(titanDFdrop['Age'].isnull() == True) , 'Age'] = themean 



! imputation par la plus frequant au / constant
 NOTE : EXEMPLE REPLACING MISSING AGE VALUES WITH ZEROS 
..  titanDFdrop['Age'].fillna(0 ou constant )



NOTE : usecols =['col name'] will only read that column from data base
..  like this pd.read_csv('data_name.csv',usecols=[col name])


! capture des valeurs manquant par une nouvelle column 
.. You can still fill missing "Cabin" values (e.g., with 'Unknown'), but now your
.. model has a separate signal to indicate that the value was originally missing.
titanicDF["newAgeCol"] = np.where(titanicDF["Cabin"].isnull() , 1 , 0)


! capture des valeurs manquant par la fin de distribution
.. mean imputation by the max of data distribution that is calculated by 
max_distribution = titanicDF["Age"].mean() + 3*titanicDF["Age"].std()

! imputation par la plus frequant/ mode

Embarked_count = titanicDF["Embarked"].value_counts()
count_most_class = 0
most_frequent = None
for theclass , repeat in Embarked_count.items() :
    if repeat > count_most_class :
        count_most_class = repeat
        most_frequent = theclass
        print(" class " ,  theclass , " repeat " , repeat )

titanicDF["Embarked"].fillna(most_frequent)
print(titanicDF["Embarked"].isnull().sum())
print(titanicDF.head())
print(titanicDF["Embarked"].value_counts())

""" 



"""
! imputation par regression

from sklearn.linear_model import LinearRegression

tips = sns.load_dataset('tips')
tips_df = pd.DataFrame(tips)
.. print the data that will be none to compare it later with prediction
print(tips_df.loc[220:230,'tip'])

.. correlation matrix that shows that tips and bill are related
print(tips_df.corr(numeric_only=True))
.. turn some tip data to none tu predict them later 
tips_df.loc[220:230,'tip'] = np.nan
print(tips_df.isnull().sum())

tip_Y = tips_df.loc[(tips_df['tip'].isnull() == False ) , 'tip']
bill_X = tips_df.loc[(tips_df['tip'].isnull() == False ) , 'total_bill']



model = LinearRegression().fit( bill_X.values.reshape(-1, 1),tip_Y)
.. shold switch model input to 2d even if it has one feature using reshape
* Why do we need .reshape(-1, 1)?
* bill_X is a pandas Series, which is 1D: shape = (n_samples,)
* Scikit-learn expects input features to be 2D: shape = (n_samples, n_features)
* -1 tells NumPy to automatically figure out the number of rows based on the original size
* 1 means 1 column
bill_mistip =  tips_df.loc[(tips_df['tip'].isnull() == True ) , 'total_bill']
predictions = model.predict(bill_mistip.values.reshape(-1, 1))
print(predictions)
"""


"""
! imputation par knn 
.. on utilise KNNImputer to calculate the nearest neibors based on the n_neighbors number on neibors and fill the missing data with the mean 
from sklearn.impute import KNNImputer
import numpy as np

X = [
    [9, 9, np.nan],
    [1, 1,  1],
    [4, 4, 4],
    [2, 2,  2],
    [6, 6, 6]
]

print("Original Data (X):\n", X)
print("=" * 30)

# Imputation using 1 neighbor
imputer_1 = KNNImputer(n_neighbors=1)
imputed_with_1 = imputer_1.fit_transform(X)
print("\nImputed with 1 Neighbor:\n", imputed_with_1)

# Imputation using 2 neighbors
imputer_2 = KNNImputer(n_neighbors=2)
imputed_with_2 = imputer_2.fit_transform(X)
print("\nImputed with 2 Neighbors:\n", imputed_with_2)

"""





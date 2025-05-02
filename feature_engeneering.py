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



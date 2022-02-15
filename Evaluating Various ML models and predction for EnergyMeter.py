
 ### Evaluating Various MachineLearning models and predction for EnergyMeter ###


#importing libraries 
#importing datasets
import pandas as pd

dataset = pd.read_csv("C:/Users/ELCOT/Downloads/Energy Meter.csv")

fileName = "Energy Meter.csv"
names = ['Voltage','Current','Power','Class']
dataset = pd.read_csv(fileName, names=names)
dataset

##########################
####summarising dataset
print(dataset.shape)
print(dataset.head(10))
print(dataset.describe)
print(dataset.groupby('Class').size())

####visuvalising the data
from pandas.plotting import scatter_matrix
from matplotlib import pyplot


dataset.plot(kind='bar',subplots=True,layout=(2,2))
pyplot.title('BAR PLOT')
pyplot.show()

dataset.hist()
pyplot.title('HISTOGRAM PLOT')
pyplot.show()

####scatter matrix library
scatter_matrix(dataset)
pyplot.title('SCATTER PLOT')
pyplot.show()




##############################
####importing machinelearning algarithms

#scikit learning
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


###importing train, test and split values
array = dataset.values
x = array[:,0:3]
y = array[:,3]
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.20, random_state=1, shuffle=True)


###using algorithms for models
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []
res = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=None)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    res.append(cv_results.mean())
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    
pyplot.ylim(.990, .999)
pyplot.bar(names, res, color='green', width = 0.6)

pyplot.title('Algorithm Comparison')
pyplot.show()
    


########################
#testing & deploiting using various algorithms

####using SVM (Support vector machine)
      #SVC (support vector classification)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC    
    
    
dataset1 = pd.read_csv("C:/Users/ELCOT/Downloads/Energy Meter.csv")    
    
url = "Energy Meter.csv"
names = ['Voltages', 'Current', 'Power', 'Class']
dataset1 = pd.read_csv(url,names=names)
    
array = dataset.values
x = array[:,0:3]
y = array[:,3]
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.50, random_state=1)

model = SVC(gamma='auto')  
model.fit(x_train, y_train)


###storing to local drive
import pickle
fileName = 'model.pkl'
pickle.dump(model, open(fileName, 'wb'))


###for load the model
loaded_model = pickle.load(open(fileName, 'rb'))
result = loaded_model.score(x_validation, y_validation)
print(result)




####using LinearDiscriminantAnalysis algarithms

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis   
    
    
dataset1 = pd.read_csv("C:/Users/ELCOT/Downloads/Energy Meter.csv")    
    
url = "Energy Meter.csv"
names = ['Voltages', 'Current', 'Power', 'Class']
dataset1 = pd.read_csv(url,names=names)
    
array = dataset.values
x = array[:,0:3]
y = array[:,3]
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.50, random_state=1)


model2 = LinearDiscriminantAnalysis()
model2.fit(x_train, y_train)

fileName = 'model2.pkl'
pickle.dump(model2, open(fileName, 'wb'))


loaded_model2 = pickle.load(open(fileName, 'rb'))
result = loaded_model2.score(x_validation, y_validation)
print(result)



################
###prediction

value = [[215.0313, 0.17388, 37.38964]]
predictions = model.predict(value)
print(predictions[0])


value = [[215.0313, 0.850669, 182.9205]]
predictions = model.predict(value)
print(predictions[0])

value = [[213.8814, 0.252315, 53.96548544]]
predictions = model.predict(value)
print(predictions[0])



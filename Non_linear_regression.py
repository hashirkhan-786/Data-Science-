1 Non Linear Regression Analysis
Estimated time needed: 20 minutes
1.1 Objectives
After completing this lab you will be able to:
• Differentiate between linear and non-linear regression
• Use non-linear regression model in Python
If the data shows a curvy trend, then linear regression will not produce very accurate results when
compared to a non-linear regression since linear regression presumes that the data is linear. Let’s
learn about non linear regressions and apply an example in python. In this notebook, we fit a
non-linear model to the datapoints corrensponding to China’s GDP from 1960 to 2014.
Importing required libraries
[ ]: import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
Although linear regression can do a great job at modeling some datasets, it cannot be used for
all datasets. First recall how linear regression, models a dataset. It models the linear relationship
between a dependent variable y and the independent variables x. It has a simple equation, of degree
1, for example y = 2x + 3.
[ ]: x = np.arange(-5.0, 5.0, 0.1)
##You can adjust the slope and intercept to verify the changes in the graph
y = 2*(x) + 3
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
#plt.figure(figsize=(8,6))
plt.plot(x, ydata, 'bo')
plt.plot(x,y, 'r')
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()
1
Non-linear regression is a method to model the non-linear relationship between the independent
variables x and the dependent variable y. Essentially any relationship that is not linear can be
termed as non-linear, and is usually represented by the polynomial of k degrees (maximum power
of x). For example:
y = ax3 + bx2 + cx + d
Non-linear functions can have elements like exponentials, logarithms, fractions, and so on. For
example:
y = log(x)
We can have a function that’s even more complicated such as :
y = log(ax3 + bx2 + cx + d)
Let’s take a look at a cubic function’s graph.
[ ]: x = np.arange(-5.0, 5.0, 0.1)
##You can adjust the slope and intercept to verify the changes in the graph
y = 1*(x**3) + 1*(x**2) + 1*x + 3
y_noise = 20 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata, 'bo')
plt.plot(x,y, 'r')
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()
As you can see, this function has x
3 and x
2 as independent variables. Also, the graphic of this
function is not a straight line over the 2D plane. So this is a non-linear function.
Some other types of non-linear functions are:
1.1.1 Quadratic
Y = X2
[ ]: x = np.arange(-5.0, 5.0, 0.1)
##You can adjust the slope and intercept to verify the changes in the graph
y = np.power(x,2)
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata, 'bo')
plt.plot(x,y, 'r')
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
2
plt.show()
1.1.2 Exponential
An exponential function with base c is defined by
Y = a + bcX
where b ff0, c > 0 , c ff1, and x is any real number. The base, c, is constant and the exponent, x, is
a variable.
[ ]: X = np.arange(-5.0, 5.0, 0.1)
##You can adjust the slope and intercept to verify the changes in the graph
Y= np.exp(X)
plt.plot(X,Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()
1.1.3 Logarithmic
The response y is a results of applying the logarithmic map from the input x to the output y. It
is one of the simplest form of log(): i.e.
y = log(x)
Please consider that instead of x, we can use X, which can be a polynomial representation of the
x values. In general form it would be written as
y = log(X) (1)
[ ]: X = np.arange(-5.0, 5.0, 0.1)
Y = np.log(X)
plt.plot(X,Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()
1.1.4 Sigmoidal/Logistic
Y = a +
b
1 + c
(X−d)
3
[ ]: X = np.arange(-5.0, 5.0, 0.1)
Y = 1-4/(1+np.power(3, X-2))
plt.plot(X,Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()
2 Non-Linear Regression example
For an example, we’re going to try and fit a non-linear model to the datapoints corresponding
to China’s GDP from 1960 to 2014. We download a dataset with two columns, the first, a year
between 1960 and 2014, the second, China’s corresponding annual gross domestic income in US
dollars for that year.
[ ]: import numpy as np
import pandas as pd
#downloading dataset
!wget -nv -O china_gdp.csv https://cf-courses-data.s3.us.cloud-object-storage.
,→appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/
,→Module%202/data/china_gdp.csv
df = pd.read_csv("china_gdp.csv")
df.head(10)
Did you know? When it comes to Machine Learning, you will likely be working with large
datasets. As a business, where can you host your data? IBM is offering a unique opportunity for
businesses, with 10 Tb of IBM Cloud Object Storage: Sign up now for free
2.0.1 Plotting the Dataset
This is what the datapoints look like. It kind of looks like an either logistic or exponential function.
The growth starts off slow, then from 2005 on forward, the growth is very significant. And finally,
it decelerate slightly in the 2010s.
[ ]: plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()
4
2.0.2 Choosing a model
From an initial look at the plot, we determine that the logistic function could be a good approximation, since it has the property of starting with a slow growth, increasing growth in the middle,
and then decreasing again at the end; as illustrated below:
[ ]: X = np.arange(-5.0, 5.0, 0.1)
Y = 1.0 / (1.0 + np.exp(-X))
plt.plot(X,Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()
The formula for the logistic function is the following:
Yˆ =
1
1 + e
β_1(X−β_2)
β_1: Controls the curve’s steepness,
β_2: Slides the curve on the x-axis.
2.0.3 Building The Model
Now, let’s build our regression model and initialize its parameters.
[ ]: def sigmoid(x, Beta_1, Beta_2):
y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
return y
Lets look at a sample sigmoid line that might fit with the data:
[ ]: beta_1 = 0.10
beta_2 = 1990.0
#logistic function
Y_pred = sigmoid(x_data, beta_1 , beta_2)
#plot initial prediction against datapoints
plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')
Our task here is to find the best parameters for our model. Lets first normalize our x and y:
[ ]: # Lets normalize our data
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)
5
How we find the best parameters for our fit line? we can use curve_fit which uses nonlinear least squares to fit our sigmoid function, to data. Optimal values for the parameters so that
the sum of the squared residuals of sigmoid(xdata, *popt) - ydata is minimized.
popt are our optimized parameters.
[ ]: from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)
#print the final parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))
Now we plot our resulting regression model.
[ ]: x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()
2.1 Practice
Can you calculate what is the accuracy of our model?
[ ]: # write your code here
Click here for the solution
# split data into train/test
msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]
# build the model using train set
popt, pcov = curve_fit(sigmoid, train_x, train_y)
# predict using test set
y_hat = sigmoid(test_x, *popt)
# evaluation
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
from sklearn.metrics import r2_score
6
print("R2-score: %.2f" % r2_score(y_hat , test_y) )
Want to learn more?
IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by
groups, by systems – by your enterprise as a whole. A free trial is available through this course,
available here: SPSS Modeler
Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson
Studio is IBM’s leading cloud solution for data scientists, built by data scientists. With Jupyter
notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio
enables data scientists to collaborate on their projects without having to install anything. Join the
fast-growing community of Watson Studio users today with a free account at Watson Studio

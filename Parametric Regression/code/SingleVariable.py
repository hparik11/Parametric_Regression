import matplotlib.pyplot as plt
import urllib2
import numpy as np
#from sklearn.metrics import mean_squared_error
#from sklearn.linear_model import LinearRegression
#from sklearn.cross_validation import train_test_split
#from sklearn import datasets, linear_model


## Scrapt the data from Website and extract meaningful Columns only

def extract_file(file):
    req = urllib2.Request(file)
    response = urllib2.urlopen(req)
    the_page = response.read()
    f = open('./Desktop/samples.txt','w')
    f.write(the_page)
    f.close()
    
    f = open('./Desktop/samples.txt','r+')
    x = []
    y = []
    m = {}
    d = f.readlines()
    f.seek(0)
    
    for i in d:
        if not i.startswith('#'):
            f.write(i)
    f.truncate()
    f.seek(0)
    d= f.readlines()
    a = []
    
    for i in d:
        a.append(i.split())
        
    for i in range(len(a)):
        m[float(a[i][0])] = float(a[i][1])
        x.append(float(a[i][0]))
        y.append(float(a[i][1]))
    
    f.close()
    
    return m                # Return a Dictionary


def Graph_Plot(x,y):
    plt.figure()
    plt.plot(x,y,'ro')
    plt.xlabel('X Values')
    plt.ylabel('Y Values')
    plt.show()

## Fit transform for Single Feature

def fit_single_feature_linear_model(data,k):
    
    
    x = data.keys()
    y = data.values()
    m = len(y)
    x_values = []
    y_values = []
    x_values.append(m)
    y_values.append(sum(y))
    
    for i in range(1,(2*k)+1):
        x_values.append(sum([d**i for d in x]))
            
    #print x_values    
    for i in range(1,k+1):
        y_values.append(sum([(x[j]**i)*y[j] for j in range(len(x))]))
            
    #print y_values
    
    A = np.matrix(np.zeros((k+1,k+1)))     
    for i in range(k+1):
        for j in range(k+1): 
            A[i,j]=x_values[i+j]
    
    #print A
    B = np.matrix(y_values) 
    
    theta = np.linalg.solve(A,B.T)
    return theta 
                 
    

def Make_Folds(data,k):

    subset_size = len(data.values())/fold_no
    
    testing_dataset = {}
    training_dataset = {}
    testing_dataset = data.items()[k*subset_size:][:subset_size]
    testing_dataset = dict((x,y) for x, y in testing_dataset)
    training_dataset = data.items()[:k*subset_size] + data.items()[(k+1)*subset_size:]
    training_dataset = dict((x,y) for x, y in training_dataset)
    
    return training_dataset, testing_dataset



def Predict_Y(theta,test,k):
    x_test = test.keys()
    y_pred = []
    
    for j in range(len(x_test)):
        y = 0
        for i in range(k+1):
            y = y + (theta[i,0]*(x_test[j]**i))
        y_pred.append(y)
    
    return y_pred
 
## Mean Square Error Method

def MSE(y_test, predict_y):
    
    return  (np.mean((np.array(y_test)-np.array(predict_y))**2))   


def Training_error(data):
    
    for i in range(0,fold_no):
        
        train,test = Make_Folds(data, i)
    
        y = train.values()
    
        Theta = fit_single_feature_linear_model(train, polynomial)
        #print Theta
    
        predict_y = Predict_Y(Theta,train,polynomial)
        
            
        mse_train.append(MSE(y, predict_y))
    
    Graph_Plot(predict_y, train.keys())
    return mse_train



def Testing_error(data):
    
    for i in range(0,fold_no):
        train,test = Make_Folds(data, i)
    
        y = test.values()
    
        Theta = fit_single_feature_linear_model(train, polynomial)
        
        predict_y = Predict_Y(Theta,test,polynomial)
        
        #print np.sqrt(np.mean((np.array(y_test)-np.array(predict_y))**2))
        #p.append((mean_squared_error(test.keys(), predict_y)))
                         
        mse_test.append(MSE(y, predict_y))
    
    Graph_Plot(predict_y, test.keys())
    
    return mse_test

  


if __name__ == '__main__':   
    
    file_num = (raw_input("Enter File Number: ")) 
    m1 = extract_file('http://www.cs.iit.edu/~agam/cs584/data/regression/svar-set' + file_num +'.dat')
    Graph_Plot(m1.keys(),m1.values())
    arr = np.array([])
    
    polynomial = int(raw_input("Enter the polynomial Value: "))
    fold_no = int(raw_input("Enter the no. of folds: "))
    mse_train = []
    mse_test = []
    
    train,test = Make_Folds(m1, 1)
    
    print "Training Error: %f" %np.mean(Training_error(m1))
    print "Testing Error: %f" %np.mean(Testing_error(m1))
    
    
    a = np.array(m1.keys())
    b = np.array(m1.values())
    
    p1 = np.append(arr,a)
    p2 = np.append(arr,b)
    
    z = np.polyfit(p1,p2,polynomial)
    print
    print "Coefficient values through polyfit function " +str(z)
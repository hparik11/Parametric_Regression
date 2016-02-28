import matplotlib.pyplot as plt
import urllib2
import numpy as np


## Scrapt the data from Website and extract meaningful Columns only

def extract_file(file):
    req = urllib2.Request(file)
    response = urllib2.urlopen(req)
    the_page = response.read()
    f = open('./Desktop/samples1.txt','w')
    f.write(the_page)
    f.close()
    
    f = open('./Desktop/samples1.txt','r+')
    
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
    
    p = len(a[0])
    
    for i in range(len(a)):
        m[float(a[i][p-1])] = [float(j) for j in a[i][:(p-1)]]
        
    
    f.close()
    
    return m                # Return a Dictionary

    
        
## Make partitions for training and testing data

def Make_Folds(Z, B, k):
    
    subset_size = Z.shape[0]/fold_no
    
    
    testing_x = Z[k*subset_size:][:subset_size]
    testing_y = B[k*subset_size:][:subset_size]
    
    training_x = np.r_[Z[:k*subset_size], Z[(k+1)*subset_size:]]
    training_y = np.r_[B[:k*subset_size], B[(k+1)*subset_size:]]
    
    return training_x, training_y, testing_x, testing_y    
        
## Function for fit transform            
              
def fit_multi_features_linear_model(data):
    
    
    x = data.values()
    
    y = data.keys()
    add_ones = np.ones((len(x),1),dtype='float')
    
    x_values = np.asarray(x)
    
    Z = np.matrix(np.c_[add_ones,x_values])
    B = np.matrix(y).T
    
    
    return Z, B
 

def Predict_Y(Theta,test):
    
    y_pred = []
    
    for j in range(test.shape[0]):
        y = 0
        for i in range(test.shape[1]):
            y = y + (Theta[i,0]*(test[j,i]))
        y_pred.append(y)
    
    return y_pred 
  
  
def Training_Error(data,fold_no):
    
    mse_train = []
    for i in range(0,fold_no):
        Z, B = fit_multi_features_linear_model(data)
        train_x, train_y, test_x, test_y = Make_Folds(Z, B, i)
        Theta = (np.linalg.inv(train_x.T * train_x)) * (train_x.T * train_y)
        #print Theta
        if i == 0:
            print
            print "###### Explicit Solution  #########"
            print Theta
            print 
            print
            print "###### Iterative Solution  #########"
            print gradient_descent(train_x,train_y)
            print
            print
            
            predict_y = Predict_Y(Theta,train_x)
            error = (train_x*Theta) - np.matrix([predict_y])
 
        else:
            
            predict_y = Predict_Y(Theta,train_x)
            error = (train_x*Theta) - np.matrix([predict_y])
                  
        mse_train.append(np.mean(np.array(error)**2))
        
    return np.mean(mse_train)
  
  
    
def Testing_Error(data,fold_no):
    
    mse_test = []
    for i in range(0,fold_no):
        Z, B = fit_multi_features_linear_model(data)
        train_x, train_y, test_x, test_y = Make_Folds(Z, B, i)
        Theta = (np.linalg.inv(train_x.T * train_x)) * (train_x.T * train_y)
        if i == 0:
            print
            print "###### Explicit Solution  #########"
            print Theta
            print 
            print
            print "###### Iterative Solution  #########"
            print gradient_descent(test_x,test_y)
            print
            print
            
            predict_y = Predict_Y(Theta,test_x)
            error = (test_x*Theta) - np.matrix(predict_y)
            
        else:
            
            predict_y = Predict_Y(Theta,test_x)
            error = (test_x*Theta) - np.matrix(predict_y)
                
        mse_test.append(np.mean(np.array(error)**2))
        
    return np.mean(mse_test)                                  

                                       
def gradient_descent(x,y):
    
    zT = x.T

    n = np.shape(x)[1]
    iterations= 10
    learning_rate = 0.001
    theta = np.ones((n,1),dtype='float')
    t = []
    
    for i in range(iterations):
        predicted = Predict_Y(theta,x)
        error = np.matrix(predicted).T - y
        
        grad = np.dot(zT, error) /x.shape[0]
        
        theta = theta - learning_rate*grad
        
    return theta                                                                                                       
             
                                                    
if __name__ == '__main__':    
   
    file_num = (raw_input("Enter File Number: ")) 
    m1 = extract_file('http://www.cs.iit.edu/~agam/cs584/data/regression/mvar-set' + file_num +'.dat') 
    
    fold_no = int(raw_input("Enter the number of Folds: "))
    print "Training Error : %f" %Training_Error(m1,fold_no)

    print "Testing Error : %f" %Testing_Error(m1,fold_no)
    
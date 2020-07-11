from  __future__  import division
from scipy.special import legendre
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib.pyplot import figure

choices_Qf = np.arange(1,101)
choices_N = np.arange(20,120,5)
choices_sigma =  np.arange(0, 2, 0.05)

array_Qf = np.split(choices_Qf, 50)
array_N = np.split(choices_N, 4)
array_sigma = np.split(choices_sigma,4)

for i in range(0,1):
    Qf = int(round(np.mean(array_Qf[2])))
    N = int(round(np.mean(array_N[3])))
    sigma = float(round(np.mean(array_sigma[2])))
#    N = 50
#    Qf = 2
#    sigma = 0.5
#    N = 50
#    Qf = 10
#    sigma = 0.5
    
    def get_Xn(N):
        X = np.random.uniform(-1, 1, size=N)
        np.asarray(X)
        X.sort()
        return X
    
    def get_Yn(X, Qf, N):
        epsilon_y = np.random.standard_normal(N)
        Fx = get_Fx(X, Qf)

        Y=[]
        for i in range(N):
            Y.append(Fx[i] + (sigma) * epsilon_y[i])
        return Y
    
    def get_error(N):
        epsilon = np.random.standard_normal(N)
        return epsilon
    
    def get_coeffs(Qf):
        a_Q = np.random.standard_normal(Qf)
        return a_Q
    
    def variance_factor(Qf):
        factor_sum = 0
        a_Q = get_coeffs(Qf)
        for q in range(0, Qf):
            cons_factor = np.array(a_Q**2/((2*q)+1)) 
            factor_sum += cons_factor + a_Q**2
        return factor_sum
    
    def normalize_coeffs(Qf):
        a_Q = get_coeffs(Qf)
        coeff_norm = []
        factor_sum = variance_factor(Qf)
        for q in range(0, Qf):        
            coeff_norm.append(a_Q[q] / np.sqrt(factor_sum))
       #ASIL KOD
        #coeff_norm = normalize([a_Q] , norm='l2') 
        return coeff_norm
        
    def get_Fx(X, Qf):
        a_Q_norm = normalize_coeffs(Qf)
        Fx_n = 0
        for q in range(Qf):
            Fx_n += get_legendre(Qf, X) * a_Q_norm[0][q]
        return Fx_n 
    
    def get_legendre(k, x):
        if k == 0: 
            return 1
        elif k == 1:
            return x
        else: 
            return((2*k-1)/k) * x * get_legendre(k-1, x) - ((k-1)/k) * get_legendre(k-2, x) #x is the non-type
            
    def main(N, Qf, sigma):
        X = get_Xn(N)
        Y = get_Yn(X, Qf, N)
        return X, Y    
    
    def activate(N, Qf):
        X, Y = main(N, Qf, sigma)
        data = {'x':X, 'y':Y} 
        Data = pd.DataFrame(data)      
        return Data
        
    def regression_and_all(Data, msk):
        Eout = []  
        X_init = Data.iloc[:,0].values
        Y_init = Data.iloc[:,1].values 
        train = Data[msk]
        test = Data[~msk]
        x_test = test.iloc[:,0].to_frame()
        x_train = train.iloc[:,0].to_frame()
        y_test = test.iloc[:,1].to_frame()
        y_train = train.iloc[:,1].to_frame()
    
        pipeline_input = [('Features', PolynomialFeatures(degree=Qf, include_bias=False)),
            ('Scaler' , StandardScaler()),
            ('Mode', LinearRegression())
                ]
        
        poly_regress = Pipeline(pipeline_input)
        model = poly_regress.fit(x_train.iloc[:, :1].values, y_train.iloc[:, :1].values)
        y_predict = model.predict(x_test.iloc[:, :1].values)
        print("\nScore for {}th degree = {}".format(Qf, r2_score(y_test,y_predict)))
       
        for i in range(len(y_predict)):
            Eout.append(np.mean(((y_predict-y_test)**2)/i))          
        Eout = np.asarray(Eout)      
        mean_error.append(Eout[1:])
        
        return mean_error, X_init, Y_init, x_test, y_predict
    
    '''Initate the Functions'''
    abs_error = []
    mean_error = [] 
    msk = np.random.rand(N) < 0.8
    
    for i in range(0,3):
        Data = activate(N, Qf)
        mean_error, X_init, Y_init, x_test, y_predict = regression_and_all(Data, msk)
    abs_error = np.mean(mean_error, 0)
    
    '''Check normalization'''
    print("\nFor E(f^2)=1  => ")
    Expected_value = 0  
    a_Q_norm=normalize_coeffs(Qf)
    for i in range(Qf):
        Expected_value += (X_init[i]**2 * a_Q_norm[0][i]) - (X_init[i] * a_Q_norm[0][i])**2
    print("\nExpected Value for Qf2=  " , Expected_value)   
        
    '''Plot All'''    
    #data points
    plt.figure(1)
    plt.title('Data Points for %sth degree, Number of samples='%Qf + str(N) + ', (sigma=' + str(sigma) +')')
    plt.scatter(X_init,Y_init, color = 'r')
    #plt.plot(Y_init , color = "green")
    plt.show()
    #Error
    plt.figure(2)
    plt.title('Eout for %sth degree, Number of samples='%Qf + str(N) + ', (sigma=' + str(sigma) +')')
    plt.plot(abs_error)
    #Prediction  
    plt.figure(3)
    plt.title('Best fit function for %sth degree, Number of samples='%Qf + str(N) + ', (sigma=' + str(sigma) +')')
    plt.scatter(X_init, Y_init, color = 'b')
    plt.plot(x_test,y_predict, linewidth=3, color = 'r')




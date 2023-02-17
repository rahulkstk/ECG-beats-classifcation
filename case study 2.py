


from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import sys
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


sns.set(context='notebook', style='white', rc={'figure.figsize':(14,10)})

file_path_save = r'C:/Users/rahul/Desktop/Semester1/MLAIDS/Labs/CaseStudy/CaseStudy3/Dataset/'
#file_path_save = r'C:/Users/dtavi/Documents/UofG/ML and AI Lab/Case study 3 Explainable Machine Learning/'

print("Loading training beats...")
train_beats = np.genfromtxt(file_path_save+'train_beats.csv', delimiter=',')
print("Loading testing beats...")
test_beats = np.genfromtxt(file_path_save+'test_beats.csv', delimiter=',')  


print("Loading training patients...")
train_patients = np.genfromtxt(file_path_save+'train_patients.csv', delimiter=',')  
print("Loading testing patients...")
test_patients = np.genfromtxt(file_path_save+'test_patients.csv', delimiter=',') 

print("Loading All data...")
all_data = np.genfromtxt(file_path_save+'all_data.csv', delimiter=',') 



def plot_counts(df,name,color):
  counts = df[275].value_counts()
  plt.figure(figsize=(8,4))
  feature_names = ['N','L','R','V','A','F','f','/']
  barplt = plt.bar(counts.index, counts.values, alpha=0.8, color=color)
  plt.title(name)
  plt.ylabel('Number of Occurrences', fontsize=12)
  plt.xlabel('Beat Category', fontsize=12)
  plt.xticks(ticks=[1,2,3,4,5,6,7,8],labels=feature_names)
  for bar in barplt:
    yval = bar.get_height()
    plt.text(bar.get_x()+.2, yval+600, yval)
  plt.show()
  
plot_counts(pd.DataFrame(all_data),'All Data','g')

plot_counts(pd.DataFrame(train_beats),'Train Beats','r')
plot_counts(pd.DataFrame(test_beats),'Test Beats','b')

plot_counts(pd.DataFrame(train_patients),'Train Patients','r')
plot_counts(pd.DataFrame(test_patients),'Test Patients','b')



#Train Model


def train_eval(df_train,df_evaluate):      
 
    # x_train = df_train.iloc[0:3000,0:274]
    # y_train = df_train.iloc[0:3000,275]
    # x_eval  = df_evaluate.iloc[0:3000,0:274]
    # y_eval  = df_evaluate.iloc[0:3000,275]  
    
    x_train = df_train.iloc[0:10000,0:274]
    y_train = df_train.iloc[0:10000,275]
    x_eval  = df_evaluate.iloc[0:10000,0:274]
    y_eval  = df_evaluate.iloc[0:10000,275]   
    
    #select classifier
    # clf = SVC() #---------------------------------------------Classifier
    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf = LogisticRegression(random_state=0)

    for c in range(3):
        if c == 0:
            clf = SVC()
            print('******************Classifier: Support Vector Classifier**************')
            classifier = 'SVC'
        elif c == 1:
            clf = KNeighborsClassifier(n_neighbors=5)
            print('******************Classifier: K Neighbors Classifier******************')
            classifier = 'KNN'
        else:
            clf = LogisticRegression(random_state=0)
            print('******************Classifier: Logistic Regression******************')
            classifier = "Logistic Regression"
            
        #KFold
        kf = KFold(n_splits=5)
        iteration = 1
        model_score_i = np.array([])
        importance_i = np.empty((0,11), int)
        
        print('evaluation of classifer with 5-fold cross validation with PFI')
        for train_index, test_index in kf.split(x_train):
            # result = next(kf.split(x_train), None)
            # result = [train_index, test_index]
            
            X_train = x_train.iloc[train_index]
            X_test  = x_train.iloc[test_index] 
            Y_train = y_train.iloc[train_index]
            Y_test  = y_train.iloc[test_index]
                
    
            model = clf.fit(X_train, Y_train)
        
            
            y_pred = model.predict(X_test)
            print('-----------ITERATION ',iteration,'----------')
            
            model_score = model.score(X_test, Y_test) #reference score 's'
            model_score_i = np.append(model_score_i, model_score)
            print('model score', model_score)
            
            #accuracy, precision, recall, f1score and a confusion matrix
            print('-------------------------------')
            print('accuracy score: \n',accuracy_score(Y_test, y_pred)) #y_eval = y_true
            print('precision_score: \n',precision_score(Y_test, y_pred, average='micro')) #for entire dataset
            print('recall_score: \n',recall_score(Y_test, y_pred, average='micro')) 
            print('f1_score: \n',f1_score(Y_test, y_pred, average='micro'))
            print('confusion_matrix: \n',confusion_matrix(Y_test, y_pred))

                        
            # #--------------------------------------------------------------------------
    
            #permutation  
            r = permutation_importance(model, X_test, Y_test,
                                    n_repeats=30,  #!!!!!CHANGE NUMBER OF REPEATS FOR PERMUTATION HERE
                                    random_state=0)
            
            
            importance = np.mean(r.importances_mean[0:24])
            for i in range(10):
                start = (i+1)*25
                end = start+24
                importance_append = np.mean(r.importances_mean[start:end])
                importance = np.append(importance, importance_append)
            
            t = [ i+1 for i in range(len(importance))]
            # plt.bar(t,importance)
        
            # plt.title(f'Importances Mean, Iteration {iteration}')
            # plt.show()
            
            
            # t = [ i+1 for i in range(len(r.importances_std))]
            # plt.bar(t,r.importances_std)
            # plt.title('Importances Std, Iteration '+i)
            # plt.show()
            
            importance_i = np.vstack([importance_i, importance])
            # print(importance)
            # print(iteration)
            # print(result)
            iteration = iteration+1
        
        print('-------------------------------')
        average_model_score = np.average(model_score)
        print('average model score: ', average_model_score)
        average_importance = np.average(importance_i, axis=0)
        plt.bar(t,average_importance)
        plt.title(f'Importances Mean, Averaged, {classifier}')
        plt.xlabel("Segments")
        plt.ylabel("Averaged Importances")
        plt.show()
        
        print(f'------------model with {classifier}-------------')
        
        model = clf.fit(x_train, y_train)
        y_pred = model.predict(x_eval)
        
        print('accuracy score: \n',accuracy_score(y_eval, y_pred)) #y_eval = y_true
        print('precision_score: \n',precision_score(y_eval, y_pred, average='micro')) #for entire dataset
        print('recall_score: \n',recall_score(y_eval, y_pred, average='micro')) 
        print('f1_score: \n',f1_score(y_eval, y_pred, average='micro'))
        cf_matrix=confusion_matrix(y_eval, y_pred)
        print('confusion_matrix: \n',confusion_matrix(y_eval, y_pred))
        sns.heatmap((cf_matrix)/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
        
    return

if __name__ == "__main__":    
    
    sys.stdout = open("log.txt", "w")
#------------------------holdout splitting method------------------------------ 
    print('%============ holdout splitting method ============%')   
    beat_split = train_eval(pd.DataFrame(train_beats), pd.DataFrame(test_beats))
  
 
    print('%============ patients-hold out ============%')   
#---------------------------patients hold out----------------------------------
    patients_split = train_eval(pd.DataFrame(train_patients), pd.DataFrame(test_patients))
    
    sys.stdout.close()
 


    
    
    
    
    
    
    
    

# Model 형성에 기여 하는 모듈들
import pandas as pd
import numpy as np
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import _check_sample_weight
from sklearn.model_selection import train_test_split
# import 해야하는 모듈들
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import collections
import warnings
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pyperclip
warnings.filterwarnings('ignore')

# first column 만드는 것
from functools import reduce
import operator

# confusion matrix 담을 것 + excel 출력 대상임.
conma=pd.DataFrame(np.array([[0,0],[0,0]]),columns=['예측0','예측1'],index=['실제0','실제1'])


# roc 곡선 그리기 위한 함수
def roc_curve_plot(y_test , pred_proba_c1):
    # 임곗값에 따른 FPR, TPR 값을 반환 받음. 
    fprs , tprs , thresholds = roc_curve(y_test ,pred_proba_c1)

    # ROC Curve를 plot 곡선으로 그림. 
    plt.plot(fprs , tprs, label='ROC')
    # 가운데 대각선 직선을 그림. 
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    
    # FPR X 축의 Scale을 0.1 단위로 변경, X,Y 축명 설정등   
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    plt.xlim(0,1); plt.ylim(0,1)
    plt.xlabel('FPR( 1 - Sensitivity )'); plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.show()




    
    
    
    

#AUC + confusion mat
# 교수님 주신 논문에서 만들었으나 아직 사용은 X 나중에 필요시 사용
from sklearn.metrics import confusion_matrix

def new_auc(y_test,pred):
    conma=confusion_matrix(y_test , pred) # y_test부분에 실제값, 2번째 파라미터로 인자값 
    TPR = conma[1,1]/(conma[1,0]+conma[1,1])
    FPR = conma[0,1]/(conma[0,0]+conma[0,1])
    return (1+TPR-FPR)/2










# Default Fit - 원래 adaboost-SAMME.R 방식으로 fit 하는 것
# 이거는 fit2없는 version
def defaultfit(clf, xlsx, n_fold):
    print("================================================Default Fit 데이터크기",len(xlsx),"================================================")
    skfold = StratifiedKFold(n_splits=n_fold)
    n_iter=0
    cv_acc=[]
    cv_auc=[]
    global conmaA
    conmaA = pd.DataFrame()
    colname_first = [["fold_{}".format(i),"fold_{}".format(i)] for i in range( n_fold*3 )] # 여기 15는 n_estimators 갯수
    colname_first = list( reduce( operator.add, colname_first ) )
    for i in range(3):
        print("================================================skfold",i+1,"번째================================================")
        xlsx=xlsx.sample(frac=1).reset_index(drop=True)
        X = xlsx.iloc[:,1:8]
        y = xlsx.iloc[:,9]

        for train_index, test_index  in skfold.split(X, y):
            # split( )으로 반환된 인덱스를 이용하여 학습용, 검증용 테스트 데이터 추출
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            #학습 및 예측 
            clf.fit(X_train , y_train,epochs=1000)    
            pred = clf.predict(X_test)

            # 반복 시 마다 정확도 측정 
            n_iter += 1
            accuracy = np.round(accuracy_score(y_test,pred), 4)
            auc=roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
            train_size = X_train.shape[0]
            test_size = X_test.shape[0]

            print("\n#{0} 교차 검증 정확도 :{1}, 학습 데이터 크기: {2}, 검증 데이터 크기: {3}"
                  .format(n_iter, accuracy, train_size, test_size))
            print('\n{0} 검증 세트 인덱스:{1}'.format(n_iter,xlsx.iloc[test_index.tolist(),0]))
            cv_acc.append(accuracy)
            cv_auc.append(auc)
            mat=confusion_matrix(y_test,pred)
            roc_curve_plot(y_test, clf.predict_proba(X_test)[:, 1] )
            conma.iloc[0,0],conma.iloc[0,1],conma.iloc[1,0],conma.iloc[1,1]=mat[0,0],mat[0,1],mat[1,0],mat[1,1]
            print("\n")
            print(n_iter,conma,"\n","acc:",accuracy,"auc:",auc)
            conmaA=pd.concat([conmaA,conma],axis=1)
            
            
    colname_second = conmaA.columns        
    accauc = np.array(cv_acc + cv_auc)
    conmaA=np.vstack((np.array(conmaA), accauc))
    conmaA=pd.DataFrame(np.array(conmaA),
                index = ['실제0','실제1','accauc'],    
                columns = [colname_first, colname_second])
    conmaA.to_excel('Result/DefalutFit'+str(len(xlsx)) + '.xlsx')
    print("================================================종료================================================")
    print('\n## 교차 검증별 acc:', np.round(cv_acc, 4))
    print('\n## 교차 검증별 auc:', np.round(cv_auc, 4))
    print('\n## 평균 검증 정확도:', np.mean(cv_acc))
    print('\n## 평균 검증 auc:', np.mean(cv_auc))
    print(conmaA)


    
    
    
# AUC_fit 하는 것
def aucweightfit(clf, xlsx, n_fold):
    print("================================================AUC Fit 데이터크기",len(xlsx),"================================================")
    skfold = StratifiedKFold(n_splits=n_fold)
    n_iter=0
    cv_acc=[]
    cv_auc=[]
    global conmaB
    conmaB=pd.DataFrame()
    colname_first = [["fold_{}".format(i),"fold_{}".format(i)] for i in range( n_fold*3 )] # 여기 15는 n_estimators 갯수
    colname_first = list( reduce( operator.add, colname_first ) )


    
    for i in range(3):
        print("================================================",i,"번째================================================")
        xlsx.sample(frac=1).reset_index(drop=True)
        X = xlsx.iloc[:,1:8]
        y = xlsx.iloc[:,9]

        for train_index, test_index  in skfold.split(X, y):
            # split( )으로 반환된 인덱스를 이용하여 학습용, 검증용 테스트 데이터 추출
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            #학습 및 예측 
            clf.fit(X_train , y_train,epochs=1000)
            clf.fit2(X_train,y_train)
            pred = clf.predict(X_test)

            # 반복 시 마다 정확도 측정 
            n_iter += 1
            accuracy = np.round(accuracy_score(y_test,pred), 4)
            auc=roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
            train_size = X_train.shape[0]
            test_size = X_test.shape[0]

            print('\n#{0} 교차 검증 정확도 :{1}, 학습 데이터 크기: {2}, 검증 데이터 크기: {3}'
                  .format(n_iter, accuracy, train_size, test_size))
            print('#{0} 검증 세트 인덱스:{1}'.format(n_iter,xlsx.iloc[test_index.tolist(),0]))
            cv_acc.append(accuracy)
            cv_auc.append(auc)
            roc_curve_plot(y_test, clf.predict_proba(X_test)[:, 1] )
            mat=confusion_matrix(y_test,pred)
            conma.iloc[0,0],conma.iloc[0,1],conma.iloc[1,0],conma.iloc[1,1]=mat[0,0],mat[0,1],mat[1,0],mat[1,1]
            print("\n")
            print(n_iter,conma,"\n","acc:",accuracy,"auc:",auc)
            conmaB=pd.concat([conmaB,conma],axis=1)
    
    colname_second = conmaB.columns   
    accauc = np.array(cv_acc + cv_auc)
    conmaB = np.vstack((np.array(conmaB), accauc))    
    conmaB = pd.DataFrame(np.array(conmaB),
                index = ['실제0','실제1','accauc'],    
                columns = [colname_first, colname_second])       
    conmaB.to_excel('Result/AUCfit'+str(len(xlsx))+'.xlsx')
    print("================================================종료================================================")
    print('\n## 교차 검증별 acc:', np.round(cv_acc, 4))
    print('\n## 교차 검증별 auc:', np.round(cv_auc, 4))
    print('## 평균 검증 정확도:', np.mean(cv_acc))
    print('## 평균 검증 auc:', np.mean(cv_auc))
    print(conmaB)

def readexcel(roadname, datanumlist):
    Datalist = []
    for i,c in enumerate(datanumlist):
        Datalist.append(pd.read_excel(roadname+str(c)+'.xlsx'))
    return Datalist

def readcsv(roadname ,datanumlist):
    Datalist = []
    for i,c in enumerate(datanumlist):
        Datalist.append(pd.read_csv(roadname+str(c)+'.csv'))
    return Datalist

def valaccauc(datas): 
    for data in datas :
        reslen = int((data.shape[1]) / 2)
        accs = np.array(data.iloc[4,1:reslen+1])
        aucs = np.array(data.iloc[4,reslen+1:data.shape[1]])
        print( '-' * 30 )
        print(' acc_mean : %.2f \n acc_std : %.2f \n auc_mean : %.2f \n auc_std : %.2f ' 
              % (np.mean(accs),np.std(accs), np.mean(aucs), np.std(aucs)) )
        print( '-' * 30 )    





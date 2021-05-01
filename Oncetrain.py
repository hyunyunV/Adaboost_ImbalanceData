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
import math
from functools import reduce
import operator
warnings.filterwarnings('ignore')

# 전역변수 데이터프레임 설정
global conmaA
conmaA=pd.DataFrame()
global conmaB
conmaB=pd.DataFrame()
global conmaC
conmaC=pd.DataFrame()

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


def new_auc(y_test,pred):
    conma=confusion_matrix(y_test , pred) # y_test부분에 실제값, 2번째 파라미터로 인자값 
    TPR = conma[1,1]/(conma[1,0]+conma[1,1])
    FPR = conma[0,1]/(conma[0,0]+conma[0,1])
    return (1+TPR-FPR)/2

def Maketable(clf,X,y): # X = X_train, y = y_train
    table = pd.DataFrame(np.array(y)) # 참고로 그냥 y하면 데이터프레임이라서 기업 인덱스가 존재함
    for i, estimator in enumerate(clf.estimators_):
        ap = pd.DataFrame(estimator.predict(X), columns = ['0_'+str(i+1),'1_'+str(i+1)] ) # 이게 아마 넘파이 일거고
        table = pd.concat([table,ap],axis=1)

    return table 

def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

def makeexcel(clf,X_train,X_test,y_train,y_test):
    for i in range(len(clf.estimators_)):
        X = pd.DataFrame()
        y = pd.DataFrame()
        proba1 = clf.estimators_[i].predict(X_train)
        proba2 = clf.estimators_[i].predict(X_test)
        proba = pd.DataFrame(np.vstack((proba1,proba2)))
        X = pd.concat([X_train,X_test])
        y = pd.concat([X_train,X_test])
        indexname = ['train']*(len(X)-500)+['test']*500
        result = pd.concat([X,y,proba],axis=1)
        result.index = indexname
        result.to_excel('estimators/first_fold_estimators'+str(i+1)+'.xlsx')        
        
        
def geoacc(mat):
    value = (mat.iloc[0,0] / sum(mat.iloc[0,:])) * (mat.iloc[1,1] / sum(mat.iloc[1,:]))
    print(value)
    return math.sqrt(value)

def makeconma(conma,mat):
    conma.iloc[0,0],conma.iloc[0,1],conma.iloc[1,0],conma.iloc[1,1]=mat[0,0],mat[0,1],mat[1,0],mat[1,1]
    return conma

def acccondition(clf):
    clf.accfit = True
    clf.newfit = False
    clf.weight = False  

    
def makecolname(n_fold,n_for):
    colname_first = [["fold_{}".format(i),"fold_{}".format(i)] for i in range( n_fold*n_for )] # 여기 15는 n_estimators 갯수
    colname_first = list( reduce( operator.add, colname_first ) )
    return colname_first
    
    
    
def ToExcel(cv_acc,cv_auc,cv_geoacc,mat,colname_first,colname_second):
    accauc = np.array(cv_acc + cv_auc)
    geoaccs = np.array(cv_geoacc + cv_geoacc)
    
    mat = np.vstack((np.array(mat), accauc, geoaccs))
    mat = pd.DataFrame(np.array(mat),
                index = ['실제0','실제1','accauc','geoacc'],    
                columns = [colname_first, colname_second])
    return mat

    
    
# AUC_fit 하는 것
def oncefit(clf, xlsx, n_fold,epochs=1000):
    print("================================================ Fit 데이터크기",len(xlsx),"================================================")
    skfold = StratifiedKFold(n_splits=n_fold)
    n_iter=0
    n_for = 3
    
    # 각 훈련에서 acc auc geoacc 담을 곳 
    global conmaA
    conmaA=pd.DataFrame()
    global conmaB
    conmaB=pd.DataFrame()
    global conmaC
    conmaC=pd.DataFrame()
    cv_acc_def=[]
    cv_auc_def=[]
    cv_acc_auc=[]
    cv_auc_auc=[]
    cv_acc_acc=[]
    cv_auc_acc=[]
    cv_geoacc_def = []
    cv_geoacc_auc = []
    cv_geoacc_acc = []
        

    # 데이터 프레임 컬럼만드는 것 
    colname_first = makecolname(n_fold,n_for)
    for i in range(n_for):
        print("================================================",i,"번째================================================")
        xlsx.sample(frac=1,random_state=i).reset_index(drop=True)  
        X = xlsx.iloc[:,4:11]
        y = xlsx.iloc[:,11]

        for train_index, test_index  in skfold.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            train_size = X_train.shape[0] # 이거 나중에 없애도 댐
            test_size = X_test.shape[0] # 이것도 나중에 없애도 댐 아래에 print 부분이랑 같이 날림 
            n_iter += 1
            
            
            #학습 및 예측 
            print(epochs)
            clf.fit(X_train , y_train,epochs = epochs)    
            if n_iter==1:
                makeexcel(clf,X_train,X_test,y_train,y_test)
            # 여기뭐 임계점 들어가야함 
            
            
            # Default Fit 부분
            # 반복 시 마다 정확도 측정 
            pred = clf.predict(X_test,0.5)
            mat=confusion_matrix(y_test,pred)
            conma_temp = makeconma(conma,mat) # 이걸로 위에줄삭제
            
            # 평가지표 생성
            accuracy = np.round(accuracy_score(y_test,pred), 4)
            auc=roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
            geoaccvalue = geoacc(conma_temp)
            
            # 평가지표담기
            cv_acc_def.append(accuracy)
            cv_auc_def.append(auc)
            cv_geoacc_def.append(geoaccvalue)
            conmaA=pd.concat([conmaA,conma_temp],axis=1)
            
            
            print("\n#{0} 교차 검증 정확도 :{1}, 학습 데이터 크기: {2}, 검증 데이터 크기: {3}"
                  .format(n_iter, accuracy, train_size, test_size))
            print('\n{0} 검증 세트 인덱스:{1}\n'.format(n_iter,xlsx.iloc[test_index.tolist(),0]))
            
            print(n_iter,conma_temp,"\n","acc:",accuracy,"auc:",auc) # 이것도 나중에 지우기
            ## 이 지점에서 Default fit 은 훈련이 끝
            

            # 여기서 부터 Adaboost 훈련
            clf.fit2(X_train, y_train)
            proba1 = clf.decision_function(X_train)
            Threshold = Find_Optimal_Cutoff(y_train, proba1)
            print("Threshold point: ",Threshold)
            pred = clf.predict(X_test,Threshold)
            mat=confusion_matrix(y_test,pred)
            conma_temp = makeconma(conma,mat)
            
            accuracy = np.round(accuracy_score(y_test,pred), 4)
            auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
            geoaccvalue = geoacc(conma_temp)

            cv_acc_auc.append(accuracy)
            cv_auc_auc.append(auc)
            cv_geoacc_auc.append(geoaccvalue)
            conmaB=pd.concat([conmaB,conma_temp],axis=1)
            
            print("\n")
            print(n_iter,conma_temp,"\n","acc:",accuracy,"auc:",auc)
            # 이 지점에서 Adaboost 훈련이 끝
            
            
            # 이 지점에서 ACC Fit 훈련이 시작 
            clf.table = Maketable(clf,X_train, y_train) # 이거는 위에 만들어야함 이거 만들어서 경민이 주면 대겠지 ??
            acccondition(clf)
            #clf.accweight = ACCfit(clf.table)  return  ACC_Weight_Vector
            #proba1 = clf.decision_function(X_train)
            #Threshold = Find_Optimal_Cutoff(y_train, proba1)
            #pred = clf.predict(X_test,Threshold)

            #mat=confusion_matrix(y_test,pred)
            #conma_temp = makeconma(conma,mat)

            # 이렇게하면 pred 까지 뽑아 낸 것임
            
            #accuracy = np.round(accuracy_score(y_test,pred), 4)
            #auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
            #geoaccvalue = geoacc(conma_temp)
            
            #cv_acc_acc.append(accuracy)
            #cv_auc_acc.append(auc)
            #cv_geoacc_auc.append(geoaccvalue)

  #          print("\n")
   #         print(n_iter,conma_temp,"\n","acc:",accuracy,"auc:",auc)
    #        conmaC=pd.concat([conmaC,conma],axis=1)
            # 이 지점에서 ACC pred이 끝
            
    
    colname_second = conmaA.columns        
    
    # 이 아래 부분은 따로 만들지 말고 다 합쳐야 겠는데?  생각해보니깐 그러면 안댐 걍 따로 만들어야함 
    # Default fit 부분 
    resultdef = ToExcel(cv_acc_def,cv_auc_def,cv_geoacc_def,conmaA,colname_first,colname_second) 
    resultdef.to_excel('Result/defalutfit'+str(len(xlsx)) + '.xlsx')

    
    # AUC fit 부분 
    resultdef = ToExcel(cv_acc_auc,cv_auc_auc,cv_geoacc_auc,conmaB,colname_first,colname_second) 
    resultdef.to_excel('Result/AUCfit'+str(len(xlsx)) + '.xlsx')

    # ACC fit 부분 # 이거는 경민이꺼 받아서 써야함
    #resultdef = ToExcel(cv_acc_acc,cv_auc_acc,cv_geoacc_acc,conmaC,colname_first,colname_second) 
    #resultdef.to_excel('Result/ACCfit'+str(len(xlsx)) + '.xlsx')

    print("================================================종료================================================")

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





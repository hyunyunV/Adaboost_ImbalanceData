import pandas as pd
from scipy import stats


import pandas as pd
from scipy import stats

# 간단한 클린징 작업 
def setindex(dfs):
    for df in dfs:
        df.set_index("Unnamed: 0",inplace = True)
    return dfs

class DataListforStat:
    def __init__(self,df):
        df.set_index("Unnamed: 0",inplace = True)
        self.df = df
        self.Datalen = int(self.df.shape[1]/2)
        self.acc = []
        self.auc = []
        self.geoacc = []
    
    
    def DivList(self):
        self.acc = self.df.loc['accauc'][:self.Datalen]
        self.auc = self.df.loc['accauc'][self.Datalen:]
        self.geoacc = self.df.loc['geoacc'][self.Datalen:]
    
def Mann(x,y, Number = 0 , leftstr = "", rightstr= "", Measure = ""):
    print(str(Number)+"\t"+leftstr + " & " + rightstr + "\tMeasure : " + Measure + "\n")
    statistic , pvalue = stats.mannwhitneyu(x,y)
    print("\nstatistic : %d , pvalue : %.7f\n" % (statistic, pvalue))
    return (statistic, pvalue)


def BeClass(dfs):
    ClassList = []
    for df in dfs:
        df = DataListforStat(df)
        ClassList.append(df)
    return ClassList

def DoDivList(dfs):
    for df in dfs:
        df.DivList()
    return dfs


def PrintMannStat(leftlist, rightlist, Numbers, leftstr,rightstr):
    FrameList = []
    i = 0
    for l, r in zip(leftlist, rightlist):
        Number = Numbers[i]
        x_1,y_1 = Mann(l.acc,r.acc, Number ,leftstr, rightstr,"ACC")
        x_2,y_2 = Mann(l.auc,r.auc, Number ,leftstr, rightstr,"AUC")
        x_3,y_3 = Mann(l.geoacc,r.geoacc, Number ,leftstr, rightstr,"GEOACC")
        now = pd.DataFrame([x_1,y_1,x_2,y_2,x_3,y_3], \
                            index = [["ACC"]*2+["AUC"]*2+["GEOACC"]*2,["statistic","p-value"]*3] , \
                            columns = [str(Number)])
        FrameList.append(now)
        i += 1
    return pd.concat(FrameList,axis=1)


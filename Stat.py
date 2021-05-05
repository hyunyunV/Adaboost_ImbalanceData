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

def PrintMannStat(leftlist, rightlist, Number, leftstr,rightstr):
    for l, r in zip(leftlist, rightlist):
        Mann(l.acc,r.acc, Number ,leftstr, rightstr,"ACC")
        Mann(l.auc,r.auc, Number ,leftstr, rightstr,"AUC")
        Mann(l.geoacc,r.geoacc, Number ,leftstr, rightstr,"GEOACC")










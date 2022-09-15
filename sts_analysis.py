import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def load_maestro_data(file):
    """
    """
    data = pd.read_csv(file)
    data = data[data['Stars'].isna()]
    data = data[data['Job Name'].notna()]
    data = data.reset_index(drop=True)
    return data

def get_stsClassification(data, RealClass, PredClass, RealThrs, PredThrs):
    """
    Given a real classifier, a predicted classifier and a threshold for each, 
    return the TP,TN,FP,FN values.
    - data:
    - RealClass:
    - PredClass:
    - RealThrs:
    - PredThrs:
    """
    dicClass = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for ind in data.index:
        realvalue = data[RealClass][ind]
        predvalue = data[PredClass][ind]
        if realvalue < RealThrs:
            if predvalue < PredThrs:
                dicClass['TP']+=1
            else:
                dicClass['FN']+=1
        else:
            if predvalue < PredThrs:
                dicClass['FP']+=1
            else:
                dicClass['TN']+=1
    return dicClass

def _get_equal_population(data,RealClass,RealThrs):
    """
    """
    positives, negatives = [],[]
    for ind in data.index:
        realvalue = data[RealClass][ind]
        if realvalue < RealThrs:
            positives.append(ind)
        else:
            negatives.append(ind)
    positives = np.asarray(positives)
    negatives = np.asarray(negatives)
    if len(positives)>len(negatives):
        sample = np.random.choice(positives,size=len(negatives))
        sample = np.concatenate((sample,negatives))
        neg_invariance = True 
    elif len(positives)<len(negatives):
        sample = np.random.choice(negatives,size=len(positives))
        sample = np.concatenate((sample,positives))
        neg_invariance = False
    else:
        raise ValueError('Both popoulations have the same number of elements')
    return sample,neg_invariance

def get_ROC(data,RealClass,PredClass,RealThrs,thresholds=None,whole=True,n_iter=100):
    """
    Plot the ROC curve for a given real and predicted classifier
    - data:
    - RealClass:
    - RealThrs:
    - PredClass:
    - thresholds: Optional list of thresholds to use instead of taking all
    possible thresholds
    - whole_db: if False, take n (given by n_iter argument) equally
    populated samples for true and false elements according the RealClass
    """
    if whole:
        #If not specific list of threshold has been pass use all data predvalues as threshols
        if thresholds == None:
            thresholds = [data[PredClass][ind] for ind in data.index]
            thresholds = list(set(thresholds))
        fprs, tprs = [], []
        for threshold in thresholds:
            dicClass = get_stsClassification(data, RealClass, PredClass, RealThrs, threshold)
            tpr = dicClass['TP']/(dicClass['TP']+dicClass['FN'])
            fpr = dicClass['FP']/(dicClass['FP']+dicClass['TN'])
            tprs.append(tpr)
            fprs.append(fpr)
        fprs = np.r_[0, fprs, 1]
        tprs = np.r_[0, tprs, 1]
        return tprs, fprs, thresholds, None
    else:
        fprs, tprs = [], []
        samples = []
        for i in range(n_iter):
            samp ,neg_invariance = _get_equal_population(data,RealClass,RealThrs)
            samples.append(samp)
        samples = np.asarray(samples)
        if thresholds == None:
            thresholds = []
            for sample in samples:
                aux = [data[PredClass][ind] for ind in sample]
                thresholds.extend(aux)
            thresholds = list(set(thresholds))
        for i,sample in enumerate(samples):
            print(i+1)
            fprs_, tprs_ = [], []
            balanceddata = data.iloc[list(sample)]
            balanceddata = balanceddata.reset_index(drop=True)
            for threshold in thresholds:
                dicClass = get_stsClassification(balanceddata, RealClass, PredClass, RealThrs, threshold)
                tpr = dicClass['TP']/(dicClass['TP']+dicClass['FN'])
                fpr = dicClass['FP']/(dicClass['FP']+dicClass['TN'])
                tprs_.append(tpr)
                fprs_.append(fpr)
            fprs.append(fprs_)
            tprs.append(tprs_)
        fprs = np.asarray(fprs)
        tprs = np.asarray(tprs)
        fprs = (fprs.mean(axis=0), fprs.std(axis=0))
        tprs = (tprs.mean(axis=0), tprs.std(axis=0))
        return tprs, fprs, thresholds, neg_invariance

def plot_ROC(data,RealClass,PredClass,RealThrs,thresholds=None,label=None,whole=True,n_iter=100):
    tprs, fprs, thresholds, neg_invariance = get_ROC(data,RealClass,PredClass,RealThrs,thresholds,whole,n_iter)
    if not whole:
        if neg_invariance:
            fprs = list(fprs[0])
            tprs, errs = tprs
        else:
            fprs, errs = fprs
            tprs = tprs[0]
        fprs, tprs, thresholds, errs = zip(*sorted(zip(fprs, tprs, thresholds,errs)))
        errs = np.asarray(errs)
    else:
        fprs, tprs, thresholds = zip(*sorted(zip(fprs, tprs, thresholds)))
    fprs = np.asarray(fprs)
    tprs = np.asarray(tprs)
    thresholds = np.asarray(thresholds)
    auc = np.trapz(tprs, x=fprs)
    
    if label==None: label='Logistic'
    plt.plot(fprs, tprs, marker='.', markersize=4,label=label+' AUC=%.3f'%auc)
   
    if not whole:
        if neg_invariance:
            plt.fill_between(fprs, tprs-errs, tprs+errs, alpha=.5)
        else:
            plt.fill_betweenx(tprs, fprs-errs, fprs+errs, alpha=.5)

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis('scaled')
    axes = plt.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])
    
    specificity = 1-fprs
    recalls = tprs
    gmeans = np.square(np.multiply(specificity,recalls))
    ix1 = np.argmax(gmeans)
    
    sthresholds = -np.sort(-thresholds)
    optaccuracy, thr_accuracy= 0,0
    for threshold in sthresholds:
        dicClass = get_stsClassification(data, RealClass, PredClass, RealThrs, threshold)
        accuracy = (dicClass['TP']+dicClass['TN'])/(dicClass['TP']+dicClass['TN']+dicClass['FP']+dicClass['FN'])
        if accuracy > optaccuracy:
            optaccuracy = accuracy
            thr_accuracy = threshold

    optprecision = 0
    for threshold in sthresholds:
        dicClass = get_stsClassification(data, RealClass, PredClass, RealThrs, threshold)
        if dicClass['TP']+dicClass['FP'] == 0:
            continue
        precision = dicClass['TP']/(dicClass['TP']+dicClass['FP'])
        if dicClass['FP'] == 0 or precision == 1:
            optprecision = precision
            thr_precision = threshold
            break
        else:
            if precision > optprecision:
                optprecision = precision
                thr_precision = threshold
    
    print('- %s \t #Observations=%d \t AUC= %.3f \t Max_GMean=%.3f \t Thr_GMean=%.3f \t OptPrecision=%.3f \t Thr_Precision=%.3f \t OptAccuracy=%.3f \t Thr_Accuracy=%.3f' % (label,data.shape[0],auc, gmeans[ix1],thresholds[ix1],optprecision,thr_precision,optaccuracy,thr_accuracy))
    
    plt.scatter(fprs[ix1], tprs[ix1], marker='o', color='black',s=30)
   
    plt.legend()

def plot_multBinaryClass(data, RealClass, PredClass, RealThrs, PredThrs,disc=None):
    print(data)
    if disc != None:
        labels = data[disc].unique()
        TPs = []
        FPs = []
        FNs = []
        TNs = []
        for label in labels:
            filtdata = data[(data[disc] == label)]
            filtdata = filtdata.reset_index(drop=True)
            dicClass = get_stsClassification(filtdata, RealClass, PredClass, RealThrs, PredThrs)
            TPs.append(dicClass['TP'])
            FPs.append(dicClass['FP'])
            FNs.append(dicClass['FN'])
            TNs.append(dicClass['TN'])
        
        TPs = np.asarray(TPs)
        FPs = np.asarray(FPs)
        FNs = np.asarray(FNs)
        TNs = np.asarray(TNs)
        accuracies = (TPs+TNs)/(TPs+FPs+FNs+TNs)
        accuracies = np.around(accuracies,decimals=3)
        
        fig, ax = plt.subplots()
        ax.bar(labels, TPs, label='TP',color='steelblue')
        ax.bar(labels, TNs, bottom=TPs, label='TN',color='lightgreen')
        ax.bar(labels, FNs, bottom=TPs+TNs,label='FN',color='sandybrown')
        ax.bar(labels, FPs, bottom=TPs+TNs+FNs,label='FP',color='firebrick')
        
        print(labels)
        print(accuracies)

        for i,acc in enumerate(accuracies):
            plt.text(x=i-0.1,y=TPs[i]/2,s=str(TPs[i]),fontsize=8)
            plt.text(x=i-0.1,y=TPs[i] + TNs[i]/2,s=str(TNs[i]),fontsize=8)
            plt.text(x=i-0.1,y=TPs[i] + TNs[i] + FNs[i]/2,s=str(FNs[i]),fontsize=8)
            plt.text(x=i-0.1,y=TPs[i] + TNs[i] + FNs[i] + FPs[i]/2,s=str(FPs[i]),fontsize=8)
            plt.text(x=i-0.15,y=TPs[i]+TNs[i]+FPs[i]+FNs[i]+1,s=str(TPs[i] + TNs[i] + FNs[i] + FPs[i]))
        ax.legend()
        plt.xticks(range(len(labels)), labels, rotation=70)
        ax.yaxis.set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        fig.autofmt_xdate() 

        plt.show()


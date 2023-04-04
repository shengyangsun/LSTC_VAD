from sklearn import metrics
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from matplotlib import pyplot as plt


def cal_f1(scores,labels):
    return metrics.f1_score(labels,scores)

def cal_rmse(scores,labels):
    return metrics.mean_squared_error(labels,scores)**0.5


def cal_pr_auc(scores,labels):
    precision, recall, thresholds = metrics.precision_recall_curve(labels, scores, pos_label=1)
    auc = metrics.auc(recall, precision)
    return auc

def cal_auc(scores,labels):
    fpr, tpr, thresholds=metrics.roc_curve(labels,scores,pos_label=1)
    auc=metrics.auc(fpr,tpr)
    return auc

def cal_false_alarm(scores,labels,threshold=0.5):
    scores=np.array([1 if score>threshold else 0 for score in scores],dtype=float)
    fp=np.sum(scores*(1-labels))
    return fp/np.sum(1-labels)

def cal_false_neg(scores,labels,threshold=0.5):
    scores=np.array([1 if score>threshold else 0 for score in scores],dtype=float)
    fn=np.sum((1-scores)*(labels))
    return fn/np.sum(labels)

def cal_precision(scores,labels,threshold=0.5):
    scores=np.array([1 if score>threshold else 0 for score in scores],dtype=float)
    tp=np.sum(scores*labels)
    return tp/np.sum(scores)

def cal_accuracy(scores,labels,threshold=0.5):
    scores=np.array([1 if score>threshold else 0 for score in scores],dtype=float)
    tp=np.sum(scores*labels)
    tn=np.sum((1-scores)*(1-labels))
    return np.sum(tp+tn)/scores.shape[0]


def cal_recall(scores,labels,threshold=0.5):
    scores=np.array([1 if score>threshold else 0 for score in scores],dtype=float)
    tp=np.sum(scores*labels)
    fn=np.sum((1-scores)*labels)

    return tp/(tp+fn)

def cal_specific(scores,labels,threshold=0.5):
    scores=np.array([1 if score>threshold else 0 for score in scores],dtype=float)
    tn=np.sum((1-labels)*(1-scores))
    return tn/np.sum(1-labels)

def cal_sensitivity(scores,labels,threshold=0.50):
    scores=np.array([1 if score>threshold else 0 for score in scores],dtype=float)
    tp=np.sum(scores*labels)
    return tp/np.sum(labels)

def cal_score_gap(scores,labels):
    labels=labels.astype(bool)
    neg_labels=(1-labels).astype(bool)
    positive=np.mean(scores[labels])
    negative=np.mean(scores[neg_labels])
    return positive-negative

def cal_geometric_mean(scores,labels,threshold=0.5):
    tn=cal_specific(scores,labels,threshold)
    tp=cal_sensitivity(scores,labels,threshold)
    return np.sqrt(tp*tn)

def cal_f_measure(scores,labels,threshold=0.5):
    p=cal_precision(scores,labels,threshold)
    r=cal_recall(scores,labels,threshold)
    return 2*p*r/(p+r)

def cal_MCC(scores,labels,threshold=0.5):
    scores=np.array([1 if score>threshold else 0 for score in scores],dtype=float)
    tp=np.sum(scores*labels)
    tn=np.sum((1-scores)*(1-labels))
    fp=np.sum(scores*(1-labels))
    fn=np.sum((1-scores)*labels)
    return (tp*tn-fp*fn)/np.sqrt((tp+fp)*(fp+fn)*(tn+fp)*(tn+fn))

def cal_pAUC(scores,labels):
    sum_gt_p=np.sum(labels)
    sum_gt_n=labels.shape[0]-sum_gt_p
    sum_pred_p=np.sum(scores[labels.astype(bool)])
    sum_pred_n=np.sum(scores[(1-labels).astype(bool)])
    return 0.5*(sum_pred_p/sum_gt_p-sum_pred_n/sum_gt_n+1)

def eval_each_part(labels_dict,scores_dict,logger=None):
    map = 0
    for key in labels_dict.keys():
        score=scores_dict[key]
        if key=='Normal':
            auc='None'
            pr_auc='None'
            gap='None'
            false_alarm=cal_false_alarm(np.array(score,dtype=float),np.array(labels_dict[key],dtype=float))
            normal_far=false_alarm
            if logger == None:
                print('{}: \tAUC \t{}, PR-AUC \t{}, FAR \t{}\tGAP\t{}'.format(key, auc, pr_auc, normal_far, gap))
            else:
                logger.info('{}: \tAUC \t{}, PR-AUC \t{}, FAR \t{}\tGAP\t{}'.format(key,auc,pr_auc,normal_far,gap))
        else:
            # print(type,np.array(score).shape,np.array((labels_dict[type])).shape)
            auc=cal_auc(np.array(score,dtype=float),np.array(labels_dict[key],dtype=float))
            pr_auc=cal_pr_auc(np.array(score,dtype=float),np.array(labels_dict[key],dtype=float))
            map+=pr_auc
            gap=cal_score_gap(np.array(score,dtype=float),np.array(labels_dict[key],dtype=float))
            false_alarm=cal_false_alarm(np.array(score,dtype=float),np.array(labels_dict[key],dtype=float))
            if logger==None:
                print('{}: \tAUC \t{:.4f}, PR-AUC \t{:.4f}, FAR \t{}\tGAP\t{:.4f}'.format(key,auc,pr_auc,false_alarm,gap))
            else:
                logger.info('{}: \tAUC \t{:.4f}, PR-AUC \t{:.4f}, FAR \t{}\tGAP\t{:.4f}'.format(key,auc,pr_auc,false_alarm,gap))
    return normal_far,map/13

def eval_classification(logits,labels):
    logits=(torch.argmax(logits,dim=1))
    a=torch.le(labels,logits).float()
    b=torch.lt(labels,logits).float()
    accuracy_top_1=torch.mean(a-b)
    return accuracy_top_1

def eval_classification_binary(logits,labels):
    # all of them are with [N]
    N=logits.shape[0]
    positive_true=torch.nonzero(labels*F.relu(logits-0.5)).shape[0]
    negative_true=torch.nonzero(((1-labels)*F.relu(0.5-logits))).shape[0]
    return (positive_true+negative_true)/N


def eval(total_scores,total_labels,logger):
    total_scores = np.array(total_scores)
    total_labels = np.array(total_labels)
    auc = cal_auc(total_scores, total_labels)
    return auc

def cal_AP(scores, labels):
    y_true = np.array(labels)
    y_scores = np.array(scores)
    return average_precision_score(y_true, y_scores)

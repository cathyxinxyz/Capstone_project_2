from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score, cohen_kappa_score, precision_recall_curve,roc_curve, auc, precision_recall_curve, average_precision_score

import matplotlib.pyplot as plt

import numpy as np

def Performance_report(grid, X_train, y_train, X_test, y_test, printout=False):
    y_trained_pred=grid.predict(X_train)
    
    training_metrics=dict()
    accuracy_train=accuracy_score(y_train, y_trained_pred)
    cohen_kappa_train=cohen_kappa_score(y_train, y_trained_pred)
    f1_per_class_train=f1_score(y_train, y_trained_pred, average=None)
    training_metrics['accuracy']=accuracy_train
    training_metrics['cohen_kappa']=cohen_kappa_train
    training_metrics['f1']=f1_score(y_train, y_trained_pred, average='macro')
    training_metrics['recall']=recall_score(y_train, y_trained_pred, average='macro')
    training_metrics['precision']=precision_score(y_train, y_trained_pred, average='macro')

    y_pred=grid.predict(X_test)
    testing_metrics=dict()
    accuracy=accuracy_score(y_test, y_pred)
    cohen_kappa=cohen_kappa_score(y_test, y_pred)
    
    testing_metrics['accuracy']=accuracy
    testing_metrics['cohen_kappa']=cohen_kappa
    
    testing_metrics['f1']=f1_score(y_test, y_pred, average='macro')
    testing_metrics['recall']=recall_score(y_test, y_pred, average='macro')
    testing_metrics['precision']=precision_score(y_test, y_pred, average='macro')
    
    
    if printout:
        print("Best: %f using %s" % (grid.best_score_, grid.best_params_)) 
        print ('\n------------------------------------------\n')
        report = classification_report(y_train, y_trained_pred)
        print ('the classification on training data are\n')
        print ('confusion matrix:\n')
        print (confusion_matrix(y_train, y_trained_pred))
        print ('accuracy={}\n'.format(accuracy_train))
        print ('cohen kappa={}\n'.format(cohen_kappa_train))
        print (report)
        
        print ('\n------------------------------------------\n')
        report = classification_report(y_test, y_pred)
        accuracy=accuracy_score(y_test, y_pred)
        print ('the classification on test data are\n')
        print ('confusion matrix:\n')
        print (confusion_matrix(y_test, y_pred))
        print ('accuracy={}\n'.format(accuracy))
        print ('cohen kappa={}\n'.format(cohen_kappa))
        print (report)
    
    return (training_metrics, testing_metrics)

def PR(grid, X_test, y_test):
    y_score=grid.predict_proba(X_test)
    
    precision, recall, _ = precision_recall_curve(y_test, y_score[:,1])

    plt.step(recall, precision, color='b', alpha=0.2,where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

def ROC(grid, X_test, y_test):
    y_score=grid.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='r', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='k', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

# This function plots the precision-recall curve for both 
def PR_by_class(grid, X, y):
    y_score=grid.predict_proba(X)

    colors=['blue', 'red']
    """add another column to y_test to rewrite the label of class 0 as 1, 
    and that of class 1 as 0 so that class 0 is the class to be predicted"""
    y_binary=np.stack([1-y, y], axis=-1)  

    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(2):
        precision[i], recall[i], _ = precision_recall_curve(y_binary[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_binary[:, i], y_score[:, i])
    
    lines=list()
    labels=list()
    for i, color in zip(range(2), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'.format(i, average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve of each class')
    plt.legend(lines, labels, loc=(0, -.5), prop=dict(size=14))

    plt.show()

def ROC_by_class(grid, X_test, y_test):
    
    colors=['blue', 'red']
    """add another column to y_test to rewrite the label of class 0 as 1, 
    and that of class 1 as 0 so that class 0 is the class to be predicted"""
    y_test_binary=np.stack([1-y_test, y_test], axis=-1)
    y_score=grid.predict_proba(X_test)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    
    for i,c in zip(range(2), colors):
        plt.plot(fpr[i], tpr[i], color=c, label='ROC curve (area = %0.2f)' % roc_auc[i]+' of class %d'%i)
        plt.plot([0, 1], [0, 1], color='k', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

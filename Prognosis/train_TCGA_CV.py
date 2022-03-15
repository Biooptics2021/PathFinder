import os
import logging
# Env
from data_loaders import *
from train_test import train, test


def get_files(path, rule=".npy"):
    all = []
    for fpathe,dirs,fs in os.walk(path):
        for f in fs:
            filename = os.path.join(fpathe,f)
            if filename.endswith(rule):
                all.append(filename)
    return all


def path_cleaning(hospital, base_dir='xx'):

    TCGA_path = ''#heatmap .npy path
    TCGA_HCC_path = ''#clinical info
    path = TCGA_path

    cleaned_path = []
    seg_list = get_files(path)
    TCGA_HCC_list = list(pd.read_csv(TCGA_HCC_path)['WSIs'])
    clinical_list = list(pd.read_csv(base_dir+hospital+'.csv')['WSIs'])

    for i in seg_list:
        clinical_list = list(set(clinical_list).intersection(set(TCGA_HCC_list)))

        if i.split('/')[-1][:12] in clinical_list:
            cleaned_path.append(i)

    return cleaned_path


def path_k_fold(path_list, k):
    k_fold_path = []
    length = len(path_list)
    step_size = int(length/k)
    for i in range(k):
        if i<k-1:
            test_set = path_list[i*step_size:(i+1)*step_size]
            train_set = list(set(path_list).difference(set(test_set)))
        else:
            test_set = path_list[i*step_size:(i+1)*step_size]
            train_set = list(set(path_list).difference(set(test_set)))
        k_fold_path.append([train_set, test_set])
    
    return k_fold_path




if __name__ == 'main':
    TCGA_list = path_cleaning("TCGA")
    cv_path = path_k_fold(TCGA_list,k = 10)
    k = 0
    for path in cv_path:
        train_data = path[0]
        test_data = path[1]
        print(len(train_data))
        print(len(test_data))


        model, optimizer, metric_logger = train(train_data,test_data,k)
        k += 1
        loss_train, cindex_train, pvalue_train, surv_acc_train, grad_acc_train, pred_train = test(model, train_data)
        loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(model, test_data)

        print("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
        logging.info("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
        print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
        logging.info("[Final] Apply model to testing set: cC-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))















































##单步训练
# print(len(a[0][0]))
# print(len(a[0][1]))
# print(len(TCGA_list))



# print(len(QHCG_list))
# print(len(QYFY_list))
# print(QYFY_list)
# print(len(TCGA_list))

# train_data = TCGA_list[:300]
# test_data = TCGA_list[300:]



# model, optimizer, metric_logger = train(train_data,test_data)
# loss_train, cindex_train, pvalue_train, surv_acc_train, grad_acc_train, pred_train = test(model, train_data)
# loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(model, test_data)

# print("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
# logging.info("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
# print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
# logging.info("[Final] Apply model to testing set: cC-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))

#save model
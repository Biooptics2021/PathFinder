import os
import logging
# Env
from data_loaders import *
from train_test import test, train


def get_files(path, rule=".npy"):
    all = []
    for fpathe,dirs,fs in os.walk(path):
        for f in fs:
            filename = os.path.join(fpathe,f)
            if filename.endswith(rule):
                all.append(filename)
    return all


#select max TUM ratio WSI as patient WSI
def max_path_cleaning(hospital, base_dir=''):

    QHCG_path = ''#heatmap .npy path
    TCGA_path = ''#heatmap .npy path
    TCGA_HCC_path = ''#clinical info path
    QHCG_TUM_MAX_path = ''#clinical info path

    path = ''
    if hospital == 'TCGA':
        path = TCGA_path
    else:
        path = QHCG_path

    cleaned_path = []
    seg_list = get_files(path)
    TCGA_HCC_list = list(pd.read_csv(TCGA_HCC_path)['WSIs'])
    QHCG_TUM_MAX_list = list(pd.read_csv(QHCG_TUM_MAX_path)['WSI_name'])
    clinical_list = list(pd.read_csv(base_dir+hospital+'.csv')['WSIs'])

    for i in seg_list:
        if hospital == 'TCGA':
            clinical_list = list(set(clinical_list).intersection(set(TCGA_HCC_list)))
            if i.split('/')[-1][:12] in clinical_list:
                cleaned_path.append(i)
        else:#QHCG
            if i.split('/')[-1][:-7] in clinical_list:
                if i.split('/')[-1][:-4] in QHCG_TUM_MAX_list:
                    cleaned_path.append(i)

    return cleaned_path


if __name__ == 'main':
    QHCG_list = max_path_cleaning("QHCG")
    TCGA_list = max_path_cleaning("TCGA")

    train_data = TCGA_list
    test_data = QHCG_list

    model, optimizer, metric_logger = train(train_data,test_data,k_th_fold=0)
    loss_train, cindex_train, pvalue_train, surv_acc_train, grad_acc_train, pred_train = test(model, train_data)
    loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(model, test_data)

    print("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
    logging.info("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
    print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
    logging.info("[Final] Apply model to testing set: cC-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
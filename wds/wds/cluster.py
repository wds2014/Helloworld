from sklearn import metrics
from cluster_acc import acc

nmi = metrics.normalized_mutual_info_score(train_label,my_dp.z)
ari = metrics.adjusted_rand_score(train_label,my_dp.z)
label_uni = len(np.unique(my_dp.z))
label_true_uni = len(np.unique(train_label))
    if label_uni <label_true_uni+1:
        acc_value = acc(my_dp.z,train_label)
    else:
        acc_value=0

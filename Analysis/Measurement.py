import numpy as np
from MeDIT.Statistics import BinaryClassification
from sklearn.metrics import roc_auc_score


def classification_stat(gs, radiologist, model, dmax):
    gs = [1 if x > 0 else 0 for x in gs]

    # for AUC
    radiologist = [(x-1)*0.25 for x in radiologist]
    model = [(x-1)*0.25 for x in model]

    # for confusion matrix
    # radiologist = [1 if x >= 3 else 0 for x in radiologist]
    # model = [1 if x >= 3 else 0 for x in model]

    gs_small = [gs[i] for i in range(len(dmax)) if dmax[i] < 1.5]
    gs_large = [gs[i] for i in range(len(dmax)) if dmax[i] >= 1.5]
    radiologist_small = [radiologist[i] for i in range(len(dmax)) if dmax[i] < 1.5]
    radiologist_large = [radiologist[i] for i in range(len(dmax)) if dmax[i] >= 1.5]
    model_small = [model[i] for i in range(len(dmax)) if dmax[i] < 1.5]
    model_large = [model[i] for i in range(len(dmax)) if dmax[i] >= 1.5]

    # print(roc_auc_score(gs, model))
    # print(roc_auc_score(gs_small, model_small))
    # print(roc_auc_score(gs_large, model_large))
    # print(roc_auc_score(gs, radiologist))
    # print(roc_auc_score(gs_small, radiologist_small))
    # print(roc_auc_score(gs_large, radiologist_large))

    bc = BinaryClassification(is_show=False)

    bc.Run(model, gs)
    print(f'Model: {bc.metric}')
    bc.Run(model_small, gs_small)
    print(f'Model small: {bc.metric}')
    bc.Run(model_large, gs_large)
    print(f'Model large: {bc.metric}')

    bc.Run(radiologist, gs)
    print(f'Radiologist: {bc.metric}')
    bc.Run(radiologist_small, gs_small)
    print(f'Radiologist small: {bc.metric}')
    bc.Run(radiologist_large, gs_large)
    print(f'Radiologist large: {bc.metric}')

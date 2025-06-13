import geopandas as gpd
from rsome import ro
from rsome import grb_solver as grb
from random import shuffle
from gurobipy import *
import gurobipy as gp
import json
import math
import time
from GNN.autoencoder import VGAE, GAE, EdgeDecoder, DirectedEdgeDecoder, InnerProductDecoder, \
    DirectedInnerProductDecoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import networkx as nx
import copy
import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GraphConv, SAGEConv
import torch.nn.functional as F
from torch_geometric.transforms import LineGraph, RandomNodeSplit
from torch_geometric import seed_everything
from torch import nn

###############################
EPOCHS = 1501
ALPHA = 0.05
LR = 0.01
WD = 5e-4
HIDDEN = 8
OUT = 2
HIDDENS = [4, 8, 16, 32]


def check_range(lower_array, upper_array, label_array):
    count = 0
    total = len(label_array)

    for lower, upper, label in zip(lower_array, upper_array, label_array):
        if lower <= label <= upper:
            count += 1

    ratio = count / total
    return ratio


def dump_files(NAME, OBJ_CQR, OBJ_CQR_NEW, OBJ_QR, OBJ_BL,
               OBJ_CQR_LABEL, OBJ_CQR_NEW_LABEL, OBJ_QR_LABEL,
               OBJ_BL_LABEL, COV_CQR, COV_CQR_NEW, COV_QR, COV_BL,
               X_CQR, X_CQR_NEW, X_QR, X_BL):
    json.dump(X_CQR, open(
        r'D:\OneDrive - City University of Hong Kong - Student\PyCharm202122\shortest_path202403\GNN\Data' + NAME + '\X_CQR.json',
        'w'))
    json.dump(X_CQR_NEW, open(
        r'D:\OneDrive - City University of Hong Kong - Student\PyCharm202122\shortest_path202403\GNN\Data' + NAME + '\X_CQR_NEW.json',
        'w'))
    json.dump(X_QR, open(
        r'D:\OneDrive - City University of Hong Kong - Student\PyCharm202122\shortest_path202403\GNN\Data' + NAME + '\X_QR.json',
        'w'))
    json.dump(X_BL, open(
        r'D:\OneDrive - City University of Hong Kong - Student\PyCharm202122\shortest_path202403\GNN\Data' + NAME + '\X_BL.json',
        'w'))
    json.dump(OBJ_CQR, open(
        r'D:\OneDrive - City University of Hong Kong - Student\PyCharm202122\shortest_path202403\GNN\Data' + NAME + '\OBJ_CQR.json',
        'w'))
    json.dump(OBJ_CQR_NEW, open(
        r'D:\OneDrive - City University of Hong Kong - Student\PyCharm202122\shortest_path202403\GNN\Data' + NAME + '\OBJ_CQR_NEW.json',
        'w'))
    json.dump(OBJ_QR, open(
        r'D:\OneDrive - City University of Hong Kong - Student\PyCharm202122\shortest_path202403\GNN\Data' + NAME + '\OBJ_QR.json',
        'w'))
    json.dump(OBJ_BL, open(
        r'D:\OneDrive - City University of Hong Kong - Student\PyCharm202122\shortest_path202403\GNN\Data' + NAME + '\OBJ_BL.json',
        'w'))
    json.dump(OBJ_CQR_LABEL, open(
        r'D:\OneDrive - City University of Hong Kong - Student\PyCharm202122\shortest_path202403\GNN\Data' + NAME + '\OBJ_CQR_LABEL.json',
        'w'))
    json.dump(OBJ_CQR_NEW_LABEL, open(
        r'D:\OneDrive - City University of Hong Kong - Student\PyCharm202122\shortest_path202403\GNN\Data' + NAME + '\OBJ_CQR_NEW_LABEL.json',
        'w'))
    json.dump(OBJ_QR_LABEL, open(
        r'D:\OneDrive - City University of Hong Kong - Student\PyCharm202122\shortest_path202403\GNN\Data' + NAME + '\OBJ_QR_LABEL.json',
        'w'))
    json.dump(OBJ_BL_LABEL, open(
        r'D:\OneDrive - City University of Hong Kong - Student\PyCharm202122\shortest_path202403\GNN\Data' + NAME + '\OBJ_BL_LABEL.json',
        'w'))
    json.dump(COV_CQR, open(
        r'D:\OneDrive - City University of Hong Kong - Student\PyCharm202122\shortest_path202403\GNN\Data' + NAME + '\COV_CQR.json',
        'w'))
    json.dump(COV_CQR_NEW, open(
        r'D:\OneDrive - City University of Hong Kong - Student\PyCharm202122\shortest_path202403\GNN\Data' + NAME + '\COV_CQR_NEW.json',
        'w'))
    json.dump(COV_QR, open(
        r'D:\OneDrive - City University of Hong Kong - Student\PyCharm202122\shortest_path202403\GNN\Data' + NAME + '\COV_QR.json',
        'w'))
    json.dump(COV_BL, open(
        r'D:\OneDrive - City University of Hong Kong - Student\PyCharm202122\shortest_path202403\GNN\Data' + NAME + '\COV_BL.json',
        'w'))


def dump_files_NO_OPT(NAME, CQR_interval, CQR_NEW_interval, QR_interval, BL_interval, C_LABEL, COV_CQR, COV_CQR_NEW,
                      COV_QR, COV_BL):
    json.dump(CQR_interval, open(
        r'D:\OneDrive - City University of Hong Kong - Student\PyCharm202122\shortest_path202403\GNN\Data' + NAME + '\CQR_interval.json',
        'w'))
    json.dump(CQR_NEW_interval, open(
        r'D:\OneDrive - City University of Hong Kong - Student\PyCharm202122\shortest_path202403\GNN\Data' + NAME + '\CQR_NEW_interval.json',
        'w'))
    json.dump(QR_interval, open(
        r'D:\OneDrive - City University of Hong Kong - Student\PyCharm202122\shortest_path202403\GNN\Data' + NAME + '\QR_interval.json',
        'w'))
    json.dump(BL_interval, open(
        r'D:\OneDrive - City University of Hong Kong - Student\PyCharm202122\shortest_path202403\GNN\Data' + NAME + '\BL_interval.json',
        'w'))
    json.dump(C_LABEL, open(
        r'D:\OneDrive - City University of Hong Kong - Student\PyCharm202122\shortest_path202403\GNN\Data' + NAME + '\C_LABEL.json',
        'w'))
    json.dump(COV_CQR, open(
        r'D:\OneDrive - City University of Hong Kong - Student\PyCharm202122\shortest_path202403\GNN\Data' + NAME + '\COV_CQR.json',
        'w'))
    json.dump(COV_CQR_NEW, open(
        r'D:\OneDrive - City University of Hong Kong - Student\PyCharm202122\shortest_path202403\GNN\Data' + NAME + '\COV_CQR_NEW.json',
        'w'))
    json.dump(COV_QR, open(
        r'D:\OneDrive - City University of Hong Kong - Student\PyCharm202122\shortest_path202403\GNN\Data' + NAME + '\COV_QR.json',
        'w'))
    json.dump(COV_BL, open(
        r'D:\OneDrive - City University of Hong Kong - Student\PyCharm202122\shortest_path202403\GNN\Data' + NAME + '\COV_BL.json',
        'w'))


def calculate_coverage(lower_dictionary, upper_dictionary, label_dictionary, keys):
    m = len(lower_dictionary)
    result_dictionary = {}
    count_ones = 0

    for key in keys:
        lower_value = lower_dictionary[key]
        upper_value = upper_dictionary[key]
        label_value = label_dictionary[key][0]

        if lower_value <= label_value <= upper_value:
            result_dictionary[key] = 1
            count_ones += 1
        else:
            result_dictionary[key] = 0

    ratio = count_ones / m

    return ratio


def new_set_minus(SET, a):
    SET_new = copy.deepcopy(SET)
    SET_new.remove(a)
    new_SET = SET_new
    return new_SET


# cal_labels 430x1,
# lower upper(from GAE)
# n (430)(dimension of calib)
def cqr(cal_labels, cal_lower, cal_upper, test_labels, test_lower, test_upper, n, alpha):
    cal_scores = np.maximum(cal_labels - cal_upper, cal_lower - cal_labels)
    qhat = np.quantile(cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, method='higher')
    prediction_sets = [test_lower - qhat, test_upper + qhat]
    cov = ((test_labels >= prediction_sets[0]) & (test_labels <= prediction_sets[1])).mean()
    eff = np.mean(test_upper + qhat - (test_lower - qhat))
    return prediction_sets, cov, eff


def cqr_new(cal_labels, cal_lower, cal_upper, test_labels, test_lower, test_upper, n, alpha):
    cal_scores = np.maximum((cal_labels - cal_upper) / np.abs(cal_upper - cal_lower),
                            (cal_lower - cal_labels) / np.abs(cal_upper - cal_lower))
    qhat = np.quantile(cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, method='higher')
    prediction_sets = [test_lower - qhat * np.abs(test_upper - test_lower),
                       test_upper + qhat * np.abs(test_upper - test_lower)]
    cov = ((test_labels >= prediction_sets[0]) & (test_labels <= prediction_sets[1])).mean()
    eff = np.mean(prediction_sets[1] - prediction_sets[0])
    return prediction_sets, cov, eff




# set all infor into dictionaries
# calib_test_info, train_info, val_info
# dict = {(edge_index): [edge_weight, upper, lower]}, for train&val, upper = lower = 0
def dictionary_all(edge_weight_calib_test, edge_weight_train, edge_weight_val, edge_index_calib_test, edge_index_train,
                   edge_index_val, calib_test_upper, calib_test_lower):
    # calib_test_upper_cqr = cqr(edge_weight_calib_test, )
    # calib_test_info_cqr = {}
    # for i in range(len(edge_weight_calib_test)):
    #     calib_test_info_cqr[(edge_index_calib_test[0][i], edge_index_calib_test[1][i])] = [edge_weight_calib_test[i],
    #                                                                                          calib_test_upper_cqr[i]]
    calib_test_info = {}
    for i in range(len(edge_weight_calib_test)):
        calib_test_info[(edge_index_calib_test[0][i], edge_index_calib_test[1][i])] = [edge_weight_calib_test[i],
                                                                                       calib_test_upper[i],
                                                                                       calib_test_lower[i]]
    train_info = {}
    for i in range(len(edge_weight_train)):
        train_info[(edge_index_train[0][i], edge_index_train[1][i])] = [edge_weight_train[i], 0, 0]
    val_info = {}
    for i in range(len(edge_weight_val)):
        val_info[(edge_index_val[0][i], edge_index_val[1][i])] = [edge_weight_val[i], 0,
                                                                  0]

    return calib_test_info, train_info, val_info


def gurobi_optimize_short_path(nodeNum, S, T, cost):
    nodes = [i for i in range(nodeNum)]

    # create a gurobi model
    m = gp.Model("VaRO")

    # create decision variables
    x = {}  # decision v
    E_ij = {}
    for i in range(nodeNum):
        for j in range(nodeNum):
            if (i != j):
                E_ij[(i, j)] = (i, j)
                x[(i, j)] = m.addVar(vtype=GRB.BINARY, name='x_' + str(i) + '_' + str(j))
    # set obj
    obj = sum(cost[i, j] * x[(i, j)] for (i, j) in E_ij)
    m.setObjective(obj, GRB.MINIMIZE)

    # constraints
    # starting node
    m.addConstr(
        sum(x[(S, j)] for j in new_set_minus(nodes, S)) - sum(x[(j, S)] for j in new_set_minus(nodes, S)) == 1)
    # terminal node
    m.addConstr(
        sum(x[(i, T)] for i in new_set_minus(nodes, T)) - sum(x[(T, i)] for i in new_set_minus(nodes, T)) == 1)
    # linking nodes
    nodes_new = [i for i in range(nodeNum)]
    nodes_new.remove(T)
    nodes_new.remove(S)
    nodes_ST = nodes_new
    for i in nodes_ST:
        nodes_i = new_set_minus(nodes, i)
        m.addConstr(sum(x[(i, j)] for j in nodes_i) - sum(x[(j, i)] for j in nodes_i) == 0)

    # solver
    m.optimize()

    # output result
    x_list = []
    cost_list = []
    for (i, j) in E_ij:
        if x[(i, j)].x != 0:
            x_list.append((i, j))
            cost_list.append(cost[i, j])
            # print('x_%d_%d=%d\t' % (i, j, x[(i, j)].x))
            # print('cost=%f\t' % (cost[i, j]))

    return x_list, cost_list, m.objVal


def gurobi_optimize_matrix(nodeNum, S, T, index_all, c_box_lb, c_box_ub, c_label):
    model = ro.Model()
    w = model.dvar(len(index_all))
    c = model.rvar(len(index_all))
    A = create_node_arc_incidence_matrix(nodeNum, index_all)
    b = np.zeros(nodeNum)
    b[S] = 1
    b[T] = -1
    c_box_lb = np.transpose(c_box_lb)
    c_box_ub = np.transpose(c_box_ub)
    uset = (c >= c_box_lb, c <= c_box_ub)

    model.minmax(c @ w, uset)
    model.st(w <= 1)
    model.st(w >= 0)
    model.st(A[m] @ w == b[m] for m in range(0, len(b)))

    model.solve(grb)
    if model.solution.status == 2:
        X_MATRIX = w.get()
        obj_label = c_label @ X_MATRIX
        OBJ_MATRIX = model.get()
        # print(OBJ_MATRIX)

        X_MATRIX_list = []
        for m in range(0, len(X_MATRIX)):
            if X_MATRIX[m] == 1:
                X_MATRIX_list.append(index_all[m].tolist())
    else:
        X_MATRIX_list = []
        OBJ_MATRIX = 0.1
        obj_label = 0.1
        print('****************************************** ' + str(S) + ',' + str(
            T) + ' infeasible ****************************************')
    return X_MATRIX_list, OBJ_MATRIX, obj_label


def check_and_update(a, b, c, d, e, f, g, h):
    if 0.1 in (a, b, c, d, e, f, g, h):
        a, b, c, d, e, f, g, h = 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
    return a, b, c, d, e, f, g, h


def x_cost_to_obj(x, cost):
    obj = 0
    for (i, j) in x:
        obj += cost[i, j]
    return obj


def obtain_prediction_interval(edge_weight_calib_test_1x860, edge_weight_train_1x1075,
                               edge_weight_val_1x215,
                               edge_index_calib_test_2x2, edge_index_train_2x2,
                               edge_index_val_2x2, calib_test_upper_list, calib_test_lower_list,
                               count, S=0, T=540, alpha=0.05,
                               epsilon=0.01,
                               nodeNum=546):
    # generate dicts for calib_test, train and val
    calib_test_info_dic, train_info_dic, val_info_dic = dictionary_all(edge_weight_calib_test_1x860,
                                                                       edge_weight_train_1x1075,
                                                                       edge_weight_val_1x215,
                                                                       edge_index_calib_test_2x2, edge_index_train_2x2,
                                                                       edge_index_val_2x2, calib_test_upper_list,
                                                                       calib_test_lower_list)

    key_calib_test = calib_test_info_dic.keys()
    key_calib_test_list = list(key_calib_test)
    num = int(len(key_calib_test_list) / 2)

    #################################### create data set ###########################################
    X_CQR_MATRIX_dic = {}
    OBJ_CQR_MATRIX_dic = {}
    OBJ_CQR_LABEL_MATRIX_dic = {}

    X_CQR_NEW_MATRIX_dic = {}
    OBJ_CQR_NEW_MATRIX_dic = {}
    OBJ_CQR_NEW_LABEL_MATRIX_dic = {}

    X_QR_MATRIX_dic = {}
    OBJ_QR_MATRIX_dic = {}
    OBJ_QR_LABEL_MATRIX_dic = {}

    X_BL_MATRIX_dic = {}
    OBJ_BL_MATRIX_dic = {}
    OBJ_BL_LABEL_MATRIX_dic = {}

    cov_cqr = {}
    cov_cqr_new = {}
    cov_qr = {}
    cov_bl = {}

    for k in range(0, count):
        shuffle(key_calib_test_list)
        key_calib = key_calib_test_list[: num]  # the first half
        key_test = key_calib_test_list[num:]  # the last half

        # generate cqr, cqr_new upper for calib
        edge_weight_calib_1x430_shuffle = []
        calib_lower_list_shuffle = []
        calib_upper_list_shuffle = []
        edge_weight_test_1x430_shuffle = []
        test_lower_list_shuffle = []
        test_upper_list_shuffle = []
        for key in key_calib:
            edge_weight_calib_1x430_shuffle.append(calib_test_info_dic[key][0])
            calib_upper_list_shuffle.append(calib_test_info_dic[key][1])
            calib_lower_list_shuffle.append(calib_test_info_dic[key][2])
        for key in key_test:
            edge_weight_test_1x430_shuffle.append(calib_test_info_dic[key][0])
            test_upper_list_shuffle.append(calib_test_info_dic[key][1])
            test_lower_list_shuffle.append(calib_test_info_dic[key][2])

        edge_weight_calib_1x430_shuffle_array = np.array(edge_weight_calib_1x430_shuffle)
        calib_lower_list_shuffle_array = np.array(calib_lower_list_shuffle)
        calib_upper_list_shuffle_array = np.array(calib_upper_list_shuffle)
        edge_weight_test_1x430_shuffle_array = np.array(edge_weight_test_1x430_shuffle)
        test_lower_list_shuffle_array = np.array(test_lower_list_shuffle)
        test_upper_list_shuffle_array = np.array(test_upper_list_shuffle)

        prediction_sets_cqr, cov, eff = cqr(edge_weight_calib_1x430_shuffle_array, calib_lower_list_shuffle_array,
                                            calib_upper_list_shuffle_array, edge_weight_test_1x430_shuffle_array,
                                            test_lower_list_shuffle_array, test_upper_list_shuffle_array,
                                            n=len(key_test),
                                            alpha=alpha)

        prediction_sets_cqr_new, cov, eff = cqr_new(edge_weight_calib_1x430_shuffle_array,
                                                    calib_lower_list_shuffle_array,
                                                    calib_upper_list_shuffle_array,
                                                    edge_weight_test_1x430_shuffle_array,
                                                    test_lower_list_shuffle_array, test_upper_list_shuffle_array,
                                                    n=len(key_test),
                                                    alpha=alpha)

        ##############################################gurobi_matrix##########################################################
        index_all = np.vstack((np.array(list(val_info_dic.keys())), np.array(list(train_info_dic.keys())),
                               np.array(key_calib), np.array(key_test)))
        c_label = np.hstack((
            edge_weight_val_1x215, edge_weight_train_1x1075, edge_weight_calib_1x430_shuffle_array,
            edge_weight_test_1x430_shuffle_array))
        key_shuffle = 'shuffle_' + str(k)

        # cqr_new model
        c_box_lb_cqr_new = np.hstack((
            edge_weight_val_1x215, edge_weight_train_1x1075, np.array(edge_weight_calib_1x430_shuffle),
            np.array(prediction_sets_cqr_new[0])))
        c_box_ub_cqr_new = np.hstack((
            edge_weight_val_1x215, edge_weight_train_1x1075, np.array(edge_weight_calib_1x430_shuffle),
            np.array(prediction_sets_cqr_new[1])))
        X_CQR_NEW_MATRIX, OBJ_CQR_NEW_MATRIX, OBJ_CQR_NEW_LABEL_MATRIX = gurobi_optimize_matrix(nodeNum, S, T,
                                                                                                index_all,
                                                                                                c_box_lb_cqr_new,
                                                                                                c_box_ub_cqr_new,
                                                                                                c_label)

        # cqr model
        c_box_lb_cqr = np.hstack((
            edge_weight_val_1x215, edge_weight_train_1x1075, np.array(edge_weight_calib_1x430_shuffle),
            np.array(prediction_sets_cqr[0])))
        c_box_ub_cqr = np.hstack((
            edge_weight_val_1x215, edge_weight_train_1x1075, np.array(edge_weight_calib_1x430_shuffle),
            np.array(prediction_sets_cqr[1])))
        X_CQR_MATRIX, OBJ_CQR_MATRIX, OBJ_CQR_LABEL_MATRIX = gurobi_optimize_matrix(nodeNum, S, T, index_all,
                                                                                    c_box_lb_cqr,
                                                                                    c_box_ub_cqr, c_label)

        # qr model
        c_box_lb_qr = np.hstack((
            edge_weight_val_1x215, edge_weight_train_1x1075, np.array(edge_weight_calib_1x430_shuffle),
            test_lower_list_shuffle_array))
        c_box_ub_qr = np.hstack((
            edge_weight_val_1x215, edge_weight_train_1x1075, np.array(edge_weight_calib_1x430_shuffle),
            test_upper_list_shuffle_array))
        X_QR_MATRIX, OBJ_QR_MATRIX, OBJ_QR_LABEL_MATRIX = gurobi_optimize_matrix(nodeNum, S, T, index_all, c_box_lb_qr,
                                                                                 c_box_ub_qr, c_label)

        # baseline model
        baseline_average = np.quantile(np.hstack([edge_weight_train_1x1075, edge_weight_val_1x215]), q=1 - alpha / 2,
                                       axis=0)
        bl_vector = np.full_like(test_upper_list_shuffle_array, baseline_average)
        c_box_lb_bl = np.hstack((
            edge_weight_val_1x215, edge_weight_train_1x1075, np.array(edge_weight_calib_1x430_shuffle), bl_vector))
        c_box_ub_bl = np.hstack((
            edge_weight_val_1x215, edge_weight_train_1x1075, np.array(edge_weight_calib_1x430_shuffle),
            bl_vector))
        X_BL_MATRIX, OBJ_BL_MATRIX, OBJ_BL_LABEL_MATRIX = gurobi_optimize_matrix(nodeNum, S, T, index_all, c_box_lb_bl,
                                                                                 c_box_ub_bl, c_label)

        OBJ_CQR_NEW_MATRIX, OBJ_CQR_NEW_LABEL_MATRIX, OBJ_CQR_MATRIX, OBJ_CQR_LABEL_MATRIX, OBJ_QR_MATRIX, \
        OBJ_QR_LABEL_MATRIX, OBJ_BL_MATRIX, OBJ_BL_LABEL_MATRIX = check_and_update(OBJ_CQR_NEW_MATRIX,
                                                                                   OBJ_CQR_NEW_LABEL_MATRIX,
                                                                                   OBJ_CQR_MATRIX,
                                                                                   OBJ_CQR_LABEL_MATRIX,
                                                                                   OBJ_QR_MATRIX,
                                                                                   OBJ_QR_LABEL_MATRIX,
                                                                                   OBJ_BL_MATRIX,
                                                                                   OBJ_BL_LABEL_MATRIX)

        X_CQR_NEW_MATRIX_dic[key_shuffle] = X_CQR_NEW_MATRIX
        OBJ_CQR_NEW_MATRIX_dic[key_shuffle] = OBJ_CQR_NEW_MATRIX
        OBJ_CQR_NEW_LABEL_MATRIX_dic[key_shuffle] = OBJ_CQR_NEW_LABEL_MATRIX
        X_CQR_MATRIX_dic[key_shuffle] = X_CQR_MATRIX
        OBJ_CQR_MATRIX_dic[key_shuffle] = OBJ_CQR_MATRIX
        OBJ_CQR_LABEL_MATRIX_dic[key_shuffle] = OBJ_CQR_LABEL_MATRIX
        X_QR_MATRIX_dic[key_shuffle] = X_QR_MATRIX
        OBJ_QR_MATRIX_dic[key_shuffle] = OBJ_QR_MATRIX
        OBJ_QR_LABEL_MATRIX_dic[key_shuffle] = OBJ_QR_LABEL_MATRIX
        X_BL_MATRIX_dic[key_shuffle] = X_BL_MATRIX
        OBJ_BL_MATRIX_dic[key_shuffle] = OBJ_BL_MATRIX
        OBJ_BL_LABEL_MATRIX_dic[key_shuffle] = OBJ_BL_LABEL_MATRIX

        ###################################### use the 1st and 2nd dimension of prediction_sets ##############################

        a = np.array(prediction_sets_cqr_new[0]).tolist()
        b = np.array(prediction_sets_cqr_new[1]).tolist()
        c = edge_weight_test_1x430_shuffle_array.tolist()
        cov_cqr_new[key_shuffle] = check_range(a, b, c)
        a = np.array(prediction_sets_cqr[0]).tolist()
        b = np.array(prediction_sets_cqr[1]).tolist()
        c = edge_weight_test_1x430_shuffle_array.tolist()
        cov_cqr[key_shuffle] = check_range(a, b, c)
        a = test_lower_list_shuffle_array.tolist()
        b = test_upper_list_shuffle_array.tolist()
        c = edge_weight_test_1x430_shuffle_array.tolist()
        cov_qr[key_shuffle] = check_range(a, b, c)
        a = [0 for i in range(0, len(c))]
        b = bl_vector.tolist()
        cov_bl[key_shuffle] = check_range(a, b, c)

    return X_CQR_NEW_MATRIX_dic, OBJ_CQR_NEW_MATRIX_dic, OBJ_CQR_NEW_LABEL_MATRIX_dic, cov_cqr_new, X_CQR_MATRIX_dic, OBJ_CQR_MATRIX_dic, \
           OBJ_CQR_LABEL_MATRIX_dic, cov_cqr, X_QR_MATRIX_dic, OBJ_QR_MATRIX_dic, OBJ_QR_LABEL_MATRIX_dic, cov_qr, X_BL_MATRIX_dic, OBJ_BL_MATRIX_dic, OBJ_BL_LABEL_MATRIX_dic, cov_bl


def obtain_prediction_interval_c_label(edge_weight_calib_test_1x860, edge_weight_train_1x1075,
                                       edge_weight_val_1x215,
                                       edge_index_calib_test_2x2, edge_index_train_2x2,
                                       edge_index_val_2x2, calib_test_upper_list, calib_test_lower_list,
                                       count, S=0, T=540, alpha=0.05,
                                       epsilon=0.01,
                                       nodeNum=546):
    # generate dicts for calib_test, train and val
    calib_test_info_dic, train_info_dic, val_info_dic = dictionary_all(edge_weight_calib_test_1x860,
                                                                       edge_weight_train_1x1075,
                                                                       edge_weight_val_1x215,
                                                                       edge_index_calib_test_2x2, edge_index_train_2x2,
                                                                       edge_index_val_2x2, calib_test_upper_list,
                                                                       calib_test_lower_list)

    key_calib_test = calib_test_info_dic.keys()
    key_calib_test_list = list(key_calib_test)
    num = int(len(key_calib_test_list) / 2)

    #################################### create data set ###########################################
    cqr_interval = {}
    cqr_new_interval = {}
    qr_interval = {}
    bl_interval = {}

    c_label_dic = {}

    cov_cqr = {}
    cov_cqr_new = {}
    cov_qr = {}
    cov_bl = {}

    for k in range(0, count):
        shuffle(key_calib_test_list)
        key_calib = key_calib_test_list[: num]  # the first half
        key_test = key_calib_test_list[num:]  # the last half

        # generate cqr, cqr_new upper for calib
        edge_weight_calib_1x430_shuffle = []
        calib_lower_list_shuffle = []
        calib_upper_list_shuffle = []
        edge_weight_test_1x430_shuffle = []
        test_lower_list_shuffle = []
        test_upper_list_shuffle = []
        for key in key_calib:
            edge_weight_calib_1x430_shuffle.append(calib_test_info_dic[key][0])
            calib_upper_list_shuffle.append(calib_test_info_dic[key][1])
            calib_lower_list_shuffle.append(calib_test_info_dic[key][2])
        for key in key_test:
            edge_weight_test_1x430_shuffle.append(calib_test_info_dic[key][0])
            test_upper_list_shuffle.append(calib_test_info_dic[key][1])
            test_lower_list_shuffle.append(calib_test_info_dic[key][2])

        edge_weight_calib_1x430_shuffle_array = np.array(edge_weight_calib_1x430_shuffle)
        calib_lower_list_shuffle_array = np.array(calib_lower_list_shuffle)
        calib_upper_list_shuffle_array = np.array(calib_upper_list_shuffle)
        edge_weight_test_1x430_shuffle_array = np.array(edge_weight_test_1x430_shuffle)
        test_lower_list_shuffle_array = np.array(test_lower_list_shuffle)
        test_upper_list_shuffle_array = np.array(test_upper_list_shuffle)

        prediction_sets_cqr, cov, eff = cqr(edge_weight_calib_1x430_shuffle_array, calib_lower_list_shuffle_array,
                                            calib_upper_list_shuffle_array, edge_weight_test_1x430_shuffle_array,
                                            test_lower_list_shuffle_array, test_upper_list_shuffle_array,
                                            n=len(key_test),
                                            alpha=alpha)

        prediction_sets_cqr_new, cov, eff = cqr_new(edge_weight_calib_1x430_shuffle_array,
                                                    calib_lower_list_shuffle_array,
                                                    calib_upper_list_shuffle_array,
                                                    edge_weight_test_1x430_shuffle_array,
                                                    test_lower_list_shuffle_array, test_upper_list_shuffle_array,
                                                    n=len(key_test),
                                                    alpha=alpha)

        ##############################################gurobi_matrix##########################################################
        key_shuffle = 'shuffle_' + str(k)

        # cqr_new model

        # cqr model

        # qr model

        # baseline model
        baseline_upper = np.quantile(np.hstack([edge_weight_train_1x1075, edge_weight_val_1x215]), q=1 - alpha / 2,
                                     axis=0)
        baseline_mean = np.quantile(np.hstack([edge_weight_train_1x1075, edge_weight_val_1x215]), q=0.5,
                                    axis=0)
        bl_vector_upper = np.full_like(test_upper_list_shuffle_array, baseline_upper)
        bl_vector_mean = np.full_like(test_upper_list_shuffle_array, baseline_mean)

        ###################################### use the 1st and 2nd dimension of prediction_sets ##############################
        key_shuffle = 'shuffle_' + str(k)

        a = np.array(prediction_sets_cqr_new[0]).tolist()
        b = np.array(prediction_sets_cqr_new[1]).tolist()
        c = edge_weight_test_1x430_shuffle_array.tolist()
        cov_cqr_new[key_shuffle] = check_range(a, b, c)
        cqr_new_interval[key_shuffle] = [a, b]

        a = np.array(prediction_sets_cqr[0]).tolist()
        b = np.array(prediction_sets_cqr[1]).tolist()
        c = edge_weight_test_1x430_shuffle_array.tolist()
        cov_cqr[key_shuffle] = check_range(a, b, c)
        cqr_interval[key_shuffle] = [a, b]

        a = test_lower_list_shuffle_array.tolist()
        b = test_upper_list_shuffle_array.tolist()
        c = edge_weight_test_1x430_shuffle_array.tolist()
        cov_qr[key_shuffle] = check_range(a, b, c)
        qr_interval[key_shuffle] = [a, b]

        a = [0 for i in range(0, len(c))]
        b = bl_vector_upper.tolist()
        m = bl_vector_mean.tolist()
        cov_bl[key_shuffle] = check_range(a, b, c)
        bl_interval[key_shuffle] = [m, b]

        c_label_dic[key_shuffle] = c

    return cov_cqr, cov_cqr_new, cov_qr, cov_bl, cqr_interval, cqr_new_interval, qr_interval, bl_interval, c_label_dic


def create_node_arc_incidence_matrix(num_nodes, edges):
    nodes = [str(i) for i in range(0, num_nodes)]
    num_arcs = len(edges)
    matrix = [[0] * num_arcs for _ in range(num_nodes)]

    for i, node in enumerate(nodes):
        for j, edge in enumerate(edges):
            if node == str(edge[0]):
                matrix[i][j] = 1
            elif node == str(edge[1]):
                matrix[i][j] = -1

    matrix = np.array(matrix)
    return matrix


def cal_standard_error(num_list):
    n = len(num_list)
    mean = sum(num_list) / n
    squared_diff_sum = sum((x - mean) ** 2 for x in num_list)
    standard_error = math.sqrt(squared_diff_sum / n)
    return standard_error


def json_dic_to_std_error(FILENAME):
    dic = json.load(open(FILENAME))
    num_list = []
    for i in dic.keys():
        for j in dic[i].keys():
            for k in dic[i][j].keys():
                for l in dic[i][j][k].keys():
                    num_list.append(dic[i][j][k][l])
    std_e = cal_standard_error(num_list)
    return std_e


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, gconv=SAGEConv):
        super().__init__()
        self.conv1 = gconv(in_channels, hidden_channels)
        self.conv2 = gconv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class DirectedGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, gconv=SAGEConv):
        super().__init__()
        self.layers = [in_channels, hidden_channels, out_channels]
        self.num_layers = len(self.layers) - 1
        self.source = torch.nn.ModuleList()
        self.target = torch.nn.ModuleList()
        for n_in, n_out in zip(self.layers[:-1], self.layers[1:]):
            self.source.append(gconv(n_in, n_out))
            self.target.append(gconv(n_in, n_out))

    def forward(self, s, t, edge_index, edge_weight):
        edge_weight = (edge_weight).sigmoid()
        for layer_id, (layer_s, layer_t) in enumerate(zip(self.source, self.target)):
            s_new = layer_s(t, edge_index, edge_weight)
            t_new = layer_t(s, torch.flip(edge_index, [0]), edge_weight)
            if layer_id < self.num_layers - 1:
                s_new = s_new.relu()
                t_new = t_new.relu()
                s_new = F.dropout(s_new, p=0.5, training=self.training)
                t_new = F.dropout(t_new, p=0.5, training=self.training)
            s = s_new
            t = t_new

        return s, t


def cqr_graph(cal_labels, cal_lower, cal_upper, val_labels, val_lower, val_upper, n, alpha):
    cal_scores = np.maximum(cal_labels - cal_upper, cal_lower - cal_labels)
    cal_scores = cal_scores / np.abs(cal_upper - cal_lower)
    qhat = np.quantile(cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, method='higher')
    prediction_sets = [val_lower - qhat * np.abs(val_upper - val_lower),
                       val_upper + qhat * np.abs(val_upper - val_lower)]
    cov = ((val_labels >= prediction_sets[0]) & (val_labels <= prediction_sets[1])).mean()
    eff = np.mean(prediction_sets[1] - prediction_sets[0])
    return prediction_sets, cov, eff


# cqr

def qr(cal_labels, cal_lower, cal_upper, val_labels, val_lower, val_upper, n, alpha):
    prediction_sets = [val_lower, val_upper]
    cov = ((val_labels >= prediction_sets[0]) & (val_labels <= prediction_sets[1])).mean()
    eff = np.mean(val_upper - val_lower)
    return prediction_sets, cov, eff


def worst_slice_coverage(x, edge_index_calib_test, idx, val_labels, prediction_sets):
    if torch.is_tensor(x):
        x = x.detach().numpy()
    if torch.is_tensor(edge_index_calib_test):
        edge_index_calib_test = edge_index_calib_test.detach().numpy()
    xtest = np.hstack([x[edge_index_calib_test[0, ~idx]], x[edge_index_calib_test[1, ~idx]]])
    ntest = xtest.shape[0]
    nfeat = xtest.shape[1]
    xtest_test = xtest[:ntest // 4]
    unitvec = np.random.randn(nfeat, 1000)
    unitvec = unitvec / np.sqrt((unitvec ** 2).sum(axis=0))
    # ab_range = np.quantile((xtest_test @ unitvec).flatten(), np.linspace(0, 1, 11))
    values = (xtest_test @ unitvec).flatten()
    ab_range = np.linspace(values.min(), values.max(), 10)

    ws_cov_min = None
    for delta in np.linspace(0.1, 0.5, 5):
        ws_cov = 1
        ws_a = None
        ws_b = None
        ws_vec = None
        for vec in unitvec.T:
            value_vec = xtest_test @ vec.reshape(-1, 1)
            for a, b in zip(ab_range[:-1], ab_range[1:]):
                contained = np.bitwise_and(value_vec > a, value_vec < b).flatten()
                if contained.mean() > delta:
                    conditional_cov = (
                            (val_labels[:ntest // 4][contained] >= prediction_sets[0][:ntest // 4][contained]) & (
                            val_labels[:ntest // 4][contained] <= prediction_sets[1][:ntest // 4][
                        contained])).mean()
                    if conditional_cov < ws_cov:
                        print(f"Worst-Slice coverage = {conditional_cov:.4f}")
                        ws_cov = conditional_cov
                        ws_a = a
                        ws_b = b
                        ws_vec = vec
        if ws_vec is None:
            return None
        xtest_true = xtest[ntest // 4:]
        value_vec = xtest_true @ ws_vec.reshape(-1, 1)
        contained = np.bitwise_and(value_vec > ws_a, value_vec < ws_b).flatten()
        ws_cov_true = ((val_labels[ntest // 4:][contained] >= prediction_sets[0][ntest // 4:][contained]) & (
                val_labels[ntest // 4:][contained] <= prediction_sets[1][ntest // 4:][contained])).mean()
        if ws_cov_min is not None and ws_cov_true < ws_cov_min:
            ws_cov_min = ws_cov_true
        elif ws_cov_min is None and ~np.isnan(ws_cov_true):
            ws_cov_min = ws_cov_true
    return ws_cov_min


def train_gae_directed(x: object, edge_index_train: object, edge_weight: object, alpha: object = ALPHA,
                       val: object = False, edge_index_val: object = None,
                       sigmoid: object = False) -> object:
    if val:
        model.eval()
    else:
        model.train()
    Z_source, Z_target = model(x, x, edge_tensor, edge_weight_for_training)
    # Z_source, Z_target = model(x, x, edge_tensor, edge_weight_gae_training)
    out_dim = Z_source.shape[-1] // 2
    z_lower_source = Z_source[:, :out_dim];
    z_upper_source = Z_source[:, out_dim:]
    z_lower_target = Z_target[:, :out_dim];
    z_upper_target = Z_target[:, out_dim:]
    # z_lower_source = Z_source[:, out_dim:2 * out_dim];
    # z_upper_source = Z_source[:, 2 * out_dim:]
    # z_lower_target = Z_target[:, out_dim:2 * out_dim];
    # z_upper_target = Z_target[:, 2 * out_dim:]
    if val:
        lower = model.decoder(z_lower_source, z_lower_target, edge_index_val, sigmoid=sigmoid)
        upper = model.decoder(z_upper_source, z_upper_target, edge_index_val, sigmoid=sigmoid)
    else:
        lower = model.decoder(z_lower_source, z_lower_target, edge_index_train, sigmoid=sigmoid)
        upper = model.decoder(z_upper_source, z_upper_target, edge_index_train, sigmoid=sigmoid)

    label = edge_weight
    low_bound = alpha / 2
    upp_bound = 1 - alpha / 2
    low_loss = torch.mean(torch.max((low_bound - 1) * (label - lower), low_bound * (label - lower)))
    upp_loss = torch.mean(torch.max((upp_bound - 1) * (label - upper), upp_bound * (label - upper)))
    loss = low_loss + upp_loss  # + 0.1 * (upper - lower).mean() # mse_loss +

    if not val:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return float(loss)


def test_gae_directed_lower_upper(best_model, x, train_edge_index, calib_test_edge_index, calib_test_edge_weight,
                                  alpha=ALPHA,
                                  return_prediction_sets=False, score='cqr', conditional=True, sigmoid=False):
    best_model = best_model.cpu()
    best_model.eval()
    Z_source, Z_target = best_model(x.cpu(), x.cpu(), edge_tensor.cpu(), edge_weight_for_training.cpu())
    out_dim = Z_source.shape[-1] // 2
    z_lower_source = Z_source[:, :out_dim];
    z_upper_source = Z_source[:, out_dim:]
    z_lower_target = Z_target[:, :out_dim];
    z_upper_target = Z_target[:, out_dim:]

    lower = best_model.decoder(z_lower_source, z_lower_target, calib_test_edge_index.cpu(), sigmoid=sigmoid)
    upper = best_model.decoder(z_upper_source, z_upper_target, calib_test_edge_index.cpu(), sigmoid=sigmoid)

    lower_list = lower.detach().numpy()  # turn tensor data into list
    upper_list = upper.detach().numpy()  # turn tensor data into list

    return lower_list, upper_list


def train_gae_mse(x, edge_index, edge_weight=None, sigmoid=False):
    Z_source, Z_target = model(x, x, edge_tensor, edge_weight_gae_training)
    mean_prediction = model.decoder(Z_source, Z_target, edge_index, sigmoid=sigmoid)
    if edge_weight == None:
        return mean_prediction
    else:
        model.train()
        label = edge_weight
        loss = torch.nn.functional.mse_loss(mean_prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss)



# import data and process
node_rename = {node: id for id, node in enumerate(range(388, 934))}
nodefile = 'ChicagoSketch_node.tntp'
node = pd.read_csv(nodefile, sep='\t', usecols=['node', 'X', 'Y'])
flowfile = 'ChicagoSketch_flow.tntp'
colname = 'Volume '
flow = pd.read_csv(flowfile, sep='\t', usecols=['From ', 'To ', colname])

node['node'] = node['node'].map(node_rename)
node = node[node['node'].notna()]

flow['From '] = flow['From '].map(node_rename)
flow['To '] = flow['To '].map(node_rename)
flow = flow[(flow['From '].notna()) & (flow['To '].notna())]
flow.drop(flow[flow[colname] <= 0].index, inplace=True)
flow[colname] = np.log(flow[colname])

scaler = StandardScaler()
node[['X', 'Y']] = scaler.fit_transform(node[['X', 'Y']].values)

df = flow.rename(columns={'From ': 's', 'To ': 'r', colname: 'w'})
df1 = pd.merge(df, node, how='left', left_on='s', right_on='node')[['s', 'r', 'w', 'X', 'Y']].rename(
    columns={'X': 'X1', 'Y': 'Y1'})
df2 = pd.merge(df1, node, how='left', left_on='r', right_on='node')[['s', 'r', 'w', 'X1', 'Y1', 'X', 'Y']].rename(
    columns={'X': 'X2', 'Y': 'Y2'})
df2['feat'] = df2[['X1', 'Y1', 'X2', 'Y2']].values.tolist()

edge_name_to_y = {(s, r): w for s, r, w in df2[['s', 'r', 'w']].values}
edge_name_to_x = {(s, r): feat for s, r, feat in df2[['s', 'r', 'feat']].values}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
G = nx.from_pandas_edgelist(df2, source='s', target='r', edge_attr='w', create_using=nx.DiGraph())
airport = from_networkx(G)
airport.x = torch.from_numpy(node[['X', 'Y']].values).to(torch.float32)
# print(airport)

G_line_graph = nx.line_graph(G, create_using=nx.DiGraph())
airport_line_graph = from_networkx(G_line_graph)
airport_line_graph.x = torch.from_numpy(np.vstack([edge_name_to_x[e] for e in G_line_graph.nodes])).to(torch.float32)
airport_line_graph.y = torch.from_numpy(np.vstack([edge_name_to_y[e] for e in G_line_graph.nodes])).to(torch.float32)
print(airport_line_graph)


# split the edges into train/val/calib+test
time_start = time.time()  # 开始计时
seed_tuple_1 = np.random.randint(0, 250000, 250000)
seed_tuple_2 = np.random.randint(1000, 20000, 10000)
EDGE_MIN = 30  # minimum edge
NODENUM = 546  # num of nodes

CQR_interval_10x10x10x10 = {}
CQR_NEW_interval_10x10x10x10 = {}
QR_interval_10x10x10x10 = {}
BL_interval_10x10x10x10 = {}

C_LABEL_10x10x10x10 = {}

COV_CQR_10x10x10x10 = {}
COV_CQR_NEW_10x10x10x10 = {}
COV_QR_10x10x10x10 = {}
COV_BL_10x10x10x10 = {}

ST_sets = 1  # number of ST
Seed_sets = 3  # number of seed
COUNT = 1000  # number of shuffle

# Choose the journey which are long enough
ST_lists_min_30 = [[31, 512], [356, 512], [357, 512], [367, 512], [367, 541], [368, 541], [512, 31], [512, 356],
                   [512, 357], [512, 367], [512, 534], [512, 535], [523, 535], [524, 535], [534, 512], [535, 512],
                   [535, 523], [535, 524], [541, 367], [541, 368]]

ST_count = 0
i = 0
# sets of starting points and terminal points
for ST_count in range(0, ST_sets):
    seed_everything(seed_tuple_1[i])
    print('*********************************************** i = ' + str(
        i) + ' ***************************************************')
    i += 1
    START = ST_lists_min_30[ST_count][0]
    TERM = ST_lists_min_30[ST_count][1]
    if G.has_node(START) and G.has_node(TERM):
        if nx.shortest_path_length(G, START, TERM) >= EDGE_MIN:
            print('*********************************************** ST_count = ' + str(
                ST_count) + ' ***************************************************')
        else:
            continue
    else:
        continue
    ST_count += 1
    CQR_interval_10x10x10 = {}
    CQR_NEW_interval_10x10x10 = {}
    QR_interval_10x10x10 = {}
    BL_interval_10x10x10 = {}

    C_LABEL_10x10x10 = {}

    COV_CQR_10x10x10 = {}
    COV_CQR_NEW_10x10x10 = {}
    COV_QR_10x10x10 = {}
    COV_BL_10x10x10 = {}

    for j in range(0, Seed_sets):
        print('************************************************ j = ' + str(
            j) + ' **************************************************')
        seed_name = seed_tuple_2[(ST_count * ST_sets + j)]
        seed_everything(seed_name)
        ############################shift num_test from 0.1 to 0.8 by 0.1########################################
        proportion = [0.8]

        CQR_interval_10x10 = {}
        CQR_NEW_interval_10x10 = {}
        QR_interval_10x10 = {}
        BL_interval_10x10 = {}

        C_LABEL_10x10 = {}

        COV_CQR_10x10 = {}
        COV_CQR_NEW_10x10 = {}
        COV_QR_10x10 = {}
        COV_BL_10x10 = {}

        for p in proportion:
            split = RandomNodeSplit(num_val=0.1, num_test=p)
            data = split(airport_line_graph)
            data = data.to(device)

            edge_array = np.array(list(dict(G_line_graph.nodes).keys()))
            edge_index_train = edge_array[data.train_mask.cpu().numpy()]
            edge_index_val = edge_array[data.val_mask.cpu().numpy()]
            edge_index_calib_test = edge_array[data.test_mask.cpu().numpy()]

            edge_weight_train = torch.Tensor(np.stack([edge_name_to_y[tuple(edge)] for edge in edge_index_train])).to(
                device)
            edge_weight_val = torch.Tensor(np.stack([edge_name_to_y[tuple(edge)] for edge in edge_index_val])).to(
                device)
            edge_weight_calib_test = torch.Tensor(
                np.stack([edge_name_to_y[tuple(edge)] for edge in edge_index_calib_test])).to(
                device)

            edge_index_train = torch.LongTensor(edge_index_train).T.to(device)
            edge_index_val = torch.LongTensor(edge_index_val).T.to(device)
            edge_index_calib_test = torch.LongTensor(edge_index_calib_test).T.to(device)
            edge_tensor = torch.LongTensor(edge_array).T.to(device)

            edge_weight_gae_training = [edge_name_to_y[tuple(edge)] if train else 1 for edge, train in
                                        zip(edge_array, data.train_mask.cpu().numpy())]
            edge_weight_gae_training = torch.Tensor(edge_weight_gae_training).to(
                device)  # torch.ones(edge_array.shape[0]).to(device)

            # w_min, w_max = edge_weight_gae_training.min(), edge_weight_gae_training.max()
            # edge_weight_gae_training = edge_weight_gae_training.sigmoid() # (edge_weight_gae_training - w_min) / (w_max - w_min)  * 0.5 + 0.5

            # delta_ij model
            encoder = DirectedGNN(in_channels=airport.x.shape[-1], hidden_channels=HIDDEN, out_channels=2 * OUT,
                                  gconv=GraphConv)
            decoder = DirectedInnerProductDecoder()  # DirectedEdgeDecoder(hidden_channels=2, out_channels=1)
            model = GAE(encoder, decoder).to(device)
            print(model)
            x = airport.x.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
            use_sigmoid = False
            for epoch in range(1, EPOCHS):
                loss = train_gae_mse(x, edge_index_train, edge_weight_train, sigmoid=use_sigmoid)
                ######################################################################################
                if epoch % 100 == 1:
                    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

            W_hat = train_gae_mse(x, torch.cat([edge_index_val, edge_index_calib_test], dim=1))
            print(W_hat.shape)

            edge_weight_val_cal_test = train_gae_mse(x, torch.cat([edge_index_val, edge_index_calib_test], dim=1))
            edge_index_val_cal_test = torch.cat([edge_index_val, edge_index_calib_test], dim=1)

            edge_weight_for_training = []
            for edge, train in zip(edge_array, data.train_mask.cpu().numpy()):
                if train:
                    edge_weight_for_training.append(edge_name_to_y[tuple(edge)])
                else:
                    idx = (edge_index_val_cal_test.numpy() == edge[:, np.newaxis]).all(axis=0).nonzero()[0][0]
                    edge_weight_for_training.append(edge_weight_val_cal_test[idx].float())

            edge_weight_for_training = torch.Tensor(edge_weight_for_training).to(device)
            # DiGAE
            encoder = DirectedGNN(in_channels=airport.x.shape[-1], hidden_channels=HIDDEN, out_channels=2 * OUT,
                                  gconv=GraphConv)
            decoder = DirectedInnerProductDecoder()  # DirectedEdgeDecoder(hidden_channels=2, out_channels=1)
            model = GAE(encoder, decoder).to(device)
            print(model)
            x = airport.x.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
            best_val_loss = float('inf')
            best_model = None
            use_sigmoid = False
            for epoch in range(1, EPOCHS):
                loss = train_gae_directed(x, edge_index_train, edge_weight_train, sigmoid=use_sigmoid)
                ######################################################################################
                if epoch % 100 == 1:
                    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
                val_loss = train_gae_directed(x, edge_index_train, edge_weight_val, val=True,
                                              edge_index_val=edge_index_val, sigmoid=use_sigmoid)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = copy.deepcopy(model)
                    # print(f'Epoch: {epoch:03d}, Best validation loss: {val_loss:.4f}')

            calib_test_lower_list, calib_test_upper_list = test_gae_directed_lower_upper(
                best_model, x, edge_index_train,
                edge_index_calib_test,
                edge_weight_calib_test)
            # cov_all, eff_all, ws_cov_all, pred_set_all, val_labels_all, idx_all = test_gae_directed(best_model, x, edge_index_train, edge_index_calib_test, edge_weight_calib_test, return_prediction_sets=True, score='cqr_new', sigmoid=use_sigmoid, conditional=True)

            edge_index_train_2x2 = edge_index_train.detach().numpy()
            edge_index_val_2x2 = edge_index_val.detach().numpy()
            edge_index_calib_test_2x2 = edge_index_calib_test.detach().numpy()

            edge_weight_train_1x1075 = edge_weight_train.detach().numpy()
            edge_weight_val_1x215 = edge_weight_val.detach().numpy()
            edge_weight_calib_test_1x860 = edge_weight_calib_test.detach().numpy()




            COV_CQR, COV_CQR_NEW, COV_QR, COV_BL, CQR_interval, CQR_NEW_interval, QR_interval, BL_interval, C_LABEL = obtain_prediction_interval_c_label(
                edge_weight_calib_test_1x860, edge_weight_train_1x1075,
                edge_weight_val_1x215,
                edge_index_calib_test_2x2, edge_index_train_2x2,
                edge_index_val_2x2, calib_test_upper_list, calib_test_lower_list,
                count=COUNT,
                S=START,
                T=TERM,
                alpha=ALPHA,
                nodeNum=NODENUM)
            key = 'p_' + str(p)
            CQR_interval_10x10[key] = CQR_interval
            CQR_NEW_interval_10x10[key] = CQR_NEW_interval
            QR_interval_10x10[key] = QR_interval
            BL_interval_10x10[key] = BL_interval

            C_LABEL_10x10[key] = C_LABEL

            COV_CQR_10x10[key] = COV_CQR
            COV_CQR_NEW_10x10[key] = COV_CQR_NEW
            COV_QR_10x10[key] = COV_QR
            COV_BL_10x10[key] = COV_BL

        key = '_seed_' + str(seed_name)
        CQR_interval_10x10x10[key] = CQR_interval_10x10
        CQR_NEW_interval_10x10x10[key] = CQR_NEW_interval_10x10
        QR_interval_10x10x10[key] = QR_interval_10x10
        BL_interval_10x10x10[key] = BL_interval_10x10

        C_LABEL_10x10x10[key] = C_LABEL_10x10

        COV_CQR_10x10x10[key] = COV_CQR_10x10
        COV_CQR_NEW_10x10x10[key] = COV_CQR_NEW_10x10
        COV_QR_10x10x10[key] = COV_QR_10x10
        COV_BL_10x10x10[key] = COV_BL_10x10

        # print(f"{np.mean(cov_all):.4f}+/-{np.std(cov_all):.4f}, {np.mean(eff_all):.4f}+/-{np.std(eff_all):.4f}")

    key = '(' + str(START) + ',' + str(TERM) + ')'
    CQR_interval_10x10x10x10[key] = CQR_interval_10x10x10
    CQR_NEW_interval_10x10x10x10[key] = CQR_NEW_interval_10x10x10
    QR_interval_10x10x10x10[key] = QR_interval_10x10x10
    BL_interval_10x10x10x10[key] = BL_interval_10x10x10

    C_LABEL_10x10x10x10[key] = C_LABEL_10x10x10

    COV_CQR_10x10x10x10[key] = COV_CQR_10x10x10
    COV_CQR_NEW_10x10x10x10[key] = COV_CQR_NEW_10x10x10
    COV_QR_10x10x10x10[key] = COV_QR_10x10x10
    COV_BL_10x10x10x10[key] = COV_BL_10x10x10

# save intervals
NAME = '\V4_comment_delta\Interval_NO_OPT2\V20_3_1_1000'
dump_files_NO_OPT(NAME, CQR_interval_10x10x10x10, CQR_NEW_interval_10x10x10x10, QR_interval_10x10x10x10,
                  BL_interval_10x10x10x10,
                  C_LABEL_10x10x10x10,
                  COV_CQR_10x10x10x10,
                  COV_CQR_NEW_10x10x10x10,
                  COV_QR_10x10x10x10, COV_BL_10x10x10x10)

time_end = time.time()  # 结束计时
time_c = time_end - time_start  # 运行所花时间
print('time cost', time_c, 's')

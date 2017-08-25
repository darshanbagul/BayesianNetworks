import xl_read
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm
import math
from math import log10, floor
import pandas as pd

def round_matrix(mat):
    x = []
    for i in mat:
        y = []
        for j in i:
            y.append(round_sig(j, 3))
        x.append(y)
    return np.vstack(x)

def round_sig(x, sig=2):
    if x > 0:
        return round(x, sig-int(floor(log10(x)))-1)
    else:
        y = -1 * x
        return -1 * round(y, sig-int(floor(log10(y)))-1)

def calc_mean(array):
    return np.mean(array)


def calc_variance(array):
    return np.var(array)

def calc_std(array):
    return np.std(array)

def calc_values_column(col_index, num_rows, xl_sheet):
    score_val = []
    for row in range(1, num_rows):
        if isinstance(xl_sheet.cell_value(row, col_index), float):
            # print xl_sheet.cell_value(row, col_index)
            score_val.append(float(xl_sheet.cell_value(row, col_index)))

    avg = calc_mean(score_val)
    var = calc_variance(score_val)
    std = calc_std(score_val)
    return avg, var, std, score_val

def plot_vector_pairs_scatter(x1, x2, x_label, y_label, title):
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.scatter(x1,x2)
    plt.show()

def calc_independent_loglikelihood_var(variable, avg, std):
    return np.sum(norm.logpdf(variable, avg, std))


def calc_cond_prob_one_var(y, x):
    sq_temp = np.dot(x, np.vstack(x))
    l1 = [len(x) , np.sum(x)]
    l2 = [np.sum(x), sq_temp]
    A = np.vstack([l1, l2])
    y_mod = [np.sum(y), np.dot(x,y)]

    A_inv = np.linalg.inv(A)
    B  = np.dot(A_inv, y_mod)

    var_temp = 0
    for i in range(len(y)):
        var_temp += np.square((B[0] + (B[1] * x[i])) - y[i])

    var = var_temp/len(y)
    sigma = np.sqrt(var)
    log_likelihood = 0
    for i in range(len(y)):
        log_likelihood += norm.logpdf(y[i], B[0]+B[1]*x[i], sigma)
    return log_likelihood

def calc_cond_prob_two_var(y, x1, x2):
    l1 = [len(x1) , np.sum(x1) , np.sum(x2)]
    l2 = [np.sum(x1), np.sum(np.multiply(x1,x1)), np.dot(x1, np.vstack(x2))]
    l3 = [np.sum(x2), np.dot(x1, np.vstack(x2)), np.sum(np.multiply(x2,x2))]
    A = np.vstack([l1, l2, l3])
    y_mod = [np.sum(y), np.dot(x1,y), np.dot(x2,y)]

    A_inv = np.linalg.inv(A)
    B  = np.dot(A_inv, y_mod)

    var_temp = 0
    for i in range(len(y)):
        var_temp += np.square((B[0] + (B[1] * x1[i]) + (B[2] * x2[i])) - y[i])

    var = var_temp/len(y)
    sigma = np.sqrt(var)
    log_likelihood = 0
    for i in range(len(y)):
        log_likelihood += norm.logpdf(y[i], B[0]+B[1]*x1[i]+B[2]*x2[i], sigma)
    return log_likelihood

def calc_cond_prob_three_var(y, x1, x2, x3):
    l1 = [len(x1) , np.sum(x1) , np.sum(x2), np.sum(x3)]
    l2 = [np.sum(x1), np.sum(np.multiply(x1,x1)), np.dot(x1, np.vstack(x2)), np.dot(x1, np.vstack(x3))]
    l3 = [np.sum(x2), np.dot(x2, np.vstack(x1)), np.sum(np.multiply(x2,x2)), np.dot(x2, np.vstack(x3))]
    l4 = [np.sum(x3), np.dot(x3, np.vstack(x1)), np.dot(x3, np.vstack(x2)), np.sum(np.multiply(x3,x3))]
    
    A = np.vstack([l1, l2, l3, l4])
    y_mod = [np.sum(y), np.dot(x1,y), np.dot(x2,y), np.dot(x3,y)]

    A_inv = np.linalg.inv(A)
    B  = np.dot(A_inv, y_mod)

    var_temp = 0
    for i in range(len(y)):
        var_temp += np.square((B[0] + (B[1] * x1[i]) + (B[2] * x2[i]) + (B[3] * x3[i])) - y[i])

    var = var_temp/len(y)
    sigma = np.sqrt(var)
    log_likelihood = 0
    for i in range(len(y)):
        log_likelihood += norm.logpdf(y[i], B[0]+ B[1]*x1[i] + B[2]*x2[i] + B[3]*x3[i], sigma)
    return log_likelihood
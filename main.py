import xlrd
import numpy as np
# import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm
import math
from math import log10, floor

fname = 'university data.xlsx'

xl_workbook = xlrd.open_workbook(fname)
xl_sheet = xl_workbook.sheet_by_index(0)

num_cols = xl_sheet.ncols
num_rows = xl_sheet.nrows

def round_matrix(mat):
    ''' 
        Function to round the elements to 3 signifant digits.
        Input: Matrix
        Output : Matrix
    '''
    x = []
    for i in mat:
        y = []
        for j in i:
            y.append(round_sig(j, 3))
        x.append(y)
    return np.vstack(x)

def round_sig(x, sig=2):
    ''' 
        Function to round a given floating point number to 3 signifant digits 
    '''
    if x > 0:
        return round(x, sig-int(floor(log10(x)))-1)
    else:
        y = -1 * x
        return -1 * round(y, sig-int(floor(log10(y)))-1)

def calc_mean(array):
    ''' 
        Function for calculating the mean of a given array or vector. 
    '''
    return np.mean(array)


def calc_variance(array):
    ''' 
        Function for calculating the variance of a given array or vector. 
    '''
    return np.var(array)

def calc_std(array):
    ''' 
        Function for calculating the standard deviation of a given array or vector. 
    '''
    return np.std(array)

def calc_values_column(col_index):
    ''' 
        Function for reading a column given by col_index from given excel sheet. 
        The function reads the column in a vector and issues calls to functions for 
        calculating the mean, variance and standard deviation of the column. 
    '''
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
    ''' 
        Function to plot the scatter plot between two given vectors x1 and x2.
    '''
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.scatter(x1,x2)
    plt.show()

def plot_matrix_rep(matrix, title, axis_labels):
    '''
        Function to plot m*n matrix data using matplotlib function matshow.
        Helps in visualising different values in given matrix.
    ''' 
    fig = plt.figure()
    plt.title(title)
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, interpolation='nearest')
    fig.colorbar(cax)

    ax.set_xticklabels(['']+axis_labels)
    ax.set_yticklabels(['']+axis_labels)
    plt.show()

def plot_covariance_matrix(cov_X, title, axis_labels):
    '''
        Function to plot the covariance matrix.
        Seperate function created to handle log of negative values.
    '''
    plot_values = []
    for i in cov_X:
        row = []
        for j in i:
            if j>0:
                row.append(np.log(j))
            else:
                row.append(-1 * np.log(-j))
        plot_values.append(row)
    plot_values = np.vstack(plot_values)
    plot_matrix_rep(plot_values, title, axis_labels)

def calc_independent_loglikelihood_var(variable, avg, std):
    ''' 
        Function to calculate the log likelihood of a given random variable using the 
        mean and standard deviation of the random variable. Assuming the random variable 
        is normally distributed.
    ''' 
    return np.sum(norm.logpdf(variable, avg, std))


def calc_cond_prob_one_var(y, x):
    '''
        Function to calculate conditional probability between two variables y and x => P(Y|X)
    '''
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
    '''
        Function to calculate conditional probability between three variables y and x1,x2 => P(Y|X1, X2)
        Here Y is a child node dependent on two parents x1 and x2.
    '''
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
    '''
        Function to calculate conditional probability between four variables y and x1,x2,x3 => P(Y|X1, X2, X3)
        Here Y is a child node dependent on three parent variables x1, x2, x3.
    '''
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

def main():
    ''' Main function for the assignment. '''
    cs_score_ind = 2
    res_overhead_ind = 3
    base_pay_ind = 4
    tuition_ind = 5

    ''' 
        Read Provided Excel File using the function "calc_values_column" and passing the column index of respective random variables.
        The Function returns mean, variance, standard deviation and the random variable vector. 
    '''

    cs_score_avg, cs_score_var, cs_score_std, cs_score_vec = calc_values_column(cs_score_ind)
    res_overhead_avg, res_overhead_var, res_overhead_std, res_overhead_vec = calc_values_column(res_overhead_ind)
    base_pay_avg, base_pay_var, base_pay_std, base_pay_vec = calc_values_column(base_pay_ind)
    tuition_avg, tuition_var, tuition_std, tuition_vec = calc_values_column(tuition_ind)

    print "UBitName = dbagul"
    print "personNumber = 50208043"

    print "mu1 = ", round_sig(cs_score_avg, 3)
    print "mu2 = ", round_sig(res_overhead_avg, 3)
    print "mu3 = ", round_sig(base_pay_avg, 3)
    print "mu4 = ", round_sig(tuition_avg, 3)

    print "var1 = ", round_sig(cs_score_var, 3)
    print "var2 = ", round_sig(res_overhead_var, 3)
    print "var3 = ", round_sig(base_pay_var, 3)
    print "var4 = ", round_sig(tuition_var, 3)

    print "sigma1 = ", round_sig(cs_score_std, 3)
    print "sigma2 = ", round_sig(res_overhead_std, 3)
    print "sigma3 = ", round_sig(base_pay_std, 3)
    print "sigma4 = ", round_sig(tuition_std, 3)

    ''' 
        Pairwise plotting of various random variables, a metric to represent the correlation between the variables 
    '''

    # plot_vector_pairs_scatter(cs_score_vec, base_pay_vec, "CS Score", "Administrator Base Salary", "CS Score vs. Administrator Base Salary")
    # plot_vector_pairs_scatter(cs_score_vec, res_overhead_vec, "CS Score", "Research Overhead", "CS Score vs. Research Overhead")
    # plot_vector_pairs_scatter(cs_score_vec, tuition_vec, "CS Score", "Tuition Fee", "CS Score vs. Tuition Fee")
    # plot_vector_pairs_scatter(res_overhead_vec, base_pay_vec, "Research Overhead", "Administrator Base Salary", "Research Overhead vs. Administrator Base Salary")
    # plot_vector_pairs_scatter(res_overhead_vec, tuition_vec, "Research Overhead", "Tuition Fee", "Research Overhead vs. Tuition Fee")
    # plot_vector_pairs_scatter(base_pay_vec, tuition_vec, "Administrator Base Salary", "Tuition Fee", "Administrator Base Salary vs. Tuition Fee")

    ''' 
        Represent the variables as a matrix, for calculating the correlation and covariance matrices of the data. 
    '''
    X = np.vstack([cs_score_vec, res_overhead_vec, base_pay_vec, tuition_vec]) 

    cov_X = np.cov(X)
    print "covarianceMat = \n", round_matrix(cov_X)


    corr_X = np.corrcoef(X)
    print "correlationMat = \n", round_matrix(corr_X)

    alpha = ['CS_Score', 'Res_Over', 'Base_Pay', 'Tuition']

    ''' 
        Represent the covariance and correlation matrices graphically to distinctly visualize the correlation. 
    '''
    # plot_matrix_rep(corr_X, 'Correlation Matrix Representation', alpha)
    # plot_covariance_matrix(cov_X, 'Covariance Matrix Representation (in log)', alpha)

    ''' 
        Calculate the probability/ log likelihood of each random variable assuming all are independent. 
    ''' 
    cs_score_ind_likelihood = calc_independent_loglikelihood_var(cs_score_vec, cs_score_avg, cs_score_std)
    base_pay_ind_likelihood = calc_independent_loglikelihood_var(base_pay_vec, base_pay_avg, base_pay_std)
    res_overhead_ind_likelihood = calc_independent_loglikelihood_var(res_overhead_vec, res_overhead_avg, res_overhead_std)
    tuition_ind_likelihood = calc_independent_loglikelihood_var(tuition_vec, tuition_avg, tuition_std)

    ''' 
        Given the variables are independent, the log likelihood of the data is just the sum of log likelihoods of 4 random variables. 
    '''
    data_likelihood = cs_score_ind_likelihood + base_pay_ind_likelihood + res_overhead_ind_likelihood + tuition_ind_likelihood
    print "logLikelihood = ", round(data_likelihood, 3)

    ''' 
        Experimenting here, by calculating various conditional probabilities assuming a variable has 1, 2 or 3 parents. 
        This helps in constructing the necessary Bayesian Network.

        Functions and their represenations:

        calc_cond_prob_one_var   => P(Y|X1)
        calc_cond_prob_two_var   => P(Y|X1,X2)
        calc_cond_prob_three_var => P(Y|X1,X2,X3)
    '''
    cs_score_base_pay_cond_prob = calc_cond_prob_one_var(cs_score_vec, base_pay_vec)
    cs_score_res_overhead_cond_prob = calc_cond_prob_one_var(cs_score_vec, res_overhead_vec)
    cs_score_tuition_cond_prob = calc_cond_prob_one_var(cs_score_vec, tuition_vec)
    
    cs_score_base_pay_research_cond_prob = calc_cond_prob_two_var(cs_score_vec, base_pay_vec, res_overhead_vec)
    cs_score_research_tuition_cond_prob = calc_cond_prob_two_var(cs_score_vec, res_overhead_vec, tuition_vec)
    cs_score_base_pay_tuition_cond_prob = calc_cond_prob_two_var(cs_score_vec, base_pay_vec, tuition_vec)

    all_dependent_cs_score = calc_cond_prob_three_var(cs_score_vec, base_pay_vec, res_overhead_vec, tuition_vec)

    
    base_pay_research_cond_prob = calc_cond_prob_one_var(base_pay_vec, res_overhead_vec)
    base_pay_tuition_cond_prob = calc_cond_prob_one_var(base_pay_vec, tuition_vec)
    base_pay_cs_score_cond_prob = calc_cond_prob_one_var(base_pay_vec, cs_score_vec)
    
    base_pay_research_tuition_cond_prob = calc_cond_prob_two_var(base_pay_vec, res_overhead_vec, tuition_vec)

    
    res_overhead_base_pay_cond_prob = calc_cond_prob_one_var(res_overhead_vec, base_pay_vec)
    res_overhead_tuition_cond_prob = calc_cond_prob_one_var(res_overhead_vec, tuition_vec)
    res_overhead_cs_score_cond_prob = calc_cond_prob_one_var(res_overhead_vec, cs_score_vec)

    res_overhead_base_pay_tuition_cond_prob = calc_cond_prob_two_var(res_overhead_vec, base_pay_vec, tuition_vec)
    
    
    tuition_base_pay_cond_prob = calc_cond_prob_one_var(tuition_vec, base_pay_vec)
    tuition_res_overhead_cond_prob = calc_cond_prob_one_var(tuition_vec, res_overhead_vec)
    tuition_cs_score_cond_prob = calc_cond_prob_one_var(tuition_vec, cs_score_vec)
    
    tuition_res_overhead_base_pay_cond_prob = calc_cond_prob_two_var(tuition_vec, res_overhead_vec, base_pay_vec)

    ''' 
        Constructed the optimal Bayesian Network which has better log likelihood than the log likelihood of the data calculated previously.
        (By experimenting with different possible Bayesian Networks)
    '''

    print "BNgraph = "

    cs_score_outbound_edges = [0,0,0,0]
    base_pay_outbound_edges = [1,0,0,0]
    res_overhead_outbound_edges = [1,1,0,0]
    tuition_outbound_edges = [1,1,1,0]

    BN_GRAPH = np.vstack([cs_score_outbound_edges, base_pay_outbound_edges, res_overhead_outbound_edges, tuition_outbound_edges])
    print BN_GRAPH

    BN_GRAPH_likelihood = all_dependent_cs_score + base_pay_research_tuition_cond_prob + res_overhead_tuition_cond_prob + tuition_ind_likelihood
    print "BNlogLikelihood = ", round(BN_GRAPH_likelihood, 3)

main()
'''
Metric functions to assess how well models meet the information needs of users.

Author: Liam Eloie
'''

import numpy as np

def cg(query_relevances, rank):
    '''
    Calculates the cumulative gain at a particular rank position.

    Args:
        query_relevances (list): A list of relevancies e.g. [0, 1, 2, 2, 3, 1].
        rank (int): The cut-off rank.

    Returns:
        int: The cumulative gain.
    '''
    return sum(query_relevances[:rank])

def dcg(query_relevances, rank, method):
    '''
    Calculates the discounted cumulative gain at a particular rank position.

    Args:
        query_relevances (list): A list of relevancies e.g. [0, 1, 2, 2, 3, 1].
        rank (int): The cut-off rank.

    Returns:
        float: The discounted cumulative gain.
    '''
    query_relevances = np.array(query_relevances, dtype='float')[:rank]
    

    if(method == 'exp'):
        discounted = (2**query_relevances - 1) / np.log2(np.array([i for i in range(2, len(query_relevances) + 2)]))

    elif(method == 'linear'):
        discounted = query_relevances / np.log2(np.array([i for i in range(2, len(query_relevances) + 2)]))
        
    return np.sum(discounted)

def idcg(query_relevances, rank, method):
    '''
    Calculates the ideal discounted cumulative gain at a particular rank position.

    Args:
        query_relevances (list): A list of relevancies e.g. [0, 1, 2, 2, 3, 1].
        rank (int): The cut-off rank.

    Returns:
        float: The ideal discounted cumulative gain.
    '''
    return dcg(sorted(query_relevances, reverse=True), rank, method)

def ndcg(query_relevances, rank, method='exp'):
    '''
    Calculates the normalised discounted cumulative gain at a particular rank position.

    Args:
        query_relevances (list): A list of relevancies e.g. [0, 1, 2, 2, 3, 1].
        rank (float): The cut-off rank.

    Returns:
        float: The normalised discounted cumulative gain. Between 0 and 1.
    '''
    dcg_value, idcg_value = dcg(query_relevances, rank, method), idcg(query_relevances, rank, method)
    
    if(dcg_value == 0 or idcg_value == 0):
        return 0

    else:
        return dcg_value / idcg_value

def precision(query_relevances, rank, rel_con=4):
    '''
    Calculates the precision at a particular rank position.

    Args:
        query_relevances (list): A list of relevancies e.g. [0, 1, 2, 2, 3, 1].
        rank (float): The cut-off rank.
        rel_con (int): The relevance integer that is considered 'relevant'.

    Returns:
        float: The precision - fraction of relevant documents retrieved from a query.
    '''
    query_relevances = query_relevances[:rank]
    num_rel = len([1 for i in query_relevances if i >= rel_con])
    num_retrieved = len(query_relevances)

    try:
        return float(num_rel) / float(num_retrieved)
    except ZeroDivisionError:
        return 0

def average_precision(query_relevances, rel_con=4):
    '''
    Calculates the average precision.

    Args:
        query_relevances (list): A list of relevancies e.g. [0, 1, 2, 2, 3, 1].
        rel_con (int): The relevance integer that is considered 'relevant'.

    Returns:
        float: The average precision.
    '''
    query_relevances = np.array(query_relevances)
    r = query_relevances >= rel_con

    try:
        avg_precision = np.mean([precision(query_relevances, rank + 1, rel_con) for rank in range(r.size) if r[rank]])

    except ZeroDivisionError:
        avg_precision = 0.0

    if np.isnan(avg_precision):
        return 0.0

    return avg_precision

def mean_average_precision(queries, rel_con):
    '''
    Calculates the mean average precision.

    Args:
        queries(list): A list of relevances for multiple queries e.g. [[0, 1, 2, 2, 3, 1], [1, 2, 0, 0, 1, 4, 3]].
        rel_con (int): The relevance integer that is considered 'relevant'.

    Returns:
        float: The mean average precision.
    '''
    return np.mean([average_precision(query_relevances, rel_con) for query_relevances in queries])
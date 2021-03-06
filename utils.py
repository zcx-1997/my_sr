# -*- coding: utf-8 -*-

"""
@Time : 2021/10/19
@Author : Lenovo
@File : utils
@Description : 
"""
import time
import numpy as np

from matplotlib import pyplot as plt
from collections import namedtuple
from typing import List, Tuple, Union

# 计时器
class Timer:
    """记录多次运行时间。"""
    def __init__(self):
        self.times = []
        # self.start()

    def start(self):
        """启动计时器。"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中。"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间。"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和。"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间。"""
        return np.array(self.times).cumsum().tolist()


# metrics
def test_eer(scores,labels):
    target_scores = []
    nontarget_scores = []
    for i in range(len(labels)):
        if labels[i] == 1:
            target_scores.append(scores[i])
        else:
            nontarget_scores.append(scores[i])
    # 排序,从小到大排序
    target_scores = sorted(target_scores)
    nontarget_scores = sorted(nontarget_scores)

    target_size = len(target_scores)
    target_position = 0
    for target_position in range(target_size):
        nontarget_size = len(nontarget_scores)
        nontarget_n = nontarget_size * target_position * 1.0 / target_size
        nontarget_position = int(nontarget_size - 1 - nontarget_n)
        if nontarget_position < 0:
            nontarget_position = 0
        if nontarget_scores[nontarget_position] < target_scores[target_position]:
            break
    threshold = target_scores[target_position]
    # print("threshold is --> ", threshold)
    eer = target_position * 1.0 / target_size
    # print("eer is --> ", eer)
    return threshold, eer


def compute_eer(scores: Union[np.ndarray, List[float]],
                labels: Union[np.ndarray, List[int]]) -> Tuple[
                    float, float, np.ndarray, np.ndarray]:
    """Compute equal error rate(EER) given matching scores and corresponding labels
    Parameters:
        scores(np.ndarray,list): the cosine similarity between two speaker embeddings.
        labels(np.ndarray,list): the labels of the speaker pairs, with value 1 indicates same speaker and 0 otherwise.
    Returns:
        eer(float):  the equal error rate.
        thresh_for_eer(float): the thresh value at which false acceptance rate equals to false rejection rate.
        fr_rate(np.ndarray): the false rejection rate as a function of increasing thresholds.
        fa_rate(np.ndarray): the false acceptance rate as a function of increasing thresholds.
     """
    if isinstance(labels, list):
        labels = np.array(labels)
    if isinstance(scores, list):
        scores = np.array(scores)
    label_set = list(np.unique(labels))
    assert len(
        label_set
    ) == 2, f'the input labels must contains both two labels, but recieved set(labels) = {label_set}'
    label_set.sort()
    assert label_set == [
        0, 1
    ], 'the input labels must contain 0 and 1 for distinct and identical id. '
    eps = 1e-8
    #assert np.min(scores) >= -1.0 - eps and np.max(
    #    scores
    #  ) < 1.0 + eps, 'the score must be in the range between -1.0 and 1.0'
    same_id_scores = scores[labels == 1]
    diff_id_scores = scores[labels == 0]
    thresh = np.linspace(np.min(diff_id_scores), np.max(same_id_scores), 1000)
    thresh = np.expand_dims(thresh, 1)
    fr_matrix = same_id_scores < thresh
    fa_matrix = diff_id_scores >= thresh
    fr_rate = np.mean(fr_matrix, 1)
    fa_rate = np.mean(fa_matrix, 1)

    thresh_idx = np.argmin(np.abs(fa_rate - fr_rate))
    result = namedtuple('speaker', ('eer', 'thresh', 'fa', 'fr'))
    result.eer = (fr_rate[thresh_idx] + fa_rate[thresh_idx]) / 2
    result.thresh = thresh[thresh_idx, 0]
    result.fr = fr_rate
    result.fa = fa_rate

    return result


def compute_min_dcf(fr_rate, fa_rate, p_target=0.05, c_miss=1.0, c_fa=1.0):
    """ Compute normalized minimum detection cost function (minDCF) given
        the costs for false accepts and false rejects as well as a priori
        probability for target speakers
    Parameters:
        fr_rate(np.ndarray): the false rejection rate as a function of increasing thresholds.
        fa_rate(np.ndarray): the false acceptance rate as a function of increasing thresholds.
        p_target(float): the prior probability of being a target.
        c_miss(float): cost of miss detection(false rejects).
        c_fa(float): cost of miss detection(false accepts).
    Returns:
        min_cdf(float): the normalized minimum detection cost function (minDCF)
     """

    dcf = c_miss * fr_rate * p_target + c_fa * fa_rate * (1 - p_target)
    c_det = np.min(dcf)
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = c_det / c_def
    return min_dcf

def plot_DET(fr_rates, fa_rates):
    plt.plot(fr_rates,fa_rates)
    plt.show()
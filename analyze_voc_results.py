# ---
# PANDAS
# Copyright (C) 2023 NAVER Corp.
# CC BY-NC-SA 4.0 license
# ---

import os
import numpy as np


def compute_averaged_results(metric, full_file, clusters, seeds):
    groups = [' Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
              ' Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]',
              ' Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]',
              ' Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
              ' Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
              ' Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]']

    d_list = []
    for s in seeds:
        if clusters is None:
            full_file_d = full_file
        else:
            full_file_d = full_file % (clusters, s)
        with open(full_file_d) as f:
            lines = f.readlines()

        cat_lines = []
        for i in lines:
            if 'categories' in i:
                if 'all' in i or 'base' in i or 'novel' in i:
                    cat_lines.append(i)

        cat_scores = {}
        i = 0
        for group in groups:
            cats = cat_lines[i:i + 3]
            cats_new = []
            for c in cats:
                c_new = float(c.split('\n')[-2].split(': ')[-1])
                cats_new.append(c_new)
            cat_scores[group] = cats_new
            i += 3
        d_list.append(cat_scores[metric])

    scores_array = np.array(d_list)
    mu = np.mean(scores_array, axis=0)
    sigma = np.std(scores_array, axis=0)
    print('Base=%0.1f, Novel=%0.1f, All=%0.1f' % (mu[1] * 100, mu[2] * 100, mu[0] * 100))
    print('Base=%0.3f, Novel=%0.3f, All=%0.3f' % (sigma[1] * 100, sigma[2] * 100, sigma[0] * 100))


def main_results(root_to_logs):
    seeds = [42, 43, 44]
    metric = ' Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]'

    ###### PANDAS
    file = 'voc_discovery_10_10_clusters_%d_pandas_seed_%d.log'
    cluster_nums = [10, 20, 50, 100, 250, 500, 1000]
    full_file = os.path.join(root_to_logs, file)

    for clusters in cluster_nums:
        print('\nPANDAS')
        print('Clusters: ', clusters)
        compute_averaged_results(metric, full_file, clusters, seeds)


def additional_studies(root_to_logs):
    metric = ' Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]'
    cluster_nums = [250]

    ###### GT PROTOTYPES
    file = 'voc_discovery_10_10_gt_prototypes.log'
    seeds = [0]
    full_file = os.path.join(root_to_logs, file)

    print('\nGT_PROTOTYPES')
    compute_averaged_results(metric, full_file, None, seeds)

    ###### EVERYTHING ELSE USES THESE SEEDS
    seeds = [42, 43, 44]
    
    ###### INVERT_SQUARE_SOFTMAX_BACKGROUND
    file = 'voc_discovery_10_10_clusters_%d_pandas_invert_square_softmax_BG_seed_%d_additional_study.log'
    full_file = os.path.join(root_to_logs, file)

    for clusters in cluster_nums:
        print('\nInvert Square, Softmax, Background Cls')
        print('Clusters: ', clusters)
        compute_averaged_results(metric, full_file, clusters, seeds)

    ###### INVERT_L1_BACKGROUND
    file = 'voc_discovery_10_10_clusters_%d_pandas_invert_l1_BG_seed_%d_additional_study.log'
    full_file = os.path.join(root_to_logs, file)

    for clusters in cluster_nums:
        print('\nInvert, L1, Background Cls')
        print('Clusters: ', clusters)
        compute_averaged_results(metric, full_file, clusters, seeds)

    ###### INVERT_SOFTMAX_BACKGROUND
    file = 'voc_discovery_10_10_clusters_%d_pandas_invert_softmax_BG_seed_%d_additional_study.log'
    full_file = os.path.join(root_to_logs, file)

    for clusters in cluster_nums:
        print('\nInvert, Softmax, Background Cls')
        print('Clusters: ', clusters)
        compute_averaged_results(metric, full_file, clusters, seeds)

    ###### DOT_PROD_L1_BACKGROUND
    file = 'voc_discovery_10_10_clusters_%d_pandas_dot_prod_l1_BG_seed_%d_additional_study.log'
    full_file = os.path.join(root_to_logs, file)

    for clusters in cluster_nums:
        print('\nDot Prod, L1, Background Cls')
        print('Clusters: ', clusters)
        compute_averaged_results(metric, full_file, clusters, seeds)

    ###### DOT_PROD_SOFTMAX_BACKGROUND
    file = 'voc_discovery_10_10_clusters_%d_pandas_dot_prod_softmax_BG_seed_%d_additional_study.log'
    full_file = os.path.join(root_to_logs, file)

    for clusters in cluster_nums:
        print('\nDot Prod, Softmax, Background Cls')
        print('Clusters: ', clusters)
        compute_averaged_results(metric, full_file, clusters, seeds)

    ###### COSINE_L1_BACKGROUND
    file = 'voc_discovery_10_10_clusters_%d_pandas_cosine_l1_BG_seed_%d_additional_study.log'
    full_file = os.path.join(root_to_logs, file)

    for clusters in cluster_nums:
        print('\nCosine, L1, Background')
        print('Clusters: ', clusters)
        compute_averaged_results(metric, full_file, clusters, seeds)
        
    ###### COSINE_SOFTMAX_BACKGROUND
    file = 'voc_discovery_10_10_clusters_%d_pandas_cosine_softmax_BG_seed_%d_additional_study.log'
    full_file = os.path.join(root_to_logs, file)

    for clusters in cluster_nums:
        print('\nCosine, Softmax, Background Cls')
        print('Clusters: ', clusters)
        compute_averaged_results(metric, full_file, clusters, seeds)

    ###### NO_BACKGROUND
    file = 'voc_discovery_10_10_clusters_%d_pandas_invert_square_l1_No_BG_seed_%d_additional_study.log'
    full_file = os.path.join(root_to_logs, file)

    for clusters in cluster_nums:
        print('\nInvert Square, L1, No Background Cls')
        print('Clusters: ', clusters)
        compute_averaged_results(metric, full_file, clusters, seeds)
        
    ###### ALL_CLUSTERS
    file = 'voc_discovery_10_10_clusters_%d_all_clusters_seed_%d.log'
    cluster_nums = [20, 30, 60, 110, 260, 510, 1010]
    full_file = os.path.join(root_to_logs, file)

    for clusters in cluster_nums:
        print('\nAll Clusters')
        print('Clusters: ', clusters)
        compute_averaged_results(metric, full_file, clusters, seeds)


if __name__ == '__main__':
    root_to_logs = '/pandas_experiments/logs'
    main_results(root_to_logs)
    additional_studies(root_to_logs)

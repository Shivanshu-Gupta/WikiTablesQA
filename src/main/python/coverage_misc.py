# MISCELLANEOUS SCRIPTS
import numpy as np

def get_epoch_dev_accs(train_log):
    epoch_accs = np.array([float(line.strip().split('=')[1]) * 100 for line in open(train_log) if line.find('dev accuracy') != -1])
    return epoch_accs

def get_dev_acc(dev_error_log):
    dev_acc = open(dev_error_log).readlines()[-3].split(':')[1].split(' ')[1]
    return float(dev_acc) * 100

def get_official_result(official_result):
    result = open(official_result).readlines()[-1].split(':')[1]
    return float(result) * 100

# for optimizer hyper-parameter search
# read dev accuracies of each epoch and get best
def print_sgd_results(model):
    optim = 'sgd'
    e0s = [0.05, 0.1, 0.2]
    eds = [0.01, 0.02, 0.05, 0.1, 0.15]
    dev_accs = {e0: {} for e0 in e0s}
    results = {e0: {} for e0 in e0s}
    for e0 in e0s:
        for ed in eds:
            root = 'scratch/experiments/coverage/00/fold1/{}/{}{}{}/'.format(model, optim, e0, ed)
            epoch_accs = get_epoch_dev_accs(root + 'train_log.txt')
            dev_accs[e0][ed] = (get_dev_acc(root + 'dev_error_log.txt'), np.argmax(epoch_accs), len(epoch_accs))
            results[e0][ed] = get_official_result(root + 'official_results.txt')
    # print tables
    for e0 in e0s:
        print(','.join(['{:5.4} ({}/{})'.format(*dev_accs[e0][ed]) for ed in eds]))
    for e0 in e0s:
            print(','.join(['{:5.4}'.format(results[e0][ed]) for ed in eds]))

def print_adam_results(model):
    optim = 'adam'
    e0s = [0.0001, 0.00025, 0.0005, 0.001, 0.002, 0.004]
    dev_accs = {}
    results = {}
    for e0 in e0s:
        root = 'scratch/experiments/coverage/00/fold1/{}/{}{}/'.format(model, optim, e0)
        epoch_accs = get_epoch_dev_accs(root + 'train_log.txt')
        dev_accs[e0] = (get_dev_acc(root + 'dev_error_log.txt'), np.argmax(epoch_accs), len(epoch_accs))
        results[e0] = get_official_result(root + 'official_results.txt')
    # print tables
    print(','.join(['{:5.4} ({}/{})'.format(*dev_accs[e0]) for e0 in e0s]))
    print(','.join(['{:5.4}'.format(results[e0]) for e0 in e0s]))

model = 'coverage'
print_sgd_results(model)
print_adam_results(model)


# analysis

# creates a map from ex_id to true/false
def get_correct_map(official_tsv):
    correct = {}
    for line in open(official_tsv):
        parts = line.split('\t')
        correct[int(parts[0].split('-')[1])] = parts[1] == 'True'
    return correct

def get_confusion_matrix(official_tsv0, official_tsv1):
    import numpy as np
    correct0 = get_correct_map(official_tsv0)
    correct1 = get_correct_map(official_tsv1)
    change = [[[], []], [[], []]]
    cf = np.zeros([2, 2])
    for exid in correct0:
        res0 = correct0[exid]
        res1 = correct1[exid]
        change[res0][res1].append(exid)
        cf[int(res0), int(res1)] += 1
    
    return change, cf


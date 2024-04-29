import argparse
import yaml
import torch
import time
import numpy as np
import os
import datetime
import csv
from collections import defaultdict, OrderedDict
from src.model_handler import ModelHandler

################################################################################
# Main #
################################################################################


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# modified
def main(config):
    configs = grid(config)  # 生成所有参数组合
    results = defaultdict(list)
    for cnf in configs:
        print_config(cnf)  # print the parameter combination now
        set_random_seed(cnf['seed'])
        model = ModelHandler(cnf)
        f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = model.train()
        # 记录每次运行的结果
        results['f1_macro'].append(f1_mac_test)
        results['auc'].append(auc_test)
        results['g_mean'].append(gmean_test)
        print("F1-Macro: {}, AUC: {}, G-Mean: {}".format(f1_mac_test, auc_test, gmean_test))

    # # 输出总结
    # for key in results:
    #     mean_val = np.mean(results[key])
    #     std_val = np.std(results[key], ddof=1)
    #     print("{}: Mean = {}, STD = {}".format(key, mean_val, std_val))

def multi_run_main(config):
    print_config(config)
    results = defaultdict(list)
    configs = grid(config)

    # 获取当前脚本的绝对路径
    base_dir = os.path.abspath(os.path.dirname(__file__))
    results_dir = os.path.join(base_dir, 'grid', datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(results_dir, exist_ok=True)

    results_file_path = os.path.join(results_dir, 'results.csv')
    # 假设你知道所有配置参数的键
    config_keys = sorted(config.keys())  # 获取所有参数的排序列表，确保一致性
    header = config_keys + ['Configuration', 'F1-Macro', 'F1-1', 'F1-0', 'AUC', 'G-Mean']

    with open(results_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # 写入头部

    for i, cnf in enumerate(configs):
        print('Running configuration {}:\n'.format(i))
        set_random_seed(cnf['seed'])
        model = ModelHandler(cnf)
        f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = model.train()
        results['f1_macro'].append(f1_mac_test)
        results['auc'].append(auc_test)
        results['g_mean'].append(gmean_test)

        # 准备写入的行，包括配置参数和性能指标
        row = [cnf.get(key, '') for key in config_keys] + [i, f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test]
        with open(results_file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

        print("Configuration {}: F1-Macro: {}, AUC: {}, G-Mean: {}".format(
            i, f1_mac_test, auc_test, gmean_test))

    # 生成总结性结果文件
    with open(os.path.join(results_dir, 'summary_results.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Mean', 'STD'])
        for key in results:
            mean_val = np.mean(results[key])
            std_val = np.std(results[key], ddof=1)
            writer.writerow([key, mean_val, std_val])
            print("{}: Mean = {}, STD = {}".format(key, mean_val, std_val))

    return results







################################################################################
# ArgParse and Helper Functions #
################################################################################
def get_config(config_path="config.yml"):
    with open(config_path, "r", encoding='utf-8') as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', required=True, type=str, help='path to the config file')
    parser.add_argument('--multi_run', action='store_true', help='flag: multi run')
    args = vars(parser.parse_args())
    return args


def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


# modified
def grid(kwargs):
    """Builds a mesh grid with given keyword arguments for this Config class.
    If the value is not a list, then it is considered fixed"""

    class MncDc:
        def __init__(self, a):
            self.a = a  # tuple!
        def __call__(self):
            return self.a

    def merge_dicts(*dicts):
        from functools import reduce
        def merge_two_dicts(x, y):
            z = x.copy()
            z.update(y)
            return z
        return reduce(lambda a, nd: merge_two_dicts(a, nd if nd else {}), dicts, {})

    sin = OrderedDict({k: v if isinstance(v, list) else [v] for k, v in kwargs.items()})
    for k, v in sin.items():
        copy_v = []
        for e in v:
            copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
        sin[k] = copy_v

    grd = np.array(np.meshgrid(*sin.values()), dtype=object).T.reshape(-1, len(sin.values()))
    return [merge_dicts(
        {k: v for k, v in kwargs.items() if not isinstance(v, list)},
        {k: vv[i]() if isinstance(vv[i], MncDc) else vv[i] for i, k in enumerate(sin)}
    ) for vv in grd]





################################################################################
# Module Command-line Behavior #
################################################################################
if __name__ == '__main__':
    cfg = get_args()
    config = get_config(cfg['config'])
    print(config)
    #multi_run_main(config)

    if cfg['multi_run']:
        multi_run_main(config)
    else:
        main(config)

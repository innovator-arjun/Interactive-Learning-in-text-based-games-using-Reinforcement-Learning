import csv
import os
import sys
from contextlib import ExitStack
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import stats


col_names = {
    'steps': 0,
    'reward': 1,
    'moves': 2,
    'test_reward': 3,
    'test_moves': 4
}


def visualize(test_name, files, columns, step_win=1000, smooth_win=101):
    band_list = []
    file_list = []
    for f in files:
        if isinstance(f, list):
            band_list.append(len(f))
            file_list = file_list + f
        else:
            band_list.append(1)
            file_list.append(f)
    data = get_data(file_list, columns, step_win=step_win, smooth_win=smooth_win)

    for col, d in data:
        plt.clf()

        count = 0
        i = 0
        while i < len(d):
            if band_list[count] == 1:
                n, v = d[i]
                plt.plot(v, label=n)
            else:
                n = d[i][0]
                data_lists = [x[1] for x in d[i:i + band_list[count]]]
                max_len = max(len(x) for x in data_lists)
                data_list = []
                for j in range(max_len):
                    data_list.append([])
                    for l in data_lists:
                        if j < len(l):
                            data_list[-1].append(l[j])
                high, low, avg = [], [], []
                for v in data_list:
                    avg.append(sum(v) / len(v))
                    high.append(avg[-1] + stats.tstd(v))
                    low.append(avg[-1] - stats.tstd(v))
                plt.plot(list(range(0, len(avg) * step_win, step_win)), avg, label=n)
                plt.fill_between(list(range(0, len(avg) * step_win, step_win)), low, high, alpha=.2)
            i += band_list[count]
            count += 1

        plt.legend(loc=0)
        # plt.title(test_name)
        plt.ylabel(col)
        plt.xlabel('Steps')
        plt.savefig('images/' + test_name + '_' + col + '.png')


def get_data(files, columns, step_win=1000, smooth_win=101):
    data = [(col, []) for col in columns]
    for path in files:
        name = path[path.rfind('game') + 8:path.find('.')]
        with open(path) as file:
            reader = csv.reader(file, delimiter=',')
            indices = []
            values = []
            for row in reader:
                indices.append(row[0])
                values.append(row[1:])
            cols = smooth(indices, values, window=step_win)
            for i, c in enumerate(columns):
                v = cols[col_names[c]]
                if smooth_win > 3 and smooth_win < len(v):
                    v = savgol_filter(v, smooth_win, 3, mode='nearest')[int(smooth_win / 2):int(-smooth_win / 2)]
                data[i][1].append((name, v))
    return data


def smooth(indices, values, window=1000):
    new_values = [[] for _ in range(len(values[0]))]

    curr_index = 0
    last_values = [0 for _ in range(len(values[0]))]
    for i in range(window, int(indices[-1]), window):
        to_avg = []
        averaged = [last_values[j] for j in range(len(values[0]))]
        while int(indices[curr_index]) < i:
            to_avg.append([float(x) for x in values[curr_index]])
            curr_index += 1
        if len(to_avg) > 0:
            for j in range(len(to_avg[0])):
                averaged[j] = sum(x[j] for x in to_avg) / len(to_avg)
                last_values[j] = averaged[j]
        for j in range(len(values[0])):
            new_values[j].append(averaged[j])
    return new_values


if __name__ == '__main__':
    columns = ['reward']
    files = []
    visualize('games', files, columns, step_win=1000, smooth_win=101)

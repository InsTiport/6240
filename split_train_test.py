import tqdm
import numpy as np
from numpy.random import default_rng

rng = default_rng()

# read train and test data
user_subreddit_rating_list = []
with open('user_subreddit.tsv', 'r') as r:
    line_counter = 0
    for line in tqdm.tqdm(r.readlines()):
        if line_counter >= 1:
            splits = line.split('\t')
            user, subreddit, rating = int(splits[0]), int(splits[1]), float(splits[2])
            user_subreddit_rating_list.append([user, subreddit, rating])

        line_counter += 1

all_data = np.array(user_subreddit_rating_list)
print('Splitting data...')
for i in range(5):
    training_idx = rng.choice(len(all_data), int(0.8 * len(all_data)), replace=False)
    test_idx = np.array(([i for i in range(len(all_data)) if i not in training_idx]))
    rng.shuffle(test_idx)
    with open(f'split_{i}.txt', 'w') as w:
        for j, idx in enumerate(training_idx):
            w.write(str(idx))
            if j != len(training_idx) - 1:
                w.write(',')
        w.write('\n')
        for j, idx in enumerate(test_idx):
            w.write(str(idx))
            if j != len(test_idx) - 1:
                w.write(',')

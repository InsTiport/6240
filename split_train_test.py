import tqdm
import numpy as np

# # read train and test data
# user_subreddit_rating_list = []
# with open('user_subreddit.tsv', 'r') as r:
#     line_counter = 0
#     for line in tqdm.tqdm(r.readlines()):
#         if line_counter >= 1:
#             splits = line.split('\t')
#             user, subreddit, rating = int(splits[0]), int(splits[1]), float(splits[2])
#             user_subreddit_rating_list.append([user, subreddit, rating])
#
#         line_counter += 1

line_count = 0
with open('reddit.csv') as r:
    for line in tqdm.tqdm(r.readlines()):
        line_count += 1


def get_split(size, train_proportion=0.8, seed=0):

    train_size = round(size * train_proportion)
    np.random.seed(seed)
    train_ids = np.random.choice(np.arange(size), train_size, replace=False)
    test_ids = np.setdiff1d(np.arange(size), train_ids)

    return train_ids, test_ids


for split in range(5):
    training_idx, test_idx = get_split(line_count - 1, seed=split)
    with open(f'split_{split}.txt', 'w') as w:
        for j, idx in enumerate(training_idx):
            w.write(str(idx))
            if j != len(training_idx) - 1:
                w.write(',')
        w.write('\n')
        for j, idx in enumerate(test_idx):
            w.write(str(idx))
            if j != len(test_idx) - 1:
                w.write(',')

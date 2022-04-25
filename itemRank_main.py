"""
Credit: adapted from https://github.com/arashkhoeini/itemrank
"""

import tqdm
import numpy as np
from ItemRank import ItemRank


def uniquify(a):
    unique_tuples = set()
    for row in a:
        unique_tuples.add(tuple(row))

    return np.array(list(unique_tuples))


def count_matched(l1, l2):
    return len(set(l1) & set(l2))


# read train and test data
user_subreddit_rating = dict()
with open('user_subreddit.tsv', 'r') as r:
    line_counter = 0
    for line in tqdm.tqdm(r.readlines()):
        if line_counter >= 1:
            splits = line.split('\t')
            user, subreddit, rating = int(splits[0]), int(splits[1]), float(splits[2])
            user_subreddit_rating[user, subreddit] = rating

        line_counter += 1

all_data_list = []
line_counter = 0
with open('reddit.csv') as r:
    for line in tqdm.tqdm(r.readlines()):
        line_counter += 1

        # exclude header
        if line_counter > 1:
            # get all data attributes
            split = line.split(',')
            user_id, subreddit_id = int(split[0]), int(split[1])

            all_data_list.append([user_id, subreddit_id, user_subreddit_rating[user_id, subreddit_id]])

all_data = np.array(all_data_list)
unique_all_data = uniquify(all_data)

print('Training and evaluating...\n')
for split in range(5):
    print(f'Split {split}:')
    with open(f'split_{split}.txt', 'r') as r:
        training_idx = np.array(r.readline().split(','), dtype=int)
        test_idx = np.array(r.readline().split(','), dtype=int)

    np_data = all_data[training_idx]
    np_test_data = all_data[test_idx]

    np_data = uniquify(np_data)
    np_test_data = uniquify(np_test_data)

    # training and testing the model
    item_rank = ItemRank(np_data, unique_all_data)
    item_rank.generate_graph()
    item_rank.generate_coef_from_graph()
    DOAs = []

    macro_avg_accuracy = []
    micro_avg_num, micro_avg_den = 0, 0
    for user_name in tqdm.tqdm(range(10000)):
        Tu = item_rank.calculate_Tu(np_test_data, user_name)
        if len(Tu) == 0:
            continue
        d = item_rank.generate_d(user_name=user_name)
        IR = np.ones(len(item_rank.movie_names))
        old_IR = IR
        converged = False
        counter = 0
        while not converged:
            counter += 1
            old_IR = IR
            IR = item_rank.item_rank(0.85, IR, d)
            converged = (old_IR - IR < 0.0001).all()
        # print(f'Converged after {str(counter)} counts.')
        doa = item_rank.calculate_DOA(np_test_data, user_name, IR)
        DOAs.append(doa)

        Lu = item_rank.calculate_Lu(user_name)
        predicted_interested_subreddits = item_rank.get_most_similar(IR, Lu, len(Tu))

        num_matched = count_matched(Tu, predicted_interested_subreddits)
        macro_avg_accuracy.append(num_matched / len(Tu))
        micro_avg_num += num_matched
        micro_avg_den += len(Tu)

        # print(f'DOA for user {user_name} is : {doa}')
    print(f'Macro DOA for split {split} is: {sum(DOAs) / len(DOAs)}')
    print(f'Macro average accuracy for split {split} is {np.mean(macro_avg_accuracy)}')
    print(f'Micro average accuracy for split {split} is {micro_avg_num / micro_avg_den}\n')

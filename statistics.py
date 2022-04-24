from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tqdm

line_count = 0

# user_id: [] containing all subreddit ids the user comments on
user_counter = defaultdict(list)
subreddit_counter = defaultdict(list)

# user_id: subreddit_id: [] containing all timestamps the user comments on this subreddit
user_counter_ts = defaultdict(lambda: defaultdict(list))
subreddit_counter_ts = defaultdict(lambda: defaultdict(list))

# record all unique state labels (found that only 2 are present)
state_label_set = set()
state_label_0_count = 0
state_label_1_count = 0
# only a tiny percent of data has state label 1
# record all data points with state label 1
state_label_1_list = []

# min and max timestamp values (found that the max corresponds to 1 month)
min_ts = max_ts = 0

'''
Read file
'''
with open('reddit.csv') as r:
    for line in tqdm.tqdm(r.readlines()):
        line_count += 1

        # exclude header
        if line_count > 1:
            # get all data attributes
            split = line.split(',')
            user_id, subreddit_id, timestamp, state_label, features = int(split[0]), int(split[1]), float(split[2]), int(split[3]), split[4:]
            if '\n' in features[-1]:
                features[-1] = features[-1].replace('\n', '')
            features = [float(feature) for feature in features]
            features = np.array(features)

            # add to storages
            user_counter[user_id].append(subreddit_id)
            subreddit_counter[subreddit_id].append(user_id)
            user_counter_ts[user_id][subreddit_id].append(timestamp)
            subreddit_counter_ts[subreddit_id][user_id].append(timestamp)

            state_label_set.add(state_label)
            min_ts = min(min_ts, timestamp)
            max_ts = max(max_ts, timestamp)
            # if user_id == 199 and subreddit_id == 19:
            #     print(state_label)
            #     print(timestamp)
            #     print('\n')
            if state_label == 0:
                state_label_0_count += 1
            else:
                state_label_1_count += 1
                state_label_1_list.append([user_id, subreddit_id])

user_counter, subreddit_counter = dict(user_counter), dict(subreddit_counter)
for user in user_counter_ts:
    user_counter_ts[user] = dict(user_counter_ts[user])
for subreddit in subreddit_counter_ts:
    subreddit_counter_ts[subreddit] = dict(subreddit_counter_ts[subreddit])
user_counter_ts, subreddit_counter_ts = dict(user_counter_ts), dict(subreddit_counter_ts)

# frequencies
user_subreddit_counts = sorted([len(user_counter[user]) for user in user_counter])
subreddit_user_counts = sorted([len(subreddit_counter[subreddit]) for subreddit in subreddit_counter])


'''
Plot
'''
PERCENTILE = 95
BINS = 20

percentage = PERCENTILE / 100
ax = sns.histplot(user_subreddit_counts,
                  binrange=(min(user_subreddit_counts), user_subreddit_counts[int(percentage * len(user_subreddit_counts))]),
                  bins=BINS, log_scale=(False, True))
ax.set(xlabel='number of total posts a user made', ylabel='user count')
plt.savefig('posts_user.png')
plt.clf()
ax = sns.histplot(subreddit_user_counts,
                  binrange=(min(subreddit_user_counts), subreddit_user_counts[int(percentage * len(subreddit_user_counts))]),
                  bins=BINS, log_scale=(False, True))
ax.set(xlabel='number of posts in a subreddit', ylabel='subreddit count')
plt.savefig('posts_subreddit.png')
plt.clf()

session_len_list = []
for subreddit in subreddit_counter:
    for user in subreddit_counter_ts[subreddit]:
        session_len_list.append(len(subreddit_counter_ts[subreddit][user]))
session_len_list = sorted(session_len_list)

ax = sns.histplot(session_len_list,
                  binrange=(min(session_len_list), session_len_list[int(percentage * len(session_len_list))]),
                  bins=BINS, log_scale=(False, True))
ax.set(xlabel='number of posts in a session', ylabel='session count')
plt.savefig('posts_session.png')


'''
Print out some stats
'''
print(f'Stats of number of posts a user made: min = {min(user_subreddit_counts)}, max = {max(user_subreddit_counts)},'
      f'median = {user_subreddit_counts[len(user_subreddit_counts) // 2]}, avg = {int(np.mean(user_subreddit_counts))}')
print(f'Stats of number of posts in a subreddit: min = {min(subreddit_user_counts)}, max = {max(subreddit_user_counts)},'
      f'median = {subreddit_user_counts[len(subreddit_user_counts) // 2]}, avg = {int(np.mean(subreddit_user_counts))}')
print(f'Stats of number of posts in a session: min = {min(session_len_list)}, max = {max(session_len_list)},'
      f'median = {session_len_list[len(session_len_list) // 2]}, avg = {int(np.mean(session_len_list))}')

print(f'There are {len(user_counter)} users and {len(subreddit_counter)} subreddits.')
print(f'Possible state labels: {state_label_set}.')
print(f'Number of interactions with state label 0: {state_label_0_count}.')
print(f'Number of interactions with state label 1: {state_label_1_count}.')
print(f'State label 1 list: {state_label_1_list}')
print(f'There are {line_count - 1} user-subreddit interactions.')
print(f'Min timestamp: {min_ts}, max timestamp: {max_ts}')


'''
Write processed data to tsv
'''
session_counter = 0
with open('user_subreddit.tsv', 'w') as w:
    w.write('user_id\tsubreddit_id\trating\tcomma_separated_list_of_timestamps_the_user_posted_on_this_subreddit\n')
    for user in user_counter_ts:
        for subreddit in user_counter_ts[user]:
            w.write(str(user))
            w.write('\t')
            w.write(str(subreddit))
            w.write('\t')
            # temporary rating, may need more accurate ways to calculate
            rating = round(len(user_counter_ts[user][subreddit]) / len(user_counter[user]), 3)
            w.write(f'{rating}')
            w.write('\t')
            w.write(str(user_counter_ts[user][subreddit]))
            w.write('\n')
            session_counter += 1

print(f'There are {session_counter} sessions.')

with open('subreddit_user.tsv', 'w') as w:
    w.write('subreddit_id\tuser_id\trating\tcomma_separated_list_of_timestamps_of_posts_made_by_the_user\n')
    for subreddit in subreddit_counter:
        for user in subreddit_counter_ts[subreddit]:
            w.write(str(subreddit))
            w.write('\t')
            w.write(str(user))
            w.write('\t')
            # temporary rating, may need more accurate ways to calculate
            rating = round(len(subreddit_counter_ts[subreddit][user]) / len(user_counter[user]), 3)
            w.write(f'{rating}')
            w.write('\t')
            w.write(str(subreddit_counter_ts[subreddit][user]))
            w.write('\n')

import pandas as pd
import numpy as np
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

class CFRecommender:
    def __init__(self, threshold=0.058):
        self.threshold = threshold
        self.streamer_per_user = defaultdict(set)
        self.users_per_streamer = defaultdict(set)

    def fit(self, train_data):
        for d in train_data:
            user, streamer = d['user_id'], d['streamer_username']
            self.streamer_per_user[user].add(streamer)
            self.users_per_streamer[streamer].add(user)

    def predict(self, user, streamer):
        max_sim = float('-inf')
        users = self.users_per_streamer[streamer]
        for s2 in self.streamer_per_user[user]:
            sim = jaccard_similarity(users, self.users_per_streamer[s2])
            if sim > max_sim:
                max_sim = sim
            if sim == 1:
                break
        return 1 if max_sim > self.threshold else 0

    def get_recommendations(self, user_id, df, n=10):
        all_streamers = set(df['streamer_username'].unique())
        user_streamers = set(df[df['user_id'] == user_id]['streamer_username'])
        unseen_streamers = list(all_streamers - user_streamers)
        predictions = [(streamer, self.predict(user_id, streamer)) for streamer in unseen_streamers]
        return sorted(predictions, key=lambda x: x[1], reverse=True)[:n]

class MF:
    def __init__(self, num_factors=10, epochs=50, initial_gamma=0.01, lambda_reg=0.13, early_stopping_rounds=5):
        self.num_factors = num_factors
        self.epochs = epochs
        self.initial_gamma = initial_gamma
        self.lambda_reg = lambda_reg
        self.early_stopping_rounds = early_stopping_rounds
        self.user_biases = defaultdict(float)
        self.item_biases = defaultdict(float)
        self.user_factors = {}
        self.item_factors = {}
        self.mu = 0
        self.max_rating = float('inf')

    def _init_factors(self, users, items):
        for user in users:
            self.user_factors[user] = np.random.normal(0, 0.1, self.num_factors)
        for item in items:
            self.item_factors[item] = np.random.normal(0, 0.1, self.num_factors)

    def fit(self, train_data, validation_data):
        users = set(d['user_id'] for d in train_data)
        items = set(d['streamer_username'] for d in train_data)
        self.mu = np.mean([d['time_spent'] for d in train_data])
        self.max_rating = max(d['time_spent'] for d in train_data)
        self._init_factors(users, items)

        best_rmse = float('inf')
        stopping_counter = 0
        for epoch in range(self.epochs):
            gamma_epoch = self.initial_gamma / (1 + 0.1 * epoch)
            print(f"\nEpoch {epoch+1}/{self.epochs} with learning rate: {gamma_epoch:.5f}")
            for d in tqdm(train_data, desc=f"Training epoch {epoch+1}"):
                user, item, rating = d['user_id'], d['streamer_username'], d['time_spent']
                pred = self._predict_raw(user, item)
                error = rating - pred
                self.user_biases[user] += gamma_epoch * (error - self.lambda_reg * self.user_biases[user])
                self.item_biases[item] += gamma_epoch * (error - self.lambda_reg * self.item_biases[item])
                user_factors = self.user_factors[user]
                item_factors = self.item_factors[item]
                self.user_factors[user] += gamma_epoch * (error * item_factors - self.lambda_reg * user_factors)
                self.item_factors[item] += gamma_epoch * (error * user_factors - self.lambda_reg * item_factors)

            if validation_data:
                rmse = self._compute_rmse(validation_data)
                print(f"Epoch {epoch+1}, Validation RMSE: {rmse:.4f}")
                if rmse < best_rmse:
                    best_rmse = rmse
                    self.best_user_biases = self.user_biases.copy()
                    self.best_item_biases = self.item_biases.copy()
                    self.best_user_factors = self.user_factors.copy()
                    self.best_item_factors = self.item_factors.copy()
                    stopping_counter = 0
                else:
                    stopping_counter += 1

                if stopping_counter >= self.early_stopping_rounds:
                    print("Early stopping triggered.")
                    self.user_biases = self.best_user_biases
                    self.item_biases = self.best_item_biases
                    self.user_factors = self.best_user_factors
                    self.item_factors = self.best_item_factors
                    break

    def _predict_raw(self, user, item):
        pred = self.mu
        if user in self.user_biases and item in self.item_biases:
            pred += self.user_biases[user] + self.item_biases[item]
            pred += np.dot(self.user_factors[user], self.item_factors[item])
        elif user in self.user_biases:
            pred += self.user_biases[user]
        elif item in self.item_biases:
            pred += self.item_biases[item]
        return pred

    def predict(self, user, item):
        pred = self._predict_raw(user, item)
        pred = max(0, min(pred, self.max_rating))
        if abs(pred - round(pred)) < 0.05:
            pred = round(pred)
        return pred

    def _compute_rmse(self, validation_data):
        squared_errors = []
        for d in validation_data:
            user, item, rating = d['user_id'], d['streamer_username'], d['time_spent']
            pred = self.predict(user, item)
            squared_errors.append((rating - pred) ** 2)
        return math.sqrt(np.mean(squared_errors))

    def get_top_n_recommendations(self, user_id, df, n=10):
        all_streamers = set(df['streamer_username'].unique())
        user_streamers = set(df[df['user_id'] == user_id]['streamer_username'])
        unseen_streamers = list(all_streamers - user_streamers)
        predictions = [(streamer, self.predict(user_id, streamer)) for streamer in unseen_streamers]
        return sorted(predictions, key=lambda x: x[1], reverse=True)[:n]

class JaccardPopularityModel:
    def __init__(self, threshold=0.058, popularity_threshold=10):
        self.threshold = threshold
        self.popularity_threshold = popularity_threshold
        self.streamer_per_user = defaultdict(set)
        self.users_per_streamer = defaultdict(set)

    def fit(self, train_data):
        for d in train_data:
            user, streamer = d['user_id'], d['streamer_username']
            self.streamer_per_user[user].add(streamer)
            self.users_per_streamer[streamer].add(user)

    def predict(self, user, streamer):
        max_sim = float('-inf')
        users = self.users_per_streamer[streamer]
        
        # Cold-start user, consider only popularity
        if user not in self.streamer_per_user:
            return 1 if len(users) > self.popularity_threshold else 0
        
        for s2 in self.streamer_per_user[user]:
            sim = jaccard_similarity(users, self.users_per_streamer[s2])
            if sim > max_sim:
                max_sim = sim
            if sim == 1:
                break
        
        return 1 if max_sim > self.threshold or len(users) > self.popularity_threshold else 0

def jaccard_similarity(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom if denom != 0 else 0

# def create_evaluation_pairs(df, test_data, n_negative=1):
#     positive_pairs = [(d['user_id'], d['streamer_username']) for d in test_data]
#     negative_pairs = []
#     all_streamers = set(df['streamer_username'].unique())
#     for user in df['user_id'].unique():
#         user_streamers = set(df[df['user_id'] == user]['streamer_username'])
#         available_streamers = list(all_streamers - user_streamers)
#         if available_streamers:
#             negative_pairs.extend([(user, random.choice(available_streamers)) for _ in range(n_negative)])
#     return positive_pairs, negative_pairs

def create_evaluation_pairs(df, test_data, n_negative=1, sample_size=0.1):
    positive_pairs = [(d['user_id'], d['streamer_username']) for d in test_data]
    negative_pairs = []
    all_streamers = set(df['streamer_username'].unique())
    
    # Sample a subset of users
    sampled_users = random.sample(df['user_id'].unique().tolist(), int(len(df['user_id'].unique()) * sample_size))
    
    for user in sampled_users:
        user_streamers = set(df[df['user_id'] == user]['streamer_username'])
        available_streamers = list(all_streamers - user_streamers)
        if available_streamers:
            negative_pairs.extend([(user, random.choice(available_streamers)) for _ in range(n_negative)])
    
    return positive_pairs, negative_pairs

def evaluate_models(models, test_data, negative_samples):
    results = {}
    y_true = [1] * len(test_data) + [0] * len(negative_samples)
    
    for model_name, model in models.items():
        y_pred = []
        for d in test_data:
            y_pred.append(model.predict(d['user_id'], d['streamer_username']))
        for u, s in negative_samples:
            y_pred.append(model.predict(u, s))
        
        accuracy = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true)
        auc = roc_auc_score(y_true, y_pred)
        
        results[model_name] = {'accuracy': accuracy, 'auc': auc}
    
    return results

print("Loading and preprocessing data...")
td = pd.read_csv('100k_a.csv', header=None)
td = td.rename(columns={0: 'user_id', 1: 'stream_id', 2: 'streamer_username', 3: 'start', 4: 'stop'})
td['start'] = td['start'] * 10
td['stop'] = td['stop'] * 10
td['time_spent'] = td['stop'] - td['start']
td = td.drop(['stream_id', 'start', 'stop'], axis=1)

print("Splitting data...")
train_df, temp_df = train_test_split(td, test_size=0.3, random_state=123)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=123)
train_data = train_df.to_dict('records')
valid_data = valid_df.to_dict('records')
test_data = test_df.to_dict('records')

print("\nTraining Collaborative Filtering model...")
cf_model = CFRecommender()
cf_model.fit(train_data)

print("\nTraining Matrix Factorization model...")
mf_model = MF(
    num_factors=10,
    epochs=50,
    initial_gamma=0.01,
    lambda_reg=0.13,
    early_stopping_rounds=5
)
mf_model.fit(train_data, valid_data)

print("\nTraining Jaccard + Popularity model...")
jaccard_pop_model = JaccardPopularityModel()
jaccard_pop_model.fit(train_data)

print('creating evaluation pairs')
positive_pairs, negative_pairs = create_evaluation_pairs(td, test_data)
all_pairs = positive_pairs + negative_pairs

models = {
    'CF': cf_model,
    'MF': mf_model,
    'Jaccard+Popularity': jaccard_pop_model
}

results = evaluate_models(models, test_data, negative_pairs)

print("\nResults:")
for model_name, metrics in results.items():
    print(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")

plt.figure(figsize=(10, 6))
for model_name, model in models.items():
    y_true = [1] * len(test_data) + [0] * len(negative_pairs)
    y_pred = [model.predict(d['user_id'], d['streamer_username']) for d in test_data] + \
             [model.predict(u, s) for u, s in negative_pairs]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend()
plt.show()

sample_user = td['user_id'].iloc[0]
print(f"\nTop 10 CF recommendations for user {sample_user}:")
cf_recommendations = cf_model.get_recommendations(sample_user, td)
for streamer, score in cf_recommendations:
    print(f"{streamer}: {score:.2f}")

print(f"\nTop 10 MF recommendations for user {sample_user}:")
mf_recommendations = mf_model.get_top_n_recommendations(sample_user, td)
for streamer, score in mf_recommendations:
    print(f"{streamer}: {score:.2f}")

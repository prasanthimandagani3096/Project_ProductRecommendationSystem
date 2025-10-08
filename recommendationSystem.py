# Import statements
import json
import pandas as pd
import spacy
import numpy as np
import gym
import random
import pylab
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
import gzip
from urllib.request import urlopen
from collections import deque
from keras import layers, models, optimizers
from keras.optimizers import Adam
import gc
from keras import backend as K

pd.set_option('display.max_colwidth', None)

def loadDataSet():
    # Loading JSON data
    data = []
    with gzip.open('project/AMAZON_FASHION.json.gz') as f:
        for l in f:
            data.append(json.loads(l.strip()))
    # Let's take a peek at the first row and the total number of rows
    print(len(data))
    print(data[0])
    return data

#Create FashionProduct class for a product representation from reviews
class FashionProduct() : pass

def preprocessingData(data):
    # Create a DataFrame for easier data manipulation
    df = pd.DataFrame(data)
    df = df[['overall','verified','reviewerID','asin','style','reviewerName','reviewText', 'summary','reviewTime']]
    # Filter verified reviews with non-null overall ratings
    filtered_df = df[(df['verified'] == True) & (~df['overall'].isnull())]

    # Group reviews by reviewers and select users with more than ten purchases
    reviewer_values = []
    grouped_df_reviwerId = filtered_df.groupby('reviewerID')

    for reviewerId, group in grouped_df_reviwerId:
        products = group[group['asin'].notna()]['asin'].unique()
        if len(products) > 10:
            reviewer_values.append({"reviewerId" : reviewerId, "products": products})

    # Filter dataset to include only reviewers with more than ten products
    filtered_df = filtered_df[(filtered_df['reviewerID'].isin([reviewer['reviewerId'] for reviewer in reviewer_values]))]

    # Group reviews by product ASIN, reviewerID, and reviewTime
    filtered_df['reviewTime'] = pd.to_datetime(filtered_df['reviewTime'])
    filtered_df.sort_values('reviewTime')
    grouped_df = filtered_df.groupby(['asin', 'reviewerID', 'reviewTime'], sort=False)

    return grouped_df, reviewer_values

data = loadDataSet()
grouped_df, reviewer_values = preprocessingData(data)

# extract nouns from review text
def extract_nouns(doc):
    return " ".join([token.text for token in doc if token.pos_ == "NOUN" or token.pos_ == "PROPN"])

#load spacy for nlp related noun extraction, stopword removal and others
nlp = spacy.load('en_core_web_sm')

# Initialize a dictionary to store product features as states
states = {}
products = {}
    
# Iterate over each product
for (product_asin, reviewerId, reviewTime), group in grouped_df:
    if (product_asin, reviewerId) in states: continue
    product = FashionProduct()
    product.product_asin = product_asin
    product.reviewerId = reviewerId
    product.time = reviewTime
    if product_asin not in products:
        products[product_asin] = product
        p= products[product_asin]
        p.reviewers = set()
        p.sizes = set()
        p.colors = set()
        p.reviews = set()
        p.rating =[]

    product.reviewers = products[product_asin].reviewers

    # extract size and color metadata from style column
    styles=  group[group['style'].notna()]['style']
    sizes = styles.apply(lambda x: x.get("Size:", "") if "Size:" in x else x.get("Size Name:", "")).unique().tolist()
    colors = styles.apply(lambda x: x.get("Color:", "")).unique().tolist()

    products[product_asin].sizes.update(sizes)
    products[product_asin].colors.update(colors)

    #extract other noun metadata from review text
    reviews = group[group['reviewText'].notna()]['reviewText']
    reviews = " ".join(reviews.apply(lambda x: " ".join([extract_nouns(chunk) for chunk in nlp(x).noun_chunks]).strip()).unique())
    products[product_asin].reviews.update(reviews)
        
    #using rms instead of average for review ratings to give slightly higher weightage to good reviews
    ratings = group[group['overall']>0]['overall'].tolist()
    products[product_asin].rating.extend(ratings)
    product.ratings = np.sqrt(np.mean( [r**2 for r in products[product_asin].rating]))

    sizes = " ".join(products[product_asin].sizes)
    colors = " ".join(products[product_asin].colors)
    reviews = " ".join(products[product_asin].reviews)
    product.metadata= " ".join((reviews+" "+sizes+" "+colors).split())

    # add past product and reviewer's product metadata
    # we will take metatdata of last 2 reviewer only as large metadata causes memory issues
    for reviewer in list(product.reviewers)[-2:]:
        state = states[(product_asin, reviewer)]
        product.metadata += " "+state.metadata

    # keep past reviewer list
    products[product_asin].reviewers.add(reviewerId)

    states[(product_asin, reviewerId)] = product

states_list= list(states.values())

# Create states for users and enhance metadata with past products
for state in states_list:
    if not any(rv['reviewerId'] == state.reviewerId for rv in reviewer_values):
        reviewer_values.append({'reviewerId': state.reviewerId, 'products' : []})
    prods = []
    for rv in reviewer_values:
        if rv['reviewerId'] == state.reviewerId:
            prods = rv['products']
    for prod1 in prods[-2:]:
        state1 = states[(prod1, state.reviewerId)]
        state.metadata += state1.metadata


# Remove states with empty metadata
states_list = [s for s in states_list if s.metadata.strip() != '']

# Create Product Recommendation env
class RecommendationEnv(gym.Env):
    def __init__(self, states, states_dict, iterations = 10):
        self.states = states
        self.state = self.states[0]
        self.states_dict = states_dict
        self.iterations = iterations
        self.index = 0
        state.action = 0

    def step(self, actions):
        # Implement the transition logic based on the action
        reward= 0
        done= False
        reviewerId= self.state.reviewerId
        prods = []
        for rv in reviewer_values:
            if rv['reviewerId'] == reviewerId:
                prods = rv['products']
        future_asins= [p for p in prods if self.states_dict[(p,reviewerId)].time>self.state.time]
        matched_recommendations = False
        #predicted recommendations
        for i in actions:
          if self.states[i].product_asin in future_asins:
            self.action = i
            matched_recommendations = True
            break;


        if matched_recommendations:
            #Higher reward as they are bought products for the user in future
            reward = 1
        else:
            self.action = actions[0]

        self.index += 1
        self.state = self.states[self.index]
        print(f"iteration :{self.index}")
        if (self.iterations == self.index): done = True

        return self.state, reward, done, {}


    def reset(self, iterations = 10):
        # Reset the state to the initial position
        self.state = self.states[0]
        self.iterations = iterations
        self.index = 0
        return self.state

# Create the custom environment
env = RecommendationEnv(states_list, states, 10)

# Implementation of DQN algorithm
class DQNAgent:
    def __init__(self, state_size, action_size, states):
        self.states = states
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000
        # create replay memory using deque
        self.memory = deque(maxlen=2000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        #Using RNN as it is recommended for text classification
        # Using the TextVectorization layer to normalize, split, and map strings
        # to integers.
        encoder = tf.keras.layers.TextVectorization(max_tokens=10000)
        metadatas = [product.metadata for product in self.states]
        ratings = [product.ratings for product in self.states]
        encoder.adapt(metadatas)

        # Define the input for metadata
        metadata_input = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name="metadata_input")
    
        x = encoder(metadata_input)
        x = layers.Embedding(len(encoder.get_vocabulary()), 32, mask_zero=True)(x)
        x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(16))(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # Create an input layer for ratings
        ratings_input = tf.keras.layers.Input(shape=(1,), name='ratings_input')

        # Concatenate the output of the previous layers with ratings
        concatenated = layers.concatenate([x, ratings_input])

        # Add additional layers for your desired architecture
        dense_layer = layers.Dense(64, activation='relu')(concatenated)
         # One Q-value per action
        output_layer = layers.Dense(len(self.states), activation='linear')(dense_layer)

        # Create the final model with both metadata and ratings as inputs
        model = tf.keras.Model(inputs=[metadata_input, ratings_input], outputs=output_layer)
        # Summary of the model
        model.summary()

        # Compile the model
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get recommendations from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.sample(range(self.action_size),10)
        else:
            # Ensure state.metadata is in the proper format for the TextVectorization layer
            metadata_input = tf.convert_to_tensor([state.metadata], dtype=tf.string)  # Ensure it's a tensor of strings
            ratings_input = np.array([[state.ratings]], dtype=np.float32)  # Ensure it's a 2D array for the ratings
        
            # Predict Q-values using the model
            q_value = self.model.predict([metadata_input, ratings_input])
            return np.argpartition(q_value[0],-10)[-10:]

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input_metadata =[]
        update_input_ratings =[]
        update_target_metadata = []
        update_target_ratings = []
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input_metadata.append(np.array(mini_batch[i][0].metadata))
            update_input_ratings.append(np.array(mini_batch[i][0].ratings))
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target_metadata.append(np.array(mini_batch[i][3].metadata))
            update_target_ratings.append(np.array(mini_batch[i][3].ratings))
            done.append(mini_batch[i][4])

        # Convert metadata and ratings to tensors or arrays
        update_input_metadata = tf.convert_to_tensor(update_input_metadata, dtype=tf.string)  # Tensor of strings
        update_input_ratings = tf.convert_to_tensor(update_input_ratings, dtype=tf.float32)  # Tensor of floats
        update_target_metadata = tf.convert_to_tensor(update_target_metadata, dtype=tf.string)
        update_target_ratings = tf.convert_to_tensor(update_target_ratings, dtype=tf.float32)

        # Ensure proper batching for ratings input
        update_input_ratings = tf.expand_dims(update_input_ratings, axis=-1)  # Shape: (batch_size, 1)
        update_target_ratings = tf.expand_dims(update_target_ratings, axis=-1)  # Shape: (batch_size, 1)

        # Predict Q-values for the current and next states
        target = self.model.predict([update_input_metadata, update_input_ratings], batch_size=batch_size)
        target_val = self.target_model.predict([update_target_metadata, update_target_ratings], batch_size=batch_size)

        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * ( np.amax(target_val[i]))

        # Train the model
        self.model.fit([update_input_metadata, update_input_ratings], target, batch_size=self.batch_size, epochs=1, verbose=1)
        clear_memory()

def clear_memory():
    gc.collect()
    K.clear_session()

state_size = len(env.states)
# Every other product can be a recommendation
action_size = state_size
agent = DQNAgent(state_size, action_size, env.states)

scores, episodes = [], []
EPISODES = 25
#cache already rewarded recommendations (optimization done based upon context and to improve the performance to a large extent)
next_states = {}
done_value = {}
action_value = {}
for e in range(EPISODES):
    done = False
    score = 0
    state = env.reset(50)

    while not done:
        if (state.product_asin, state.reviewerId) in next_states:
          next_state = next_states[(state.product_asin, state.reviewerId)]
          reward = 1
          done = done_value[(state.product_asin, state.reviewerId)]
          action = action_value[(state.product_asin, state.reviewerId)]
          env.index += 1
        else:
          # get action for the current state and go one step in environment
          actions = agent.get_action(state)
          next_state, reward, done, info = env.step(actions)
          action = env.action
          if (reward == 1):
            next_states[(state.product_asin, state.reviewerId)]= next_state
            done_value[(state.product_asin, state.reviewerId)] = done
            action_value[(state.product_asin, state.reviewerId)] = env.action

        # save the sample <s, a, r, s'> to the replay memory
        agent.append_sample(state, action, reward, next_state, done)
        # every time step do the training
        agent.train_model()
        score += reward
        state = next_state

        if done:
            # every episode update the target model to be same with model
            agent.update_target_model()


            scores.append(score)
            episodes.append(e)
            pylab.plot(episodes, scores, 'b')
            print("episode:", e, "  score:", score, "  memory length:",
                  len(agent.memory), "  epsilon:", agent.epsilon)
    if (score > 48): break

print("Number of episodes : ", len(episodes))
print("Number of Scores : ", len(scores))
print("Episodes :  Scores")
for i in range(25):
    print(episodes[i]," : ",scores[i])

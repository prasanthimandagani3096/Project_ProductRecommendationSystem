# Course: AI With Reinforcement Learning

# Project Title: Recommendation System

## Goal:

Recommendation systems are becoming essential in today's world to improve the user experience on a variety of platforms, such as social media, news feeds, streaming services, e-commerce, and personalized learning. Although the recommendation problem was once thought to be a classification or prediction problem, it is now generally accepted that a sequential decision problem is a better way to represent the interaction between the user and the system. As a result, it can be expressed as a Markov decision process (MDP), which algorithms for reinforcement learning (RL) can solve.
Our goal in this project is to learn how to represent a recommendation system as MDP and understand how different RL algorithms can solve this, how to represent the user environments, their database etc., and ultimately create a RL-based recommendation system.

## Problem statement:

The idea of a recommendation system is to help users find their favorite items to purchase, their friends on social networks, and their favorite movies to watch. Most recommendation systems learn user preferences and item popularity from historical data and retrain models on a regular basis. However, there are some disadvantages to this. First, they are designed to maximize the immediate incentive of getting users to click or buy while ignoring long-term rewards such as user activity. Furthermore, they have a greedy attitude and overemphasize item popularity, neglecting to investigate new goods (i.e., the cold-start problem). Finally, they do not perform effectively when new things are regularly introduced. In this project, we aim to address these problems by building an RL-based recommendation system. RL can learn to maximize long-term rewards, strike a balance between exploration and exploitation, and learn continually online.

## Methodology:

### Data Collection:

I am using Amazon product reviews dataset for this Recommendation systems

### Data Pre-processing

Here we reduce the dataset by filtering rows by taking in verified reviews, we clean the data by removing irrelavent data/columns. We group the data based on reviewers to understand the preferences. We organise the data into user states and products by using Product Asin, review time etc to capture valuable information.

### The RL Algorithms

Explored various reinforcement learning approaches for recommendation systems, such as Multi armed bandits with context, value-based methods, and policy-based methods.

1. Contextual bandits collect and observe the context before each action, and choose actions based on the context. They learn about how actions and context affect reward. In the case of recommendations and search, the context would be data we have about the customer (e.g., demographics, device, indicated/historical preferences) and environment (e.g., day of week, time of day).
2. Value-based methods learn the optimal value function. This is either a state function (mapping the state to a value) or a state-action function (mapping the state-action to a value). Using the value function, the agent acts by choosing the action that has the highest value in each state. Ex: Deep Q-Network (DQN).
3. Policy-based methods learn the policy function that maps the state to action directly, without having to learn Q-values. As a result, they perform better in continuous and stochastic environments (i.e., no discrete states or actions) and tend to be more stable, given a sufficiently small learning rate. Ex: Deep REINFORCE

I used DQN RL algorithm. It's well-suited for recommendation systems, where actions involve suggesting products to users. It also helps with focusing on long term rewards.

## Dataset:

* source: https://datarepo.eng.ucsd.edu/mcauley\_group/data/amazon\_v2/categoryFiles/AMAZON\_FASHION.json.gz
* Attached the json file of data under /AMAZON.json folder.

## Step-by-step Algorithm Logic:

1. Loaded the dataset from gzip file.
2. Data preprocessing -

   1. filtered the verified reviewers data
   2. selected reviewers with more than 10 purchases and more than products.
   3. Grouped data by reviewerid and sorting based on reviewed time.

3. Collected the products as states.
4. Created RecommendationSystem gym environment -  this takes each action and reviews if the recommendation is present in the future purchases or not. If yes, attcahes reward as 1 otherwise none.
5. Used DQN algorithm to train the model

   1. This uses RNN (Recurrent Neural Network) because the order of the information matters in this system.
   2. This trains the model based on the Q-value.
   3. Using epsilon greedy method I've maintained the balance between exploration and exploitation.

## Measure of success:

I have used the Q-value as metric to evaluate the performance of a DQN model. Q-value measures the expected reward for performing an action in a given state. Increasing average Q-value is a sign that model is getting better at recommending.
I've ran 25 episodes and created a score metric which collects the rewards in each episode. Our aim would be to gradually improve the reward per each episode.

## Prior/Related works:

1. New Recommendation System Using Reinforcement Learning ([source](https://www.researchgate.net/profile/Phaitoon-Srinil/publication/252413398_New_Recommendation_System_Using_Reinforcement_Learning/links/59f89708a6fdcc075ec98e03/New-Recommendation-System-Using-Reinforcement-Learning.pdf))
   This paper talks about how a recommendation system can be achieved using ε-greedy selection policy and SARSA (state action replacement method). They experimented to find a relation between ε  and user click rate.
2. Reinforcement Learning based Recommender Systems: A Survey ([source](https://dl.acm.org/doi/pdf/10.1145/3543846?casa_token=GBOgA8-piZUAAAAA:Fyv4x4N81ZrCm-ZwMnYHiDi7G7adhsr_qkDxIZnZP0-yCY7b4nNpwPWLbAhiABy4njBlJ-pGeTLx))
   This paper talks about how a Recommender system can be formulated as a Markov decision process (MDP) and be solved by reinforcement learning (RL) algorithms and the RL and Deep-RL algorithms that already exist and their advantages and drawbacks.
3. Reinforcement Learning to Optimize Long-term User Engagement in Recommender Systems ([source](https://dl.acm.org/doi/pdf/10.1145/3292500.3330668?casa_token=k3PwPXf-R9MAAAAA:quBkXAYC8rwqhSX0mjS4NVDy1Z1s-lO-Si8e6Ljq20TBj78E1g2Qdg7_5QlgNvR8esVMeBJ6S5uT))
   This paper demonstrates how to optimize the long term user engagement by the effectiveness of FeedRec framework for feed streaming recommendation using hierarchical RNNs to model complex user behaviors, refer to as Q-Network and an S-Network to simulate the environment and assist the Q-Network.
4. DRN: A Deep Reinforcement Learning Framework for News Recommendation ([source](https://dl.acm.org/doi/pdf/10.1145/3178876.3185994))
   This paper talks about a DQN- based RL framework which can effectively model the dynamic news features and user preferences, and plan for future explicitly, in order to achieve higher reward in the long run. They further consider user return pattern as a supplement to click / no click label in order to capture more user feedback information. In addition, they apply an effective exploration strategy into the Framework to improve the recommendation diversity and look for potential more rewarding recommendations.
5. Recommendations with Negative Feedback via Pairwise Deep Reinforcement Learning ([source](https://arxiv.org/abs/1802.06501))
   This paper shows that items that users skip (i.e., not click) provide useful signal about user preferences and should also be incorporated into recommender systems. They mentioned that a system with only positive items will not change its state or update its strategy when users skip the recommended items. So, they defined the state of their Markov decision process (MDP) to include negative items.

## Other References:

1. http://romisatriawahono.net/lecture/rm/survey/information%20retrieval/Bobadilla%20-%20Recommender%20Systems%20-%202013.pdf
2. https://applyingml.com/resources/rl-for-recsys/

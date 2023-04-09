# Rescue-Trainer
A Multi-Agent Reinforcement Learning (MARL) model to train agents to rescue victims in a Disaster Situation. The environemt of RL is based on GYM's Taxi-v3 and NCMML Project. A DQN model with Greedy Policy is used to train the agents in the environment.

The model is trained to pick up victims from their designated locations and drop them off at the rescue location. The model is trained on two victims and two agents. The rendering of the code is hard coded for 2 agents and 2 victims but the rest of the functions are written to be able to be expanded for multiple victims and multiple agents. 

Training a MARL model for a disaster situation is very useful. Be it Tsunami's or Floods or Earthquake, certain patterns of the disasters always remain the same, i.e. the roads are blocked, people are displaced, there is low connectivity irrespective of which country it is and what kind of building it is. In the project I have touched upon 0.001 % of RL to address the usefulness of MARL in disaster management and planning. MARL can also be used to train a drone to navigate environments. 

# A snapshot of the first 500 steps of the Agent being trained.
![output_final](/home/shinde/Documents/trainings/Spiced_Academy/Github/tahini-tensor-student-code/spiced_projects/WEEK_12/Rescue-Trainer/final_gif_10000_100_steps.gif)

# Requirements
- Python 3.9 or above
- tensorflow
- gym
- Pygame
- keras-rl2
- Jupyter notebook

# Usage
- Run the Jupyter notebook to train the Agents.
- The number of steps can be modified and visualisation can be kept to False or True as per requirement.


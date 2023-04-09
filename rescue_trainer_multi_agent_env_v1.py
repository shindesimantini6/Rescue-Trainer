# Import all required packages
import numpy as np
from gym import Env
from gym.spaces import Discrete, MultiDiscrete, Box
import pygame
from pygame import Rect

from contextlib import closing
from io import StringIO
from os import path
from typing import Optional

import numpy as np

import time
import logging

from gym import Env, logger, spaces, utils
# from gym.envs.toy_text.utils import categorical_sample
from gym.error import DependencyNotInstalled
# Initialize log
logger = logging.getLogger(__name__)

# fhandler = logging.FileHandler(filename= "train_32_episodes.log", mode='a')
# logger.addHandler(fhandler)

logging.basicConfig(filename= "train_20_epi_500_steps_2_vic_part_2.log", level=logging.INFO)

# Action
IDLE = 0
UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4
PICKUP = 5
DROP = 6

# Graphics
WINDOW_PIXELS = 1000
BG_COLOR = (255, 255, 255)
GRID_COLOR = (171, 171, 171)
AGENT_COLOR = (0, 0, 255)
RESOURCE_COLOR = (10, 155, 0)
RADIUS_COLOR = (180, 255, 180)
AGITATED_COLOR = (255, 0, 0)
WHITE = (255,255,255)


def add_element_and_get_index(element, element_list):
    """
    Checks if an element is in a list and adds it to the list if not.
    Returns the index of the element in the list.

    Args:
        element (str):
            Element to be added to the list
        element_list (list):
            List to add the element to

    Returns:
        Index of inserted element in the list
    """

    if element not in element_list:
        element_list.append(element)
    return element_list.index(element)

class Resuerv4(Env):
        
    # Initialization function (constructor)
    def __init__(self, render_mode: Optional[str] = None, 
                 nb_agents=2, nb_victims=2, gridsize=5, 
                 nb_steps=500, reward_extracting=20.0, 
                 reward_else=-1.0,window = None,
                 reward_illegal = -10,
                 seed=1,debug=False):
        # render_mode:          "human", "ansi", "rgb_array"
        # nb_agents:            Number of agents
        # nb_victims:           Number of victims
        # gridsize:             Size of square grid
        # nb_steps:             Number of steps per episode
        # reward_extracting:    Resource extraction reward value
        # reward_else:          Default reward value
        # reward_illegal:       Illegal Dropoff
        # window:               Pygame graphics rendering object
        self.debug = debug

        self.locs = [(0, 0), (0, 4), (4, 0), (4, 3)]

         # pygame utils
        self.window = window
        self.clock = None

        # Compute cell pixel size
        self.cell_pixels = WINDOW_PIXELS / gridsize

        # Set number of possible actions and reset step number
        self.nb_actions = 7
        self.step_nb = 0

        # Destination location
        self.dest_loc = [4,0]

        # Set environment variables
        self.nb_agents = nb_agents
        self.nb_resources = nb_victims
        self.gridsize = gridsize
        self.nb_steps = nb_steps
        self.reward_extracting = reward_extracting
        self.reward_else = reward_else
        self.reward_illegal = reward_illegal
        self.seed = seed
        self.taxi_imgs = None
        self.taxi_orientation = 0
        self.passenger_img = None
        self.destination_img = None
        self.median_horiz = None
        self.median_vert = None
        self.background_img = None
        self.black_image = None
        self.move = None
        self.rewards_each_step = None
        self.agent_move_reward = []

        self.picked_victims = []
        self.extracted_victims = []

        # self.agent_id_new = []
        # self.victim_new = []

        # Set up action and observation spaces
        self.action_space = Discrete(self.nb_actions ** self.nb_agents)
        self.observation_space = MultiDiscrete([len(self.locs)] * 2 * self.nb_resources + [self.gridsize] * 2 * self.nb_agents)

        # Set random seed for testing
        np.random.seed(seed)
        
        # Randomize starting coordinates [x,y] for agents and resources
        self.state_agent = np.random.randint(self.gridsize, size=(self.nb_agents, 2))
        logging.info(self.state_agent)

         #np.random.randint(len(self.locs) + 1, size=(self.nb_resources, 2))
        xy = np.array([(a, b) for a, b in [x for i,x in enumerate(self.locs) if x!=tuple(self.dest_loc)]])
        random_indices = np.random.choice(len(xy), self.nb_resources, replace=False)
        self.state_victims = xy[random_indices]
        logging.info(self.state_victims)

        # Map action space to the base of possible actions, so that every digit corresponds to the action of one agent
        self.action_map = {}
        for i in range(self.action_space.n):
            # Change i to base 5 (possible actions)
            action = [0] * self.nb_agents
            num = i
            index = -1
            while num > 0:
                action[index] = num % self.nb_actions
                num = num // self.nb_actions
                index -= 1
            self.action_map[i] = action
        #logging.info(self.action_map)

   # Step function
    def step(self, action: int):
        reward = 0
        self.agent_move_reward = []
        xy = np.array([(a, b) for a, b in self.locs])

        terminated = False
        time.sleep(0.1)
        # logging.info(self.picked_victims)
        logging.info(f"Step no: {self.step_nb}")

        # Update position of each agent according to action map and grid boundaries
        for i, action in enumerate(self.action_map[action]):
            # logging.info(i)
            
            # logging.info(f"agent no: {i}, {self.state_agent}")
            # logging.info(self.step_nb)
            # logging.info("Victim and Agent state before it al starts agent and victim", self.state_agent, self.state_victims)
            # Check whether any agent is positioned upon a victim
            if action == UP:
                self.move = "UP"
                # .append("UP")
                logging.info(f"agent_id {i}: Moved up")
                #logging.info(f"{self.step_nb}, agent_id {i}: Moved up")
                self.state_agent[i, 1] = self.state_agent[i, 1] - 1 if self.state_agent[i, 1] > 0 else 0
                # Add default reward to total reward and create temporary list for extracted resources
                reward += self.reward_else
                # logging.info(self.state_agent[i, :])
                # logging.info(self.state_victims)

            elif action == RIGHT:
                self.move = "RIGHT"
                logging.info(f"agent_id {i}: Moved right")
                #logging.info(f"{self.step_nb} , agent_id {i}: Moved right")
                self.state_agent[i, 0] = self.state_agent[i, 0] + 1 if self.state_agent[
                                                                           i, 0] < self.gridsize - 1 else self.gridsize - 1
                # Add default reward to total reward and create temporary list for extracted resources
                reward += self.reward_else
                # logging.info(self.state_agent[i, :])
                # logging.info(self.state_victims)
                
            elif action == DOWN:
                self.move = "DOWN"
                # self.move.append("DOWN")
                #logging.info(f"{self.step_nb} , agent_id {i}: Moved down")
                logging.info(f"agent_id {i}: Moved down")
                self.state_agent[i, 1] = self.state_agent[i, 1] + 1 if self.state_agent[
                                                                           i, 1] < self.gridsize - 1 else self.gridsize - 1
                # Add default reward to total reward and create temporary list for extracted resources
                reward += self.reward_else
                # logging.info(self.state_agent[i, :])
                # logging.info(self.state_victims)

            elif action == LEFT:
                self.move = "LEFT"
                # self.move.append("LEFT")
                #logging.info(f"{self.step_nb} , agent_id {i}: Moved left")
                logging.info(f"agent_id {i}: Moved left")
                self.state_agent[i, 0] = self.state_agent[i, 0] - 1 if self.state_agent[i, 0] > 0 else 0
                # Add default reward to total reward and create temporary list for extracted resources
                reward += self.reward_else
                # logging.info(self.state_agent[i, :])
                # logging.info(self.state_victims)

            elif action == PICKUP:  # Agent takes action pick up
                self.move = "PICKUP"
                # self.move.append("PICKUP")
                # logging.info the strp number and victim and agent and agent id
                # logging.info(f"{self.step_nb}, agent_id {i}: pick up")
                logging.info(f"agent_id {i}: pick up")
                # logging.info the strp number and victim and agent and agent id
                # logging.info(f"{self.step_nb}, agent_id {i}: pick up")

                # logging.info(f"Agent and victim location before picking : {i} , {self.state_agent[i, :]}, {self.state_victims}")

                # If any of the agent is at the same position as the victim
                if any((self.state_victims[:] == self.state_agent[i, :]).all(1)):

                    # find the index of the victim location same as the agent and find the index of that victim from the victim states
                    idx = [(x) for x, value in enumerate(self.state_victims) if np.all(value == self.state_agent[i, :])][0]
                    logging.info(f"Current picked victims : {self.picked_victims}")
                    logging.info(f"Victim being picked: {self.state_victims[idx, :]}")

                    # Loop through the picked_victims list to check if the victim has already been picked up
                    if self.picked_victims:
                        victim_locations = []  # Create empty list to store the victim locations from picked_list

                        # Store the victim locations in a list
                        for ids, agent_victim in enumerate(self.picked_victims):
                            #logging.info(agent_victim[1])
                            #logging.info(self.state_victims[idx, :])
                            victim_locations.append(agent_victim[1])
                        #logging.info(victim_locations)

                        # Add another clause that the victim has not been previously picked up
                        if (self.state_victims[idx, :].tolist() not in victim_locations):
                            # Store the values of the agent id and the picked up victim
                            picked_up_victim_agent_id = [i, self.state_victims[idx, :].tolist()]
                            # logging.info(picked_up_victim_agent_id)
                            add_element_and_get_index(picked_up_victim_agent_id, self.picked_victims)
                            # logging.info(self.picked_victims)
                            logging.info(f"Agent and victim location after picking up : {i}, {self.state_agent[i, :]}, {self.state_victims[idx,:]}")
                    
                    else:  # The picked_victims list is empty
                        
                        # Store the values of the agent id and the picked up victim
                        picked_up_victim_agent_id = [i, self.state_victims[idx, :].tolist()]
                        add_element_and_get_index(picked_up_victim_agent_id, self.picked_victims)
                        # logging.info(self.picked_victims)
                        logging.info(f"Agent and victim location after picking up : {i}, {self.state_agent[i, :]}, {self.state_victims[idx,:]}")

                else:  # passenger not at location
                    reward += self.reward_illegal


            elif action == DROP:  # Agent picks the drop off action
                self.move = "DROP"
                # self.move.append("DROP")
                # logging.info(self.picked_victims)

                # logging.info the step, agent id and victims
                # logging.info(f"{self.step_nb} , agent_id {i}: drop off")
                logging.info(f"agent_id {i}: drop off")
                # logging.info the step, agent id and victims
                # logging.info(f"{self.step_nb} , agent_id {i}: drop off")

                # logging.info(f"agent no: {i}, {self.state_agent[i, :]}")
                # logging.info(self.state_victims)
                
                # if self.picked_victims:
                #     logging.info(self.picked_victims)
                #     for ids, agent_victim in enumerate(self.picked_victims):
                #         # Enumerate through the picked victims and store the agent_id and victim location
                #         add_element_and_get_index(agent_victim[0], self.agent_id_new)
                #         logging.info(self.agent_id_new)

                #         add_element_and_get_index(agent_victim[1], self.victim_new)
                #         logging.info(self.victim_new)
                # else:
                #     continue

                # Check if the agent is at the destination location and if the agent id of the extracted victim is
                # the same as this id, if yes then the agent gets a reward as it has dropped the victim
                if np.all(self.state_agent[i,:] == self.dest_loc):

                    # Loop through the picked victims list to find if the agent id is the same as the one doing the action
                    if self.picked_victims:

                        # Enumerate through the picked victims to extract the location of the victim
                        for ids, agent_victim in enumerate(self.picked_victims):
                            if i == ids:  # If the picked up agent is the same as the one in the loop right now
                                idx = [(x) for x, value in enumerate(self.state_victims) if np.all(value == agent_victim[1])][0]
                                # logging.info(agent_victim[1])
                                # logging.info(self.extracted_victims)
                                if self.extracted_victims:  # If extracted list is not none
                                    if ((self.state_victims[idx, :]).tolist() not in self.extracted_victims):  # if the victim is not already extracted
                                        # logging.info("-----------------\n")
                                        # logging.info(f"Victim state after drop action initiated {j}, {vicself.state_victimstim}")
                                        # logging.info(f"Agent state after drop action initiated {agent}")
                            
                                        logging.info(f"Victim {self.state_victims[idx, :]} dropped off")
                                        # change the state of the victim to the same as the destination
                                        # self.state_victims[idx, :] = self.dest_loc

                                        # Append the victim to the extracted list
                                        add_element_and_get_index(self.state_victims[idx, :].tolist(), self.extracted_victims)
                                        reward += self.reward_extracting
                                        logging.info(self.extracted_victims)
                                else:  # if the extracted list is empty
                                    # logging.info("-----------------\n")
                                    # logging.info(f"Victim state after drop action initiated {j}, {vicself.state_victimstim}")
                                    # logging.info(f"Agent state after drop action initiated {agent}")
                        
                                    logging.info(f"Victim {self.state_victims[idx, :]} dropped off")
                                    # change the state of the victim to the same as the destination
                                    # self.state_victims[idx, :] = self.dest_loc

                                    # Append the victim to the extracted list
                                    add_element_and_get_index(self.state_victims[idx, :].tolist(), self.extracted_victims)
                                    reward += self.reward_extracting
                    # logging.info(self.extracted_victims)
                    # logging.info(len(self.extracted_victims))
                    #logging.info(self.victim_isactive())
                    if self.victim_isactive() == False:
                        logging.info(f"Are both victims still in the danger zone: {self.victim_isactive()}")
                        terminated = True
                        reward += self.reward_extracting + 10
                        break

                # else if the agent has picked up but not found the destination as yet
                elif np.all(self.state_agent[i,:] != self.dest_loc):
                     # find the index of the location same as the victim from the extracted list 
                    if self.picked_victims:
                        # Enumerate through the picked victims to extract the location of the victim
                        for ids, agent_victim in enumerate(self.picked_victims):
                            if i == ids:  # If the picked up agent is the same as the one in the loop right now
                                # logging.info(self.state_agent[i, :])
                                # logging.info(i)
                                # logging.info(self.state_victims)
                                logging.info(f"Agent has picked up but can't find drop")

                else:  # dropoff at wrong location or # passenger not at location
                    # agent just drops off the victim at a wrong location and give it a -ve reward
                    logging.info(self.state_agent[i, :])
                    logging.info(i)
                    logging.info(self.state_victims)
                    reward += self.reward_illegal
                    logging.info(f"Agent has not picked up neither has found the drop off location")
            
            self.rewards_each_step = reward
            agent_move_reward = [self.state_agent[i], self.move, self.rewards_each_step]
            self.agent_move_reward.append(agent_move_reward)

        # if self.extracted_victims:
        #     logging.info(self.extracted_victims)
        #     # If the victim is in the extracted list remove it from the original states.
        #     for victim_dropped in self.extracted_victims:
        #         logging.info(victim_dropped)
        #         if any((self.state_victims[:] == victim_dropped).all(1)):
        #             self.state_victims = np.all(self.state_victims[:].tolist().remove(victim_dropped))

        # Increase step by 1
        self.step_nb += 1
        
        done = False
        # If episode step limit is reached, finish episode
        if self.step_nb == self.nb_steps or terminated == True:
            logging.info(terminated)
            done = True
        info = {}
        
        # self.render()
        observation = self.observe()

        if self.debug:
            logging.info("Reward:", reward)
            logging.info('Observation:', observation)

        return observation, reward, done, info

    
    def victim_isactive(self):
        if len(self.extracted_victims) != len(self.state_victims):
            return True
        else:
            return False
        
    # Environment reset function
    def reset(self):
        logging.info(f"number of steps completed: {self.step_nb}")
        logging.info("Resetting the whole environment")
        self.extracted_victims = []
        self.picked_victims = []
        self.agent_id_new = []
        self.victim_new = []
        # Randomize starting coordinates [x,y] for agents and resources
        self.state_agent = np.random.randint(self.gridsize, size=(self.nb_agents, 2))
        xy = np.array([(a, b) for a, b in [x for i,x in enumerate(self.locs) if x!=tuple(self.dest_loc)]])
        random_indices = np.random.choice(len(xy), self.nb_resources, replace=False)
        self.state_victims = xy[random_indices]
        # Reset step number
        self.step_nb = 0
        return self.observe()

    # Graphics rendering function
    def render(self, mode='human'):
        try:
            import pygame  # dependency to pygame only if rendering with human
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[toy_text]`"
            )

        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Resource Extraction Game")
            if mode == "human":
                self.window = pygame.display.set_mode((WINDOW_PIXELS, WINDOW_PIXELS))
            elif mode == "rgb_array":
                self.window = pygame.Surface((WINDOW_PIXELS, WINDOW_PIXELS))

        assert (
            self.window is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.taxi_imgs is None:
            file_name = path.join(path.dirname(__file__), "Rescue-Scheduler/images/superman.jpeg") 
            self.taxi_img = pygame.transform.scale(
                pygame.image.load(file_name), (self.cell_pixels,self.cell_pixels)
                )

        if self.passenger_img is None:
            file_name = path.join(path.dirname(__file__), "Rescue-Scheduler/images/passenger.png")
            self.passenger_img = pygame.transform.scale(
                pygame.image.load(file_name), (self.cell_pixels, self.cell_pixels)
            )
        if self.destination_img is None:
            file_name = path.join(path.dirname(__file__), "Rescue-Scheduler/images/hotel.png")
            self.destination_img = pygame.transform.scale(
                pygame.image.load(file_name), (self.cell_pixels, self.cell_pixels)
            )
            self.destination_img.set_alpha(170)
        if self.median_horiz is None:
            file_names = [
                path.join(path.dirname(__file__), "Rescue-Scheduler/images/gridworld_median_left.png"),
                path.join(path.dirname(__file__), "Rescue-Scheduler/images/gridworld_median_horiz.png"),
                path.join(path.dirname(__file__), "Rescue-Scheduler/images/gridworld_median_right.png"),
            ]
            self.median_horiz = [
                pygame.transform.scale(pygame.image.load(file_name), (self.cell_pixels,self.cell_pixels))
                for file_name in file_names
            ]
        if self.median_vert is None:
            file_names = [
                path.join(path.dirname(__file__), "Rescue-Scheduler/images/gridworld_median_top.png"),
                path.join(path.dirname(__file__), "Rescue-Scheduler/images/gridworld_median_vert.png"),
                path.join(path.dirname(__file__), "Rescue-Scheduler/images/gridworld_median_bottom.png"),
            ]
            self.median_vert = [
                pygame.transform.scale(pygame.image.load(file_name), (self.cell_pixels,self.cell_pixels))
                for file_name in file_names
            ]
        if self.black_image is None:
            file_name = path.join(path.dirname(__file__), "Rescue-Scheduler/images/Solid_black.svg.png")
            self.black_image = pygame.transform.scale(
                pygame.image.load(file_name), (self.cell_pixels*2,self.cell_pixels*4)
            )
        if self.background_img is None:
            file_name = path.join(path.dirname(__file__), "Rescue-Scheduler/images/taxi_background.png")
            self.background_img = pygame.transform.scale(
                pygame.image.load(file_name), (self.cell_pixels,self.cell_pixels)
            )
        
        for y in range(0, self.gridsize):
            for x in range(0, self.gridsize):
                cell = (x * self.cell_pixels, y * self.cell_pixels)
                self.window.blit(self.background_img, cell)
        
        for y in range(self.gridsize, self.gridsize+2):
            for x in range(0, self.gridsize):
                cell = (x * self.cell_pixels, y * self.cell_pixels)
                self.window.blit(self.black_image, cell)

        self.window.blit(self.destination_img, (4 * self.cell_pixels, 0 * self.cell_pixels))

        font = pygame.font.SysFont(None, 30)
        step_img = font.render(f"Step No: {self.step_nb}", True, WHITE)
        self.window.blit(step_img, (800, 1050))
            
        # # Draw victims
        # for victim in (self.state_victims).tolist():
        #     # Draw agents as blue circles
        #     if self.picked_victims:
        #         print(self.state_victims)
        #         for ids, agent_victim in enumerate(self.picked_victims):
        #             print(agent_victim[1])
        #             if np.all(victim == agent_victim[1]):    

        #                 self.window.blit(self.background_img, (victim[0] * self.cell_pixels, victim[1] * self.cell_pixels))
        #                 logging.info(f"Victim: {victim} picked")
        #                 vic_img = font.render(f"Victim: {victim} extracted", True, WHITE)
        #                 self.window.blit(vic_img, (50, 1130))
        #             else:
        #                 self.window.blit(self.passenger_img, (victim[0] * self.cell_pixels, victim[1] * self.cell_pixels))
        #     else:
        #         self.window.blit(self.passenger_img, (victim[0] * self.cell_pixels, victim[1] * self.cell_pixels))
        
        for victim in (self.state_victims).tolist():
            if self.extracted_victims:
                if np.all(victim in self.extracted_victims):
                    self.window.blit(self.passenger_img, (4 * self.cell_pixels, 0 * self.cell_pixels))
                    logging.info(f"Victim: {victim} extracted")
                    self.window.blit(self.background_img, (victim[0] * self.cell_pixels, victim[1] * self.cell_pixels))
                    logging.info(f"Victim: {victim} extracted")
                    vic_img = font.render(f"Victim: {victim} extracted", True, WHITE)
                    self.window.blit(vic_img, (500, 1140))
            elif self.picked_victims:
                for ids, agent_victim in enumerate(self.picked_victims):
                    if np.any(victim == agent_victim[1]):    
                            self.window.blit(self.background_img, (victim[0] * self.cell_pixels, victim[1] * self.cell_pixels))
                            logging.info(f"Victim: {victim} picked")
                            self.window.blit(self.black_image, (50, 1140))
                            vic_img = font.render(f"Victim: {victim} picked", True, WHITE)
                            self.window.blit(vic_img, (50, 1140))
                    else:
                        self.window.blit(self.passenger_img, (victim[0] * self.cell_pixels, victim[1] * self.cell_pixels))
            else:
                self.window.blit(self.passenger_img, (victim[0] * self.cell_pixels, victim[1] * self.cell_pixels))
                    
        
        img = font.render("Movement of the Rescuer (Agent)", True, WHITE)
        self.window.blit(img, (50, 1050))
        pygame.display.flip()
        # logging.info(self.agent_move_reward)
        if self.agent_move_reward and len(self.agent_move_reward) > 1:
            agent_1 = self.agent_move_reward[0][0]
            move_1 = self.agent_move_reward[0][1]
            reward_1 = self.agent_move_reward[0][2]

            self.window.blit(self.taxi_img, (agent_1[0] * self.cell_pixels, agent_1[1] * self.cell_pixels))


            img0 = font.render(f"Agent 1 action is: {move_1}", True, WHITE)
            self.window.blit(img0, (50, 1075))
            
            img02 = font.render(f"Agent 1 current reward is: {reward_1}", True, WHITE)
            self.window.blit(img02, (50, 1105)) 
            

            agent_2 = self.agent_move_reward[1][0]
            move_2 = self.agent_move_reward[1][1]
            reward_2 = self.agent_move_reward[1][2]

            self.window.blit(self.taxi_img, (agent_2[0] * self.cell_pixels, agent_2[1] * self.cell_pixels))
            img0 = font.render(f"Agent 2 action is: {move_2}", True, WHITE)
            self.window.blit(img0, (500, 1075))
              
            
            img02 = font.render(f"Agent 2 current reward is: {reward_2}", True, WHITE)
            self.window.blit(img02, (500, 1105))

        elif self.agent_move_reward and len(self.agent_move_reward) > 1:
            agent_1 = self.agent_move_reward[0][0]
            move_1 = self.agent_move_reward[0][1]
            reward_1 = self.agent_move_reward[0][2]

            self.window.blit(self.taxi_img, (agent_1[0] * self.cell_pixels, agent_1[1] * self.cell_pixels))


            img0 = font.render(f"Agent 1 action is: {move_1}", True, WHITE)
            self.window.blit(img0, (50, 1075))
            
            img02 = font.render(f"Agent 1 current reward is: {reward_1}", True, WHITE)
            self.window.blit(img02, (50, 1105))

        else:
            for agent in self.state_agent:
                self.window.blit(self.taxi_img, (agent[0] * self.cell_pixels, agent[1] * self.cell_pixels))
                img0 = font.render(f"Reset Environment", True, WHITE)
                self.window.blit(img0, (50, 1075))
        

        # Discard old frames and show the last one
        pygame.display.flip()
        time.sleep(0.1)
        pygame.image.save(self.window, f"/home/shinde/Documents/trainings/Spiced_Academy/Github/tahini-tensor-student-code/spiced_projects/WEEK_12/train_20_epi_500_steps_2_vic_part_2/screenshot_{self.step_nb}.jpeg")
        time.sleep(0.1)

    # Observation generation function
    def observe(self):
        return np.concatenate((self.state_victims.flatten(),
                               self.state_agent.flatten()))

    def close(self):
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
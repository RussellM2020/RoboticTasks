from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from .mujoco_env import MujocoEnv
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc import autoargs
import pickle




class Gripper(MujocoEnv, Serializable):

    FILE = 'gripperFree.xml'

    def __init__(self):
    
        super(Gripper, self).__init__()

   
    

    def get_current_obs(self): 
        


        return np.concatenate([ self.model.data.qpos.flat[:], self.model.data.qvel.flat[:]]).reshape(-1)

    #def reset(self, resample_block=True):
    def reset(self, init_state=None):
    
        self.model.data.qpos = self.init_qpos 
        self.model.data.qvel = np.zeros_like(self.model.data.qvel)
        self.model.data.qacc = self.init_qacc
        self.model.data.ctrl = self.init_ctrl
        
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)

        return self.get_current_obs()

        

   
   
   
   



    def step(self, action):
         
      
        action[-1] = -action[-2]
     
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()


        
        rightFinger, leftFinger = self.model.data.site_xpos[0], self.model.data.site_xpos[1]
        rightBlockPatch, leftBlockPatch = self.model.data.site_xpos[2], self.model.data.site_xpos[3]
        
        objPos = self.get_body_com("block")
        

        leftDist = np.linalg.norm(leftBlockPatch - leftFinger)
        rightDist = np.linalg.norm(rightBlockPatch - rightFinger)


        reachRew = -leftDist - rightDist
        pickRew = 0

        if (objPos[2]> 0.025) :
            pickRew+=100* objPos[2]

        # if ((objPos[2]> 0.025) and (leftDist<0.01) and (rightDist <0.01)):
        #     pickRew+=500*objPos[2]
        
        reward = reachRew + pickRew

        done = False
      
        return Step(next_obs, reward, done)


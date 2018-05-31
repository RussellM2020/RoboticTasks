from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from .mujoco_env import MujocoEnv
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc import autoargs
import pickle




class Gripper(MujocoEnv, Serializable):

    FILE = 'gripper.xml'

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

    def viewer_setup(self):
       
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 1.0

        # robot view
        # rotation_angle = 90
        # cam_dist = 1
        # cam_pos = np.array([0, 0.5, 0.2, cam_dist, -45, rotation_angle])

        # 3rd person view
        cam_dist = 0.5
        rotation_angle = 0
        cam_pos = np.array([0, 0, 0.05, cam_dist, 0, rotation_angle])

        # top down view
        # cam_dist = 0.2
        # rotation_angle = 0
        # cam_pos = np.array([0, 0, 1.5, cam_dist, -90, rotation_angle])

        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid = -1
 



    def step(self, action):
         
        
        gripperPos = self.get_body_com("gripper")
        
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
            pickRew+=100*objPos[2]


       
        # if ((objPos[2]> 0.025) and (leftDist<0.1) and (rightDist <0.1)):
        #      pickRew+=500*objPos[2]
        
        reward = reachRew + pickRew

        done = False
      
        return Step(next_obs, reward, done)


# -*- coding: utf-8 -*-

import cv2

from time import time, sleep

import imageio


class EnvPlayer:
  def __init__(self, env, agent=None):
    self.env = env
    if 'rgb_array' not in env.metadata['render.modes']:
      raise ValueError("Env {} does not support rgb rendering!")
    self.agent = agent
    self.done = True
    if self.agent:
      if hasattr(self.agent, "name"):
        self.agent_name = self.agent.name
      else:
        self.agent_name = 'X'
    else:
      self.agent_name = 'random'
    self.win_name = 'Gym_Env_Player_Agent_{}'.format(self.agent_name)
    return
      
  def _get_next_frame(self):
    if self.done:
      self.state = self.env.reset()    
      self.done = False
    else:
      if self.agent:
        act = self.agent.act(self.state)
      else:
        act = self.env.action_space.sample()
      obs, r, done, info = self.env.step(act)
      self.done = done    
      self.state = obs
      self.reward =r
      self.last_action = act
    np_frm_rgb = self.env.render(mode='rgb_array')   
    np_frm = cv2.cvtColor(np_frm_rgb, cv2.COLOR_RGB2BGR)
    return np_frm
  
  def _start_video(self):
    cv2.namedWindow(self.win_name)
    cv2.moveWindow(self.win_name, 1, 1) 
    return
      
  def _end_video(self):
    cv2.destroyAllWindows()
    self.env.close()
    return
  
  def _show_message(self, np_img, _text):
    top, left, w, h = cv2.getWindowImageRect(self.win_name)
    font                   = cv2.FONT_HERSHEY_TRIPLEX
    bottomLeftCornerOfText = (h // 2, w //2 - w // 4)
    fontScale              = 1
    fontColor              = (255,0,0)
    cv2.putText(
        img=np_img, 
        text=_text, 
        org=bottomLeftCornerOfText, 
        fontFace=font, 
        fontScale=fontScale,
        color=fontColor)
    return np_img
  
  def _quit_requested(self):
    key = cv2.waitKey(1) & 0xFF
    if (key == ord('q')) or (key == ord('Q')):
      return True
    else:
      return False
  
  def play(self, cont=True, sleep_time=0.05, save_gif=None):
    buff_frames = []
    self._start_video()
    while True:    
      out_frame = self._get_next_frame()
      cv2.imshow(self.win_name,out_frame)
      buff_frames.append(cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB))
      if self._quit_requested(): break
      if sleep_time:
        sleep(sleep_time)
      if self._quit_requested(): break
      if self.done:
        out_frame = self._show_message(out_frame, "EPISODE DONE")
        cv2.imshow(self.win_name,out_frame)
        if self._quit_requested(): break
        sleep(2)
        if self._quit_requested(): break
        if not cont:
          break
    self._end_video()
    if save_gif:
      imageio.mimsave(save_gif, buff_frames)
    return


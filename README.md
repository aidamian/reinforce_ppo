# REINFOCE & PPO PyTorch implementation 

### Test #1

`reinforce_test1.py` uses `CartPole-v0` environment for basic REINFORCE. Included are grad calculation examples.
Numeber of steps until a solution is found:    
  Simple G calculation: 1353.0
  Normed disc rewards:   352.0

![cart_normed_disc_rewards](https://github.com/andreidi/pytorch_reinforce_cart/blob/master/cart_reinforce.gif)



### Test #2

`reinforce_test2.py` uses either vanilla *REINFORCE* or *PPO PG* to solve pong. Below are the comparision results for both methods:

#### PPO

For the *PPO PG* the policy manages to win some (almost constantly) at episode 350
```
Episode: 350, score:    1.2
  Workers: [-1.  1.  1.  1.  1.  3.  3.  1.]
Episode: 355, score:    0.5
  Workers: [ 3.  3. -3.  0.  0.  0. -2.  3.]
Episode: 360, score:    1.5
  Workers: [ 3.  3.  3.  1. -2. -2.  3.  3.]
Episode: 365, score:    0.1
  Workers: [ 0. -1.  0.  3.  0. -2.  1.  0.]
Episode: 370, score:    2.5
  Workers: [5. 1. 3. 3. 1. 3. 1. 3.]
```

and finally around episode  the results are showing good and consistent policy

```
Episode: 500, score:    4.8
  Workers: [3. 5. 5. 5. 5. 5. 5. 5.]
Episode: 505, score:    5.0
  Workers: [5. 5. 5. 5. 5. 5. 5. 5.]
Episode: 510, score:    5.0
  Workers: [5. 5. 5. 5. 5. 5. 5. 5.]
```
![ppo](https://github.com/andreidi/pytorch_reinforce_cart/blob/master/ppo_results.png)

![ppo](https://github.com/andreidi/pytorch_reinforce_cart/blob/master/PPO_play_test.gif)

#### REINFORCE

in contrast the vanilla *REINFORCE* version does not perform well even after 800 episodes

![ppo](https://github.com/andreidi/pytorch_reinforce_cart/blob/master/REINFORCE_8_workers.png)

![ppo](https://github.com/andreidi/pytorch_reinforce_cart/blob/master/REINFORCE_play_test.gif)

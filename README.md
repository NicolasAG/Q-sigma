# Q-sigma
RL - Implementation of n-step SARSA, n-step TreeBackup and n-step Q-sigma in a simple 10x10 grid world

## Description
Start in `S`, move `V` steps at a time (where `V` is the velocity) in some direction.

If you hit a wall `X`, go back to `S` and `V` is reset to 0

### Grid world - Race track:
```
_ | _ | _ | _ | _ | _ | _ | _ | _ | G |
_ | X | X | X | X | X | X | X | X | X |
_ | _ | _ | _ | _ | _ | _ | _ | _ | _ |
_ | _ | _ | _ | _ | _ | _ | _ | _ | _ |
_ | _ | _ | _ | _ | _ | _ | _ | _ | _ |
X | _ | _ | _ | _ | _ | _ | _ | _ | _ |
_ | _ | _ | # | _ | _ | _ | _ | _ | _ |
_ | _ | _ | _ | _ | _ | _ | _ | _ | _ |
_ | _ | _ | _ | _ | _ | _ | _ | _ | _ |
S | _ | _ | _ | _ | _ | _ | _ | _ | _ |
```
### Actions:
- move RIGHT & Velocity -1 -- 0
- move RIGHT & Velocity +0 -- 1
- move RIGHT & Velocity +1 -- 2
- move UP & Velocity -1 -- 3
- move UP & Velocity +0 -- 4
- move UP & Velocity +1 -- 5
- move LEFT & Velocity -1 -- 6
- move LEFT & Velocity +0 -- 7
- move LEFT & Velocity +1 -- 8

### Rewards:
- GOAL = +1000
- WALL = -10
- STEP = -1
- CHECK POINT = 0

## Run
`python main.py <algo> <flags>`

Algo:
- "SARSA" (n-step)
- "TreeBackup" (n-step)
- "Qsigma" (n-step)
  - q_mode = "rnd": 
  - q_mode = "alt":
  - q_mode = "inc":
  - q_mode = "dec":

Flags:
- `-ne #` number of episodes to train for (default=1000)
- `-ns #` number of steps to look forward: it's the n in n-step algorithms (default=5)
- `-g #` gamma, discount factor (default=0.99)
- `-a #` alpha, learning rate (default=0.10)
- `-e #` epsilon, policy stochasticity, proba of chosing a random non-greddy action (default=0.10)
- `-b #` beta, environement stochasticity, proba of not updating `V` no matter the action (default=0.00)
- `-q_mode <>` mode of Qsigma (default=alt)
- `base #` log base to use in Qsigma-inc or Qsigma-dec

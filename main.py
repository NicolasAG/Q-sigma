#!/usr/bin/env python

import numpy as np
import argparse
import math
from datetime import datetime as dt

WORLD = np.array([
    ["_", "_", "_", "_", "_", "_", "_", "_", "_", "G"],
    ["_", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
    ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
    ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
    ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
    ["X", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
    ["_", "_", "_", "_", "#", "_", "_", "_", "_", "_"],
    ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
    ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
    ["S", "_", "_", "_", "_", "_", "_", "_", "_", "_"]
])
STATES = range(WORLD.size)  # 1D array from 0 to 28
WIDTH = 10
grid = np.indices((WIDTH, WIDTH))
STATE2WORLD = [(x, y) for x, y in zip(grid[0].flatten(), grid[1].flatten())]
START = 90  # state index of start
CHECKPNT = 64  # state index of check point
GOAL = 9  # state index of goal
WALLS = [  # state index of walls
    11, 12, 13, 14, 15, 16, 17, 18, 19, 50
]
V = 0  # Velocity
V_MAX = 3
V_MIN = 0
# Set of actions: _X=V-1 ; X=V+0 ; X_=V+1
_RIGHT = 0; RIGHT = 1; RIGHT_ = 2
_UP = 3; UP = 4; UP_ = 5
_LEFT = 6; LEFT = 7; LEFT_ = 8
ACTIONS = range(9)

CRASH = -10.  # reward for hitting a wall
CHECK = 0.  # reward for reaching the checkpoint
WIN = 1000.  # reward for reaching the goal
STEP = -1.  # reward for moving

PI = np.zeros((len(STATES), len(ACTIONS)))  # policy: <state, action> -> <float>
Q = np.zeros((len(STATES), len(ACTIONS)))  # <state, action> -> <float>


def reset():
    """
    reset grid world and velocities
    """
    global WORLD
    WORLD = np.array([
        ["_", "_", "_", "_", "_", "_", "_", "_", "_", "G"],
        ["_", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
        ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
        ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
        ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
        ["X", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
        ["_", "_", "_", "_", "#", "_", "_", "_", "_", "_"],
        ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
        ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
        ["S", "_", "_", "_", "_", "_", "_", "_", "_", "_"]
    ])
    global V, V_MAX
    V = 0
    V_MAX = 3


def make_greedy(s, epsilon):
    """
    Make PI(s,:) greedy according to Q(s,:) for all actions for a given state s
    :param s: given state
    :param epsilon: probability of choosing a non-optimal action
    """
    # action probabilities = epsilon / (|A|-1) for all actions by default
    # over |A|-1 because 1 of them will be optimal and have probability 1-epsilon
    global PI
    PI[s, :] = [epsilon / (len(ACTIONS) - 1.)] * len(ACTIONS)

    # Get the best action for that state (greedy w.r.t. Q):
    best_a = 0
    best_q_val = -np.inf
    for i, q_val in enumerate(Q[s, :]):
        if q_val > best_q_val:
            best_q_val = q_val
            best_a = i

    # Change default probability of best action to be 1-epsilon
    PI[s, best_a] = 1. - epsilon
    # print "best action:", best_a
    assert np.isclose(np.sum(PI[s, :]), 1.)


def choose_action(s, epsilon):
    """
    Choose an action from state s according to epsilon-greedy policy
    :param s: current state
    :param epsilon: probability of choosing a non-optimal action
    :return: action to take from s
    """
    make_greedy(s, epsilon)
    return np.random.choice(ACTIONS, p=PI[s, :])  # sample from ACTIONS with proba distribution PI[s, :]


def choose_sigma(method, q_mode, last_sigma=None, base=None, t=None):
    """
    Return a sigma for the Q(sigma) algorithm according to a given method
    :param method: the algorithm to follow: SARSA, or TreeBackup, or Qsigma
    :param q_mode: Qsigma mode (only if method='Qsigma'): random, alternative, decreasing, or increasing mode
    :param last_sigma: the previous sigma returned by this function (only used for Qsigma in alternating mode)
    :param base: base of the logarithm to take (only used in non-alternating mode)
    :param t: current time step (only used for Qsigma in non-alternating mode)
    :return: 1 for SARSA, 0 for TreeBackup, 1 with probability p for Qsigma (in non-alternating mode)
    """
    if method == "SARSA":
        return 1
    elif method == "TreeBackup":
        return 0
    elif method == "Qsigma":
        if q_mode == "rnd":  # RANDOM mode
            return 1 if np.random.random() < 0.5 else 0
        elif q_mode == "alt":  # ALTERNATING mode
            assert last_sigma in [0, 1]
            return 1 - last_sigma
        elif q_mode == "inc":  # INCREASING probability mode
            assert base >= 3
            assert t >= 0
            sample_proba = 1 - math.exp(-math.log(1+t, base))  # increases with t
            # print "t =", t, "& P(sig=1) =", sample_proba
            return 1 if np.random.random() < sample_proba else 0
        elif q_mode == "dec":  # DECREASING probability mode
            assert base >= 3
            assert t >= 0
            sample_proba = math.exp(-math.log(1+t, base))  # decreases with t
            # print "t =", t, "& P(sig=1) =", sample_proba
            return 1 if np.random.random() < sample_proba else 0
        else:
            print "ERROR: use Qsigma but no mode specified: random, alternating, increasing or decreasing?"
            return None
    else:
        print "ERROR: unknown method", method
        return None


def move(s, a, beta):
    """
    Perform action a in state s, and observe r in s'
    :param s: current state
    :param a: action to take from state s
    :param beta: proba of no velocity update (environment stochasticity)
    :return: next state and observed reward
    """
    # update velocity with probability 1-beta
    global V, V_MAX
    if np.random.random() < 1-beta:
        if a in [_RIGHT, _UP, _LEFT] and V > V_MIN:
            V -= 1
        elif a in [RIGHT_, UP_, LEFT_] and V < V_MAX:
            V += 1
    # else:
    #     print "velocity not updated!"

    r_border = range(WIDTH-1, WIDTH**2, WIDTH)  # states on the right border
    l_border = range(0, WIDTH**2, WIDTH)  # states on the left border
    t_border = range(WIDTH)  # states on the top border

    units = range(V)
    check = False  # flag to indicate if we visited the checkpoint
    # move RIGHT of V units:
    if a < len(ACTIONS) / 3:
        for i in units:
            WORLD[STATE2WORLD[s+i]] = '>'  # draw my path gradualy in the world
            # crash: reset world and velocities, return to start state
            if s+i in r_border or s+i+1 in WALLS:
                reset()
                return START, CRASH
            # went through the checkpoint: increase V_MAX and return bonus (only the first time!)
            elif s+i+1 == CHECKPNT:
                check = V_MAX != 5
                V_MAX = 5
            # goal: draw where I end up & return
            elif s+i+1 == GOAL:
                WORLD[STATE2WORLD[s+i+1]] = 'O'
                return s+i+1, WIN
        # draw where I end up & return
        WORLD[STATE2WORLD[s+V]] = 'O'
        return (s+V, CHECK) if check else (s+V, STEP)

    # move UP of V units:
    elif a < 2*len(ACTIONS) / 3:
        for i in units:
            WORLD[STATE2WORLD[s-i*WIDTH]] = '|'  # draw my path gradualy in the world
            # crash: reset world and velocities, return to start state
            if s-i*WIDTH in t_border or s-(i+1)*WIDTH in WALLS:
                reset()
                return START, CRASH
            # went through the checkpoint: increase V_MAX and return bonus (only the first time!)
            elif s-(i+1)*WIDTH == CHECKPNT:
                check = V_MAX != 5
                V_MAX = 5
            # goal: draw where I end up & return
            elif s-(i+1)*WIDTH == GOAL:
                WORLD[STATE2WORLD[s-(i+1)*WIDTH]] = 'O'
                return s-(i+1)*WIDTH, WIN
        # nothing special: draw where I end up & return
        WORLD[STATE2WORLD[s-V*WIDTH]] = 'O'
        return (s-V*WIDTH, CHECK) if check else (s-V*WIDTH, STEP)

    # move LEFT of V units:
    elif a < len(ACTIONS):
        for i in units:
            WORLD[STATE2WORLD[s-i]] = '<'  # draw my path gradualy in the world
            # crash: reset world and velocities, return to start state
            if s-i in l_border or s-i-1 in WALLS:
                reset()
                return START, CRASH
            # went through the checkpoint: increase V_MAX and return bonus (only the first time!)
            elif s-i-1 == CHECKPNT:
                check = V_MAX != 5
                V_MAX = 5
            # goal: draw where I end up & return
            elif s-i-1 == GOAL:
                WORLD[STATE2WORLD[s-i-1]] = 'O'
                return s-i-1, WIN
        # draw where I end up & return
        WORLD[STATE2WORLD[s-V]] = 'O'
        return (s-V, CHECK) if check else (s-V, STEP)

    return s, STEP  # should never happen


def main():
    def my_float(x):  # Custom type for argparse arguments: gamma, alpha, epsilon, beta
        x = float(x)
        if x < 0.0 or x > 1.0:
            raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % x)
        return x

    def my_int(x):  # Custom type for argparse arguments: base
        x = int(x)
        if x < 3:
            raise argparse.ArgumentTypeError("%r lower than 3" % x)
        return x

    parser = argparse.ArgumentParser(description='MDP SARSA vs Expected SARSA.')
    parser.add_argument(
        'method',
        choices=["Qsigma", "SARSA", "TreeBackup"],
        help="The algorithm to use for solving a simple grid world MDP."
    )
    parser.add_argument(
        '-ne', '--n_episodes', type=int, default=1000,
        help="number of episodes to train the agent for."
    )
    parser.add_argument(
        '-ns', '--n_steps', type=int, default=5,
        help="number of steps to look forward: 0->TemporalDifference, infinity->MonteCarlo methods"
    )
    parser.add_argument(
        '-g', '--gamma', type=my_float, default=0.99,
        help="discount factor."
    )
    parser.add_argument(
        '-a', '--alpha', type=my_float, default=0.1,
        help="learning rate."
    )
    parser.add_argument(
        '-e', '--epsilon', type=my_float, default=0.1,
        help="epsilon in epsilon-greedy policy, ie: Stochasticity of policy."
    )
    parser.add_argument(
        '-b', '--beta', type=my_float, default=0.0,
        help="probability of no velocity update. ie: Stochasticity of environment."
    )
    parser.add_argument(
        '--q_mode', default="alt",
        choices=["rnd", "alt", "dec", "inc"],
        help="Qsigma mode - rnd: P(sigma=1)=0.5, alt:alternate sigmas, dec: P(sigma=1) decreases over time, inc: P(sigma=1) increases over time. (Only used if args.method='Qsigma')"
    )
    parser.add_argument(
        '--base', type=my_int, default=50,
        help="base of the logarithm used in probability of sampling in alternating Qsigma. Lower values = probability changing quickly; Higher values = probability changing slower."
    )
    args = parser.parse_args()
    print args

    K = 10

    average_steps = []  # average steps over all episodes
    average_reward = []  # average reward over all episodes
    for k in range(K):  # perform the experiment K times!

        global Q, PI  # restart learning!!
        PI = np.zeros((len(STATES), len(ACTIONS)))  # policy: <state, action> -> <float>
        Q = np.zeros((len(STATES), len(ACTIONS)))  # <state, action> -> <float>

        n_steps = []  # number of steps for each episode
        rewards = []  # total reward for each episode

        start = dt.now()
        ep = 0
        while ep < args.n_episodes:
            print "\nEpisode", ep+1, "/", args.n_episodes, "..."
            reset()  # reset grid world and velocities before the start of each episode.
            steps = 0  # keep track of the number of steps to finish an episode
            reward = 0  # keep track of the total reward for that episode

            states = []  # keep track of visited states
            actions = []  # keep track of actions taken
            q = []  # keep track of q_values: q[t] = Q[states[t], actions[t]]
            pi = []  # keep track of action probabilities: pi[t] = PI[states[t], actions[t]]
            sigmas = [1]  # keep track of selected sigmas
            targets = []  # keep track of the target rewards: 

            states.append(START)  # select and store starting state S_0
            a = choose_action(START, args.epsilon)  # select A_0 ~ S_0
            actions.append(a)  # store A_0
            q.append(Q[START, a])  # Q_0
            pi.append(PI[START, a])  # PI_0
            T = np.inf

            t = -1
            while True:
                t += 1
                # print "t =", t
                # print "T =", T
                # print "#of states =", len(states)
                assert len(actions) == len(q) == len(pi) == len(sigmas)
                # print "#of actions = size of q = size of pi = size of sigmas =", len(states)
                # print "#of targets =", len(targets)

                if t < T:
                    # print WORLD
                    # print "\nEpisode", ep+1, "/", args.n_episodes, "..."
                    # print "state:", states[t], "V:", V, "action:", actions[t]
                    s_next, r = move(states[t], actions[t], args.beta)  # take action A_t
                    states.append(s_next)  # store S_{t+1}
                    steps += 1
                    reward += r
                    # print "next state:", s_next, "reward:", r
                    # print "Q =", Q
                    if s_next == GOAL:
                        T = t+1
                        targets.append(r - q[t])  # store target_t
                    else:
                        a_next = choose_action(states[t+1], args.epsilon)  # select A_{t+1} ~ S_{t+1}
                        actions.append(a_next)  # store A_{t+1}
                        # print "action chosen:", a_next
                        sig = choose_sigma(args.method, args.q_mode, last_sigma=sigmas[-1], base=args.base, t=t)  # select sigma according to method and Qsigma mode
                        # print "sigma chosen:", sig
                        sigmas.append(sig)  # store sigma_{t+1}
                        q.append(Q[s_next, a_next])  # store Q_{t+1} = Q[S_{t+1}, A_{t+1}]
                        target = r + sig*args.gamma*q[t+1] + (1-sig)*args.gamma*np.sum(PI[s_next, :]*Q[s_next, :]) - q[t]
                        targets.append(target)  # store target_t
                        pi.append(PI[s_next, a_next])  # store PI_{t+1}
                tau = t - args.n_steps + 1
                # print "tau =", tau
                if tau >= 0:
                    E = 1
                    G = q[tau]
                    # print "G = q[tau] =", G
                    # print "targets =", targets
                    for k in range(tau, min(tau+args.n_steps-1, T-1)):
                        G += E*targets[k]
                        E *= args.gamma*((1-sigmas[k+1])*pi[k+1] + sigmas[k+1])
                    Q[states[tau], actions[tau]] += args.alpha*(G - Q[states[tau], actions[tau]])  # Update Q function
                    # print "updated Q[%d, %d]!!" % (states[tau], actions[tau])
                    # print "G =", G
                    # print "Q[s,a] =", Q[states[tau], actions[tau]]
                    # print Q[states[tau], :]
                    make_greedy(states[tau], args.epsilon)  # Update policy to be epsilon-greedy w.r.t. Q
                if tau == T - 1:
                    break
            print WORLD
            ep += 1
            n_steps.append(steps)
            rewards.append(reward)

        # print "number of steps for each episode:", n_steps
        avg_n_steps = np.average(n_steps)  # average number of steps for each episode.
        print "average number of steps:", avg_n_steps
        average_steps.append(avg_n_steps)

        # print "reward of each episode:", rewards
        avg_reward = np.average(rewards)  # average reward of each episode.
        print "average return:", avg_reward
        average_reward.append(avg_reward)

    print "\nsteps:", average_steps
    print "steps avg:", np.average(average_steps)  # average over K experiments
    print "rewards:", average_reward
    print "rewards avg:", np.average(average_reward)  # average over K experiments


if __name__ == '__main__':
    main()

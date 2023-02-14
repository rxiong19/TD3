import numpy as np
import torch
import gym
import argparse
import os

import utils
import TD3
import OurDDPG
import DDPG

class clock():
    def __init__(self,clock_dim=2):
        self.state = np.zeros(clock_dim)
        self.val = -1
        self.clock_dim = clock_dim
    def tick(self):
        self.state +=1
        if (self.val >= np.power(2,self.clock_dim)):
            self.val = 0
        np.unpackbits(np.array([self.val],dtype='>i8').view(np.uint8))[-self.clock_dim:]
        return self.state
    def reset(self):
        self.value = -1



# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10, clock= None):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        clock.reset()
        state = state[:-1]
        state = np.insert(state, -1, axis=0,values=clock.tick())
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--policy", default="TD3")
    # OpenAI gym environment name
    parser.add_argument("--env", default="MountainCarContinuous-v0") #MountainCar-v0  #Pendulum-v1
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=4, type=int)
    # Time steps initial random policy is used
    parser.add_argument("--start_timesteps", default=100, type=int) #1000 310000
    # How often (time steps) we evaluate
    parser.add_argument("--eval_freq", default=2500, type=int) #250
    # Max time steps to run environment
    parser.add_argument("--max_timesteps", default=150000, type=int) # max_timesteps = 30000
    # Std of Gaussian exploration noise
    parser.add_argument("--expl_noise", default=0.5)
    # Batch size for both actor and critic
    parser.add_argument("--batch_size", default=256, type=int)
    # Discount factor
    parser.add_argument("--discount", default=0.99)
    # Target network update rate
    parser.add_argument("--tau", default=0.005)
    # Noise added to target policy during critic update
    parser.add_argument("--policy_noise", default=0.2)
    # Range to clip target policy noise
    parser.add_argument("--noise_clip", default=0.5)
    # Frequency of delayed policy updates
    parser.add_argument("--policy_freq", default=2, type=int)
    # Save model and optimizer parameters
    parser.add_argument("--save_model", action="store_true")
    # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_model", default="")
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    # Set seeds
    clock_dim =4

    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    clock = clock(clock_dim)

    state_dim = env.observation_space.shape[0] + clock_dim -1
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    elif args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]

    state, done = env.reset(), False
    clock.reset()
    state = state[:-1]
    state = np.insert(state, -1, axis=0,values=clock.tick())
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    noise = utils.OrnsteinUhlenbeckActionNoise(np.zeros(action_dim),np.ones(action_dim)* args.expl_noise)

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + noise()
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        next_state = np.insert(state, -1, axis=0,values=clock.tick())
        done_bool = float(
            done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            clock.reset()
            state = np.insert(state, -1, axis=0,values=clock.tick())
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model:
                policy.save(f"./models/{file_name}")



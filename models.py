import matplotlib.pyplot as plt
import numpy as np

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.util.util import make_vec_env
from imitation.policies.serialize import load_policy
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO


SEED = 42
rng = np.random.default_rng(SEED)


def env_param(env_name="seals:seals/CartPole-v0", model_name="ppo-huggingface", organisation="HumanCompatibleAI",
              model_env_name="seals-CartPole-v0", min_episodes=50):
    env = make_vec_env(
        env_name,
        rng=rng,
        n_envs=1,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
    )
    expert = load_policy(
        model_name,
        organization=organisation,
        env_name=model_env_name,
        venv=env,
    )
    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=min_episodes),
        rng=rng,
    )
    return env, expert, rollouts

def bc_progressive_learning(env_name="seals:seals/CartPole-v0", model_name="ppo-huggingface",
                            organisation="HumanCompatibleAI", model_env_name="seals-CartPole-v0", min_episodes=10,
                            max_epochs=10, num_evaluate_episodes=10):

    # Make demonstrations
    env, expert, rollouts = env_param(env_name, model_name, organisation, model_env_name, min_episodes)

    transitions = rollout.flatten_trajectories(rollouts)

    reward_in_learning = []

    # Instantiate the model
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
    )

    def on_epoch_end():
        env.seed(SEED)
        mean_reward, _ = evaluate_policy(bc_trainer.policy, env, num_evaluate_episodes)
        reward_in_learning.append(mean_reward)

    # Train Model
    bc_trainer.train(
        n_epochs=max_epochs,
        on_epoch_end=on_epoch_end,
        progress_bar=False,
    )
    return reward_in_learning



def bc_complete_learning(env_name="seals:seals/CartPole-v0", model_name="ppo-huggingface",
                         organisation="HumanCompatibleAI", model_env_name="seals-CartPole-v0", min_episodes=10,
                         max_epochs=10, num_evaluate_episodes=10):
    # Make demonstrations
    env, expert, rollouts = env_param(env_name, model_name, organisation, model_env_name, min_episodes)

    transitions = rollout.flatten_trajectories(rollouts)

    # Instantiate the model
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
    )
    # Train Model
    bc_trainer.train(
        n_epochs=max_epochs,
        progress_bar=False,
    )
    # Evaluate The Model
    env.seed(SEED)
    reward_in_learning, _ = evaluate_policy(bc_trainer.policy, env, num_evaluate_episodes)
    # Save mean reward
    return reward_in_learning


def bc_evolutive_demo_number(n_demo=10, env_name="seals:seals/MountainCar-v0", model_name="ppo-huggingface",
                             organisation="HumanCompatibleAI", model_env_name="seals-MountainCar-v0", max_epochs=10,
                             max_train_episode=10):
    reward_in_demo = []
    for num_demo in range(1, n_demo + 1):
        print(f'Demo size = {num_demo * 500} => ')
        reward_in_learning = bc_complete_learning(env_name, model_name, organisation, model_env_name, num_demo,
                                                  max_epochs, max_train_episode)
        reward_in_demo.append(reward_in_learning)

    return reward_in_demo


def gail_progressive_learning(env_name="seals:seals/CartPole-v0", model_name="ppo-huggingface",
                         organisation="HumanCompatibleAI", model_env_name="seals-CartPole-v0", min_episodes=10,
                         max_epochs=10, num_evaluate_episodes = 10, total_timesteps = 800_000):
    # Make demonstrations
    env, expert, rollouts = env_param(env_name, model_name, organisation, model_env_name, min_episodes)

    reward_in_learning = []

    learner = PPO(
        env=env,
        policy=MlpPolicy,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0004,
        gamma=0.95,
        n_epochs=max_epochs,
        seed=SEED,
    )
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )
    gail_trainer = GAIL(
        demonstrations=rollouts,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=8,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True
    )

    def evaluation_callback(num_round):
        env.seed(SEED)
        mean_reward, _ = evaluate_policy(learner, env, num_evaluate_episodes)
        reward_in_learning.append(mean_reward)

    # train the learner and evaluate again
    gail_trainer.train(total_timesteps, callback = evaluation_callback)  # Train for 800_000 steps to match expert.
    return reward_in_learning



class EvaluateAllModel:
    def __init__(self, env_name="seals:seals/CartPole-v0", model_name="ppo-huggingface",
                 organisation="HumanCompatibleAI", model_env_name="seals-CartPole-v0", max_transitions_episodes=10,
                 max_epochs=10, num_evaluate_episodes=10, num_round_gail = 30):

        self.env_name = env_name
        self.model_name = model_name
        self.organisation = organisation
        self.model_env_name = model_env_name
        self.max_transitions_episodes = max_transitions_episodes
        self.max_epochs = max_epochs
        self.num_evaluate_episodes = num_evaluate_episodes

        self.bc_reward_in_learning = bc_progressive_learning(
            env_name=self.env_name,
            model_name=self.model_name,
            organisation=self.organisation,
            model_env_name=self.model_env_name,
            min_episodes=max_transitions_episodes,
            max_epochs=max_epochs,
            num_evaluate_episodes=num_evaluate_episodes
        )

        self.bc_reward_in_demo = bc_evolutive_demo_number(
            n_demo=self.max_transitions_episodes, env_name=self.env_name,
            model_name=self.model_name, organisation=self.organisation,
            model_env_name=self.model_env_name,
            max_epochs=self.max_epochs,
            max_train_episode=self.max_transitions_episodes
        )

        self.gail_reward_in_learning = gail_progressive_learning(
            env_name=self.env_name,
            model_name=self.model_name,
            organisation=self.organisation,
            model_env_name=self.model_env_name,
            min_episodes=max_transitions_episodes,
            max_epochs=max_epochs,
            num_evaluate_episodes=num_evaluate_episodes,
            total_timesteps = 2048 * num_round_gail
        )

        self.gail_reward_in_learning_2 = gail_progressive_learning(
            env_name=self.env_name,
            model_name=self.model_name,
            organisation=self.organisation,
            model_env_name=self.model_env_name,
            min_episodes=5*max_transitions_episodes,
            max_epochs=max_epochs,
            num_evaluate_episodes=num_evaluate_episodes,
            total_timesteps=2048 * num_round_gail
        )






import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.bc import BC
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
import tempfile

# Environnement et Expert
def setup_environment(env_name="CartPole-v1", expert_name="ppo-huggingface", org="HumanCompatibleAI"):
    """
    Crée l'environnement et charge l'expert pré-entraîné.
    """
    env = make_vec_env(env_name, rng=np.random.default_rng(), n_envs=1)
    expert = load_policy(
        expert_name,
        organization=org,
        env_name=env_name,
        venv=env
    )
    return env, expert

# Configuration et Entraînement de DAgger avec suivi des récompenses
def train_with_dagger(env, expert, steps=300, eval_episodes=10, eval_intervals=30):
    """
    Entraîne un agent avec l'algorithme DAgger et collecte les récompenses au fil des étapes.
    """
    # Initialisation de l'entraîneur de Clonage Comportemental
    bc_trainer = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        rng=np.random.default_rng(),
    )
    
    rewards_over_time = []  # Stockage des récompenses moyennes

    with tempfile.TemporaryDirectory(prefix="dagger_training_") as tmpdir:
        print(f"Répertoire temporaire pour l'entraînement : {tmpdir}")
        
        # Initialisation de l'entraîneur DAgger
        dagger_trainer = SimpleDAggerTrainer(
            venv=env,
            scratch_dir=tmpdir,
            expert_policy=expert,
            bc_trainer=bc_trainer,
            rng=np.random.default_rng(),
        )
        
        for step in range(0, steps + 1, eval_intervals):
            # Entraîner sur un intervalle
            dagger_trainer.train(eval_intervals)

            # Évaluer la politique entraînée
            rewards, _ = evaluate_policy(dagger_trainer.policy, env, n_eval_episodes=eval_episodes, return_episode_rewards=True)
            mean_reward = np.mean(rewards)
            print(f"Étape {step}/{steps}, Récompense moyenne : {mean_reward}")
            rewards_over_time.append(mean_reward)

    return dagger_trainer, rewards_over_time

# Tracé des récompenses
def plot_rewards(rewards, steps, eval_intervals):
    """
    Trace la courbe de récompense en fonction des étapes.
    """
    plt.figure(figsize=(10, 6))
    x = np.arange(0, steps + 1, eval_intervals)
    plt.plot(x[:len(rewards)], rewards, marker='o')
    plt.title("Évolution de la récompense moyenne pendant l'entraînement DAgger")
    plt.xlabel("Étapes d'entraînement")
    plt.ylabel("Récompense moyenne")
    plt.grid()
    plt.show()

# Programme Principal
if __name__ == "__main__":
    env_name = "seals:seals/MountainCar-v0"
    expert_name = "ppo-huggingface"
    organization = "HumanCompatibleAI"

    # Initialisation de l'environnement et de l'expert
    print("Initialisation de l'environnement et chargement de l'expert...")
    env, expert = setup_environment(env_name, expert_name, organization)
    
    # Entraînement avec DAgger
    print("Entraînement avec DAgger en cours...")
    dagger_trainer, rewards_over_time = train_with_dagger(env, expert, steps=300, eval_intervals=10)
    
    # Tracé de la récompense
    print("Tracé de la récompense...")
    plot_rewards(rewards_over_time, steps=300, eval_intervals=10)

    # Évaluation de la politique entraînée
    print("Évaluation de la politique entraînée...")
    

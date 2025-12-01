"""Training loop for MADDPG with advanced DDPG features.

Implements a modified DDPG algorithm with:
- Prioritized Experience Replay (PER)
- Distributional value learning (C51)
- Noisy networks for exploration
- N-step returns
"""

import numpy as np
import os
import torch
from collections import deque
from datetime import datetime


def ddpg(env, agent, n_episodes, eps_start=0., eps_min=0., eps_decay=0., 
         beta_start=1., beta_end=1., continue_after_solved=True, save_dir='results'):
    """Train agent using enhanced DDPG algorithm.

    Args:
        env: Unity ML-Agents environment instance
        agent: DDPG agent to train
        n_episodes: Maximum number of training episodes
        eps_start: Initial epsilon for OU noise (if used)
        eps_min: Minimum epsilon value
        eps_decay: Multiplicative decay rate for epsilon
        beta_start: Initial importance sampling exponent
        beta_end: Final importance sampling exponent
        continue_after_solved: Whether to continue training after solving
        save_dir: Directory to save training artifacts
        
    Returns:
        List of episode scores (max score per episode)
    """
    # Environment setup
    brain_name = env.brain_names[0]
    num_agents = agent.n_agents
    
    # Create results directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Training state tracking
    solved = False
    episode_scores = []
    scores_window = deque(maxlen=100)

    # Initialize exploration parameters
    agent.eps = eps_start
    agent.beta = beta_start

    # Track noisy layer parameters for monitoring exploration
    actor_noisy_params = [param.view(-1) for name, param in agent.actor_local.named_parameters()
                          if name.endswith('noisy_weight') or name.endswith('noisy_bias')]

    critic_noisy_params = [param.view(-1) for name, param in agent.critic_local.named_parameters()
                           if name.endswith('noisy_weight') or name.endswith('noisy_bias')]

    # Progress reporting format
    status_format = "\rEpisode {:d} | Total Steps: {:d} | Ep Score: {:>6.3f} | Avg 100: {:>6.3f}"\
                    " | Eps: {:>6.4f} | Alpha: {:>6.4f} | Beta: {:>6.4f}"
    status_msg = status_format.format(0, 0, float('nan'), float('nan'), agent.eps, agent.a, agent.beta)

    last_status_len = 0
    total_steps = 0
    try:
        for episode_num in range(1, n_episodes + 1):
            # Reset environment for new episode
            env_info = env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations
            episode_rewards = np.zeros(num_agents)
            timestep = 0
            
            # Episode loop
            while True:
                # Select actions using current policy
                actions = agent.act(states)
                
                # Execute actions in environment
                env_info = env.step(actions)[brain_name]
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done
                
                # Store experience and learn
                agent.step(states, actions, rewards, next_states, dones)
                
                # Update tracking variables
                timestep += 1
                total_steps += num_agents
                episode_rewards += rewards
                states = next_states
                
                print(status_msg + " *** step {:d} ***".format(timestep), end='')
                
                if np.any(dones):
                    break

            # Reset n-step collectors after episode
            agent.reset()

            # Record episode performance (use max score between two agents)
            episode_max_score = np.max(episode_rewards)
            episode_scores.append(episode_max_score)
            scores_window.append(episode_max_score)

            # Calculate rolling average
            avg_score = np.mean(scores_window)

            # Update status message
            status_msg = status_format.format(episode_num, total_steps,
                                            episode_max_score, avg_score,
                                            agent.eps, agent.a, agent.beta)
            # Append noise statistics if using noisy layers
            if actor_noisy_params:
                noise_magnitudes = np.concatenate([param.data.abs().cpu().numpy()
                                                  for param in actor_noisy_params])
                noise_mean, noise_std = np.mean(noise_magnitudes), np.std(noise_magnitudes)
                status_msg += " | Actor Noise: {:>6.4f} ± {:<6.4f}".format(noise_mean, noise_std)
                action_std = agent.estimate_actor_noise_std()
                status_msg += " | Action Std: {:>6.4f}".format(action_std)
            if critic_noisy_params:
                noise_magnitudes = np.concatenate([param.data.abs().cpu().numpy()
                                                  for param in critic_noisy_params])
                noise_mean, noise_std = np.mean(noise_magnitudes), np.std(noise_magnitudes)
                status_msg += " | Critic Noise: {:>6.4f} ± {:<6.4f}".format(noise_mean, noise_std)

            # Print progress (newline every 10 episodes)
            if episode_num % 10 == 0:
                print(status_msg, end='\n')
                last_status_len = 0
            else:
                print(status_msg + " " * max(0, last_status_len - len(status_msg)), end='')
                last_status_len = len(status_msg)

            # Check if environment is solved
            if not solved and len(scores_window) >= 100 and avg_score >= 0.5:
                print('\n🎉 Environment solved in {:d} episodes! Average Score: {:.3f}'
                      .format(episode_num, avg_score))
                solved = True
                
                # Save checkpoint when solved
                checkpoint_path = os.path.join(save_dir, f'checkpoint_solved_{timestamp}.pth')
                torch.save({
                    'actor_state_dict': agent.actor_local.state_dict(),
                    'critic_state_dict': agent.critic_local.state_dict(),
                }, checkpoint_path)
                print(f'Checkpoint saved to {checkpoint_path}')
                
                if not continue_after_solved:
                    break

            # Update exploration parameters
            agent.eps = max(eps_min, agent.eps * eps_decay)
            agent.beta = beta_start + (beta_end - beta_start) * (episode_num / n_episodes)

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    
    # Save final checkpoint
    final_checkpoint = os.path.join(save_dir, f'checkpoint_final_{timestamp}.pth')
    torch.save({
        'actor_state_dict': agent.actor_local.state_dict(),
        'critic_state_dict': agent.critic_local.state_dict(),
    }, final_checkpoint)
    print(f'\nFinal checkpoint saved to {final_checkpoint}')
    
    # Save training scores
    scores_file = os.path.join(save_dir, f'training_scores_{timestamp}.npy')
    np.save(scores_file, episode_scores)
    print(f'Training scores saved to {scores_file}')

    return episode_scores
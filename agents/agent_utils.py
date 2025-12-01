"""Utility methods for agent checkpointing."""

import torch


def save_checkpoint(agent, filepath):
    """Save agent networks and optimizer states to file.
    
    Args:
        agent: Agent instance to save
        filepath: Path to save checkpoint file
    """
    torch.save({
        'actor_state_dict': agent.actor_local.state_dict(),
        'actor_target_state_dict': agent.actor_target.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_state_dict': agent.critic_local.state_dict(),
        'critic_target_state_dict': agent.critic_target.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
    }, filepath)


def load_checkpoint(agent, filepath, device):
    """Load agent networks and optimizer states from file.
    
    Args:
        agent: Agent instance to load into
        filepath: Path to checkpoint file
        device: Device to load tensors onto
    """
    checkpoint = torch.load(filepath, map_location=device)
    agent.actor_local.load_state_dict(checkpoint['actor_state_dict'])
    agent.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
    agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    agent.critic_local.load_state_dict(checkpoint['critic_state_dict'])
    agent.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
    agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

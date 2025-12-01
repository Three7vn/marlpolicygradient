"""
Wrapper module for backwards compatibility with notebooks.
Imports training function from training module.
"""

from training import ddpg

__all__ = ['ddpg']

#!/bin/bash
# Start Jupyter Notebook with Python 3.10 environment

echo "🚀 Starting Jupyter with Python 3.10..."
echo ""
echo "Python version:"
./venv310/bin/python --version
echo ""
echo "Starting Jupyter Notebook..."
echo "Your browser will open automatically."
echo "Press Ctrl+C to stop the server when done."
echo ""

cd "$(dirname "$0")"
./venv310/bin/jupyter notebook tennismaddpg.ipynb

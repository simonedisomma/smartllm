#!/bin/bash
# Virtual environment
source .venv_3.12/bin/activate

# Source the environment variables
source ./set_env.sh      

# Run an example
#python3 -m examples.slides

#Activate logging with logger DEBUG level
python3 -m examples.blog_post

#!/bin/bash

# Source the environment variables
source ./set_env.sh

# Run all the tests
python -m unittest discover tests/

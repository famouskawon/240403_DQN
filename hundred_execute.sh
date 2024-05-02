#!/bin/bash

for i in {1..100}
do
   python 080204_double_dqn_cartpole.py
   python 080204_dqn_cartpole.py
done

#!/bin/bash

for i in {1..10}
do
   python 080204_double_dqn_cartpole.py
   python 080204_dqn_cartpole.py
   python 000000_cql_cartpole.py
done

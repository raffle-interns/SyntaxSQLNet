#!/bin/bash
set -e
python train.py --model='agg' --num_epochs=50&
python train.py --model='keyword' --num_epochs=50
python train.py --model='andor' --num_epochs=50&
python train.py --model='distinct' --num_epochs=50
python train.py --model='op' --num_epochs=50&
python train.py --model='having' --num_epochs=50
python train.py --model='desasc' --num_epochs=50&
python train.py --model='limitvalue' --num_epochs=50
python train.py --model='value' --num_epochs=50&
python train.py --model='column' --num_epochs=150
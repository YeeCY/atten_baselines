 OPENAI_LOGDIR=/home/lzy/experiments/coin0/ python run_gridworld_generalize.py
nohup tensorboard --logdir=./ --port=6007 &
git push origin master:lzy_test

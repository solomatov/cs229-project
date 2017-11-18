tmux new-session -d
tmux select-window -t 0
tmux send-keys "source ./env/bin/activate; clear" C-m
tmux split-window -v -l 3 -d "source ./env/bin/activate; python -m visdom.server"

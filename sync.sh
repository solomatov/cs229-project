rsync \
    -avzP \
    --delete \
    --exclude '/env' \
    --exclude '/data' \
    --exclude '/.git' \
    --exclude '/.idea' \
    --exclude '/experiments.txt' \
    . \
    $1 

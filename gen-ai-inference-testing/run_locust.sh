#!/bin/bash

echo "starting locust"
locust -f $SCRIPT -H $ENDPOINT_NAME --master \
    --expect-workers $WORKERS -u $USERS -t $RUN_TIME -r $SPAWN_RATE --csv $RESULTS_PREFIX --headless &


for (( c=1; c<=$WORKERS; c++ ))
do
    echo "starting locust worker $c"
    locust -f $SCRIPT -H $ENDPOINT_NAME --worker --master-host=localhost &
    # Each worker loads the prompt dataset on startup, which revalidates it against the
    # Hugging Face Hub. Starting all of them at once trips the Hub's rate limit for
    # unauthenticated requests, and a rate-limited worker generates no load at all.
    sleep 0.3
done


echo "waiting for locust testing to complete"
wait
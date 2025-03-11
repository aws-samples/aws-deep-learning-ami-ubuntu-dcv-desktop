#!/bin/bash

[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1
[  -z "$MODEL_ID"  ] && echo "MODEL_ID environment variable must exist" && exit 1


export MODEL_SERVER_CORES=8
/opt/program/serve \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"
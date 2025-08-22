#!/bin/bash

[ ! -d /snapshots ] && echo "/snapshots dir must exist" && exit 1
[  -z "$MODEL_ID"  ] && echo "MODEL_ID environment variable must exist" && exit 1

/opt/program/serve \
&& /bin/bash -c "trap : TERM INT; sleep infinity & wait"
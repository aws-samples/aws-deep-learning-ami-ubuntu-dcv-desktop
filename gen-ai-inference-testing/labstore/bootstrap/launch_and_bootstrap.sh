#!/usr/bin/env bash
# Launch a Neuron desktop into a capacity reservation, then kick off unattended
# pre-staging on the instance and report progress.
#
# Design points:
#   1. UbuntuAMIOverride is passed explicitly (create-stack fails without it).
#   2. Notifies on failure as well as success, so a create-stack error does not
#      sit unseen.
#   3. Fires bootstrap.sh over SSM so the slow setup happens before a human
#      sits down.
#
# Usage:
#   RESERVATION_ID=cr-xxxx VPC_ID=vpc-xxxx KEY_NAME=<key pair> ./launch_and_bootstrap.sh
# Optional: STACK_NAME, SUBNET_ID, S3_BUCKET, EFS_ID (default: the lab store
# stack's exports, see ../aws-defaults.sh), AWS_REGION, REPO_URL, BRANCH.
set -uo pipefail

# shellcheck source=../aws-defaults.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/aws-defaults.sh"
REGION="$(lab_region)"
STACK_NAME="${STACK_NAME:-trn2-dl-desktop-2}"
RESERVATION_ID="${RESERVATION_ID:-}"
TEMPLATE="${TEMPLATE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/deep-learning-ubuntu-desktop.yaml}"
REPO_URL="${REPO_URL:-https://github.com/aws-samples/aws-deep-learning-ami-ubuntu-dcv-desktop.git}"
BRANCH="${BRANCH:-main}"
RUN_BOOTSTRAP="${RUN_BOOTSTRAP:-yes}"

INSTANCE_TYPE=trn2.48xlarge
VPC_ID="${VPC_ID:-}"
KEY_NAME="${KEY_NAME:-}"
# Durable lab store (lab-store.yaml). Give it mount targets in every AZ the
# reservation could be in.
S3_BUCKET="${S3_BUCKET:-$(lab_bucket)}"
EFS_ID="${EFS_ID:-$(lab_efs_id)}"
# SUBNET_ID is resolved from the reservation's own AZ below unless set: a
# targeted reservation is AZ-pinned, so an instance in another AZ cannot use it.
SUBNET_ID="${SUBNET_ID:-}"
# HOME is not guaranteed. cloud-init UserData runs with a minimal environment,
# and under `set -u` an unset HOME is fatal -- which is exactly how this script
# died on the remote launcher ("line 40: HOME: unbound variable") before it ever
# reached the reservation poll. Same bug class as labstore.sh; swept across all
# ops scripts rather than patched one file at a time.
if [[ -z "${HOME:-}" ]]; then
  HOME="$(getent passwd "$(id -u)" 2>/dev/null | cut -d: -f6)"
  [[ -n "$HOME" ]] || HOME=/tmp
  export HOME
fi

LOG="${LOG:-$HOME/.cache/trn2-desktop/launch.log}"
mkdir -p "$(dirname "$LOG")" 2>/dev/null || LOG=/tmp/launch.log

exec > >(tee -a "$LOG") 2>&1

# SNS notification is OPT-IN and off by default: set SNS_TOPIC_ARN to a topic
# you own to enable it. Useful if the launch runs unattended on a machine that
# may be asleep, since macOS desktop notifications are worthless then.
SNS_TOPIC_ARN="${SNS_TOPIC_ARN:-}"

notify() {
  local title="$1" body="$2"

  # Channel 1: SNS -- machine-independent. Failures here must never abort a
  # launch, so everything is swallowed.
  if [[ -n "$SNS_TOPIC_ARN" ]]; then
    aws sns publish --region "$REGION" --topic-arn "$SNS_TOPIC_ARN" \
      --subject "$(printf '%.99s' "$title")" \
      --message "$title

$body

stack       : $STACK_NAME
reservation : $RESERVATION_ID
time        : $(date -u +%Y-%m-%dT%H:%M:%SZ)
log         : $LOG" >/dev/null 2>&1 || true
  fi

  # Channel 2: macOS desktop notification, if anyone is watching.
  command -v terminal-notifier >/dev/null 2>&1 \
    && terminal-notifier -title "$title" -message "$body" -sound Glass >/dev/null 2>&1 && return
  command -v osascript >/dev/null 2>&1 \
    && osascript -e "display notification \"${body//\"/\\\"}\" with title \"${title//\"/\\\"}\" sound name \"Glass\"" >/dev/null 2>&1 || true
}
die() { echo "ERROR: $1"; notify "trn2 launch FAILED" "$1"; exit 1; }
for v in VPC_ID KEY_NAME S3_BUCKET EFS_ID; do
  [[ -n "${!v}" ]] || die "$v is not set and could not be resolved; set it explicitly"
done

echo "=============================================================="
echo "launch attempt $(date -u +%Y-%m-%dT%H:%M:%SZ)  stack=$STACK_NAME"

[[ -n "$RESERVATION_ID" ]] || die "RESERVATION_ID not set"
[[ -f "$TEMPLATE" ]] || die "template not found: $TEMPLATE"

EXISTING=$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK_NAME" \
  --query 'Stacks[0].StackStatus' --output text 2>/dev/null)
if [[ -n "$EXISTING" && "$EXISTING" != None ]]; then
  echo "stack $STACK_NAME already exists (status=$EXISTING) -- nothing to do"; exit 0
fi

# DCV access CIDR. Auto-detection is correct when run from your laptop, but
# WRONG when this script runs on an EC2 instance: checkip would return that
# instance's address and the security group would lock you out of DCV
# entirely. So an explicit DESKTOP_ACCESS_CIDR always wins.
if [[ -n "${DESKTOP_ACCESS_CIDR:-}" ]]; then
  CIDR="$DESKTOP_ACCESS_CIDR"
  echo "access CIDR: $CIDR (explicit)"
else
  PUBLIC_IP=$(curl -s --max-time 20 http://checkip.amazonaws.com/ | tr -d '[:space:]')
  [[ -n "$PUBLIC_IP" ]] || die "could not detect public IP; set DESKTOP_ACCESS_CIDR instead of guessing"
  CIDR="${PUBLIC_IP}/32"
  echo "access CIDR: $CIDR (auto-detected)"
  # Guard against the exact mistake above: if we are on EC2, auto-detection is
  # almost certainly wrong for a human's DCV client.
  if curl -s --max-time 2 -o /dev/null http://169.254.169.254/latest/meta-data/ 2>/dev/null; then
    die "auto-detected the CIDR while running on EC2 -- that would be this instance, not your workstation. Set DESKTOP_ACCESS_CIDR."
  fi
fi

# Resolve the subnet from the reservation's own AZ. Targeted reservations are
# AZ-pinned; a mismatch is an unusable reservation, not a slow one.
CB_AZ=$(aws ec2 describe-capacity-reservations --region "$REGION" \
  --capacity-reservation-ids "$RESERVATION_ID" \
  --query 'CapacityReservations[0].AvailabilityZone' --output text 2>/dev/null)
[[ -n "$CB_AZ" && "$CB_AZ" != None ]] || die "could not read AZ for $RESERVATION_ID"
if [[ -z "$SUBNET_ID" ]]; then
  SUBNET_ID=$(aws ec2 describe-subnets --region "$REGION" \
    --filters "Name=vpc-id,Values=$VPC_ID" "Name=availability-zone,Values=$CB_AZ" \
              "Name=default-for-az,Values=true" \
    --query 'Subnets[0].SubnetId' --output text 2>/dev/null)
fi
[[ -n "$SUBNET_ID" && "$SUBNET_ID" != None ]] || die "no default subnet in $CB_AZ of $VPC_ID"
echo "reservation AZ: $CB_AZ -> subnet $SUBNET_ID"

# The durable store must have a mount target in this AZ or EFS is unreachable.
#
# Distinguish "query failed" from "zero mount targets". The previous version
# sent stderr to /dev/null and treated an empty result as zero, so an IAM
# AccessDeniedException was reported as "no mount target in <az>" --
# sending me hunting for a missing mount target that was in fact present.
# A wrong diagnosis is worse than none.
MT_OUT=$(aws efs describe-mount-targets --region "$REGION" --file-system-id "$EFS_ID" \
  --query "length(MountTargets[?AvailabilityZoneName=='$CB_AZ'])" --output text 2>&1)
MT_RC=$?
if (( MT_RC != 0 )); then
  die "could not query EFS mount targets (exit $MT_RC): $(printf '%s' "$MT_OUT" | head -2)"
fi
if ! [[ "$MT_OUT" =~ ^[0-9]+$ ]]; then
  die "unexpected mount-target query output: $(printf '%s' "$MT_OUT" | head -2)"
fi
if (( MT_OUT == 0 )); then
  die "EFS $EFS_ID genuinely has no mount target in $CB_AZ -- create one before launching"
fi
echo "lab store: $EFS_ID has $MT_OUT mount target(s) in $CB_AZ"

# Wait for the reservation to go active ('scheduled' means not yet started).
# 1440 iterations x 60s = 24h, so the script can be started well ahead of the
# reservation's start time, with margin for activation lag.
for _ in $(seq 1 1440); do
  read -r STATE AVAIL < <(aws ec2 describe-capacity-reservations --region "$REGION" \
    --capacity-reservation-ids "$RESERVATION_ID" \
    --query 'CapacityReservations[0].[State,AvailableInstanceCount]' --output text 2>/dev/null)
  echo "$(date -u +%H:%M:%SZ) reservation=$STATE available=${AVAIL:-?}"
  [[ "$STATE" == active ]] && break
  case "$STATE" in expired|cancelled|failed|payment-failed) die "reservation is $STATE";; esac
  sleep 60
done
[[ "$STATE" == active ]] || die "reservation never became active (last=$STATE)"

echo "creating stack ..."
aws cloudformation create-stack \
  --region "$REGION" --stack-name "$STACK_NAME" \
  --template-body "file://$TEMPLATE" \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameters \
    ParameterKey=DesktopInstanceType,ParameterValue="$INSTANCE_TYPE" \
    ParameterKey=CapacityReservationId,ParameterValue="$RESERVATION_ID" \
    ParameterKey=DesktopVpcId,ParameterValue="$VPC_ID" \
    ParameterKey=DesktopVpcSubnetId,ParameterValue="$SUBNET_ID" \
    ParameterKey=DesktopHasPublicIpAddress,ParameterValue=true \
    ParameterKey=DesktopAccessCIDR,ParameterValue="$CIDR" \
    ParameterKey=KeyName,ParameterValue="$KEY_NAME" \
    ParameterKey=S3Bucket,ParameterValue="$S3_BUCKET" \
    ParameterKey=EbsVolumeSize,ParameterValue=1000 \
    ParameterKey=EbsVolumeType,ParameterValue=gp3 \
    ParameterKey=EBSOptimized,ParameterValue=true \
    ParameterKey=AWSUbuntuAMIType,ParameterValue=Ubuntu2404LTS \
    ParameterKey=FSxForLustre,ParameterValue=disabled \
    ParameterKey=UbuntuAMIOverride,ParameterValue= \
    ParameterKey=EFSFileSystemId,ParameterValue="$EFS_ID" \
  --output text >/dev/null || die "create-stack rejected the request (run preflight.sh)"

notify "trn2 stack creating" "$STACK_NAME accepted; waiting for resources"

aws cloudformation wait stack-create-complete --region "$REGION" --stack-name "$STACK_NAME"
if [[ $? -ne 0 ]]; then
  REASONS=$(aws cloudformation describe-stack-events --region "$REGION" --stack-name "$STACK_NAME" \
    --query 'StackEvents[?ResourceStatus==`CREATE_FAILED`].[LogicalResourceId,ResourceStatusReason]' \
    --output text | head -5)
  echo "$REASONS"
  die "stack did not reach CREATE_COMPLETE: $(echo "$REASONS" | head -1)"
fi
echo "stack CREATE_COMPLETE"

IID=$(aws ec2 describe-instances --region "$REGION" \
  --filters "Name=tag:Name,Values=${STACK_NAME}-deep-learning-ubuntu-desktop" \
            "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)
PIP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
LIFECYCLE=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].InstanceLifecycle' --output text)
END=$(aws ec2 describe-capacity-reservations --region "$REGION" \
  --capacity-reservation-ids "$RESERVATION_ID" --query 'CapacityReservations[0].EndDate' --output text)

echo "instance=$IID ip=$PIP lifecycle=$LIFECYCLE block_ends=$END"
[[ "$LIFECYCLE" == "capacity-block" ]] || echo "WARN: lifecycle is '$LIFECYCLE', expected capacity-block"
notify "trn2 desktop launched" "$IID at $PIP -- bootstrap starting"

if [[ "$RUN_BOOTSTRAP" != yes ]]; then
  echo "RUN_BOOTSTRAP=$RUN_BOOTSTRAP -- skipping pre-staging"; exit 0
fi

# Wait for SSM to come online. UserData reboots the host, so SSM drops out
# and returns; do not treat a gap as failure.
echo "waiting for SSM ..."
for _ in $(seq 1 60); do
  P=$(aws ssm describe-instance-information --region "$REGION" \
       --filters "Key=InstanceIds,Values=$IID" \
       --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null)
  [[ "$P" == Online ]] && break
  sleep 15
done
[[ "$P" == Online ]] || die "SSM never came online for $IID"

# Run bootstrap detached: it takes far longer than an SSM command may run.
# It is fetched from the branch so the orchestration stays version-controlled.
echo "firing bootstrap over SSM ..."
# Pass the commands via a JSON FILE, not inline.
#
# The inline form with embedded single quotes is rejected by the AWS CLI
# parameter parser ("ParamValidation: Expected ',' received '''") -- which is
# exactly how an earlier launch created the desktop but never dispatched the
# bootstrap, leaving a bare machine.
#
# Two further things this encodes, both learned the hard way on that instance:
#   * Run as ubuntu through a LOGIN shell. SSM injects
#     AWS_SHARED_CREDENTIALS_FILE=/var/lib/amazon/ssm/credentials pointing at
#     the Default Host Management role, which has no S3 access; a login shell
#     drops it and falls back to IMDS, i.e. the stack's own instance role.
#   * Pass S3_BUCKET explicitly rather than trusting a default.
SSM_JSON=$(mktemp /tmp/ssm-bootstrap-XXXXXX.json)
"${PY_JSON:-python3}" - "$BRANCH" "$REPO_URL" "$S3_BUCKET" "$EFS_ID" "$REGION" >"$SSM_JSON" <<'PYEOF'
import json, sys
branch, repo, bucket, efs, region = sys.argv[1:6]
inner = (
    f"cd /tmp && rm -rf /tmp/bs && "
    f"git clone --depth 1 -b {branch} {repo} /tmp/bs && "
    f"chown -R ubuntu:ubuntu /tmp/bs && "
    f"chmod +x /tmp/bs/gen-ai-inference-testing/labstore/bootstrap/bootstrap.sh "
    f"/tmp/bs/gen-ai-inference-testing/labstore/*.sh"
)
run = (
    f"su - ubuntu -c '"
    f"S3_BUCKET={bucket} LAB_S3_BUCKET={bucket} LAB_S3_PREFIX=lab "
    f"EFS_ID={efs} AWS_REGION={region} "
    f"nohup bash /tmp/bs/gen-ai-inference-testing/labstore/bootstrap/bootstrap.sh "
    f">/home/ubuntu/bootstrap-nohup.log 2>&1 &'"
)
json.dump({"commands": [inner, run, "echo dispatched"]}, sys.stdout)
PYEOF
CID=$(aws ssm send-command --region "$REGION" --instance-ids "$IID" \
  --document-name AWS-RunShellScript \
  --comment "trn2 pre-staging bootstrap" \
  --parameters "file://$SSM_JSON" \
  --timeout-seconds 600 \
  --query 'Command.CommandId' --output text 2>&1)
rm -f "$SSM_JSON"
[[ "$CID" == *-*-* ]] || die "SSM send-command failed: $CID"
echo "bootstrap dispatched (ssm command $CID)"

# Poll the status file the bootstrap writes, so we report phases not log spam.
LAST=""
for _ in $(seq 1 240); do
  Q=$(aws ssm send-command --region "$REGION" --instance-ids "$IID" \
    --document-name AWS-RunShellScript \
    --parameters commands="tail -1 /home/ubuntu/bootstrap-status 2>/dev/null || echo PHASE=PENDING" \
    --query 'Command.CommandId' --output text 2>/dev/null)
  sleep 8
  CUR=$(aws ssm get-command-invocation --region "$REGION" --command-id "$Q" \
         --instance-id "$IID" --query 'StandardOutputContent' --output text 2>/dev/null | tr -d '\n')
  if [[ -n "$CUR" && "$CUR" != "$LAST" ]]; then
    echo "[$(date -u +%H:%M:%SZ)] $CUR"
    LAST="$CUR"
    case "$CUR" in
      *PHASE=READY*)  notify "trn2 ready" "Server up on :8080 (nginx), cache backed up. $PIP"; break;;
      *PHASE=FAILED*) notify "trn2 bootstrap FAILED" "$CUR"; break;;
    esac
  fi
  sleep 52
done

cat <<EOF

=============================================================
 trn2 desktop
   instance : $IID
   DCV      : https://$PIP:8443   (user ubuntu; set pw via SSM first)
   SSM      : aws ssm start-session --target $IID --region $REGION
   SSH      : ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$PIP
   reservation ends: $END  (save work to EFS/S3 well before this)
   bootstrap log: /home/ubuntu/bootstrap.log  (status: ~/bootstrap-status)
   extend   : aws ec2 describe-capacity-block-extension-offerings \\
                --capacity-reservation-id $RESERVATION_ID \\
                --capacity-block-extension-duration-hours <hours>
=============================================================
EOF

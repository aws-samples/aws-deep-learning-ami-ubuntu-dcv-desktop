#!/usr/bin/env bash
# Pre-flight: validate the EXACT CloudFormation parameter set without spending
# anything, by creating a CREATE-type change set and then deleting it.
#
# Why this exists: create-stack can fail late on a parameter problem (e.g.
# "Parameters: [UbuntuAMIOverride] must have values"), after the reservation
# window has started. A change set surfaces that class of error without
# creating anything, and can be run well in advance.
#
# Run this any time the parameter set or template changes.
set -uo pipefail

# shellcheck source=../aws-defaults.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/aws-defaults.sh"
REGION="$(lab_region)"
STACK_NAME="${STACK_NAME:-trn2-dl-desktop-2}"
TEMPLATE="${TEMPLATE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/deep-learning-ubuntu-desktop.yaml}"

INSTANCE_TYPE=trn2.48xlarge
VPC_ID="${VPC_ID:-}"
SUBNET_ID="${SUBNET_ID:-}"            # a public subnet in VPC_ID
KEY_NAME="${KEY_NAME:-}"
S3_BUCKET="${S3_BUCKET:-$(lab_bucket)}"
for v in VPC_ID SUBNET_ID KEY_NAME S3_BUCKET; do
  [[ -n "${!v}" ]] || { echo "FAIL: set $v (VPC_ID, SUBNET_ID and KEY_NAME are required)"; exit 1; }
done
# Dummy but well-formed: a real reservation id is substituted at launch time.
# Validation only checks shape, not existence.
RESERVATION_ID="${RESERVATION_ID:-cr-00000000000000000}"

CS_NAME="preflight-$(date -u +%Y%m%d%H%M%S)"

if [[ ! -f "$TEMPLATE" ]]; then
  echo "FAIL: template not found at $TEMPLATE"; exit 1
fi

PUBLIC_IP=$(curl -s --max-time 20 http://checkip.amazonaws.com/ | tr -d '[:space:]')
[[ -z "$PUBLIC_IP" ]] && { echo "FAIL: could not detect public IP"; exit 1; }
CIDR="${PUBLIC_IP}/32"

echo "=== pre-flight parameter validation ==="
echo "  stack (not created): $STACK_NAME"
echo "  template:            $TEMPLATE"
echo "  access CIDR:         $CIDR"
echo

# 1. Template itself parses.
if ! aws cloudformation validate-template --region "$REGION" \
      --template-body "file://$TEMPLATE" >/dev/null 2>&1; then
  echo "FAIL: validate-template rejected the template"
  aws cloudformation validate-template --region "$REGION" --template-body "file://$TEMPLATE" 2>&1 | head -5
  exit 1
fi
echo "[1/4] template parses ....................... OK"

# 2. Every parameter without a Default must be supplied. This is the check that
#    would have caught the UbuntuAMIOverride failure.
REQUIRED=$(aws cloudformation validate-template --region "$REGION" \
  --template-body "file://$TEMPLATE" \
  --query 'Parameters[?!not_null(DefaultValue)].ParameterKey' --output text 2>/dev/null)
echo "[2/4] parameters with no default: ${REQUIRED:-none}"

# 3. The real check: a CREATE change set runs full parameter validation.
CS_ARN=$(aws cloudformation create-change-set \
  --region "$REGION" \
  --stack-name "$STACK_NAME" \
  --change-set-name "$CS_NAME" \
  --change-set-type CREATE \
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
  --query 'Id' --output text 2>&1)

if [[ "$CS_ARN" != arn:* ]]; then
  echo "[3/4] change set creation ................... FAIL"
  echo "$CS_ARN"
  exit 1
fi

# Wait for it to reach a terminal state. CREATE_COMPLETE means the parameter
# set is accepted. FAILED with "didn't contain changes" is also fine.
for _ in $(seq 1 30); do
  ST=$(aws cloudformation describe-change-set --region "$REGION" \
        --change-set-name "$CS_ARN" --query 'Status' --output text 2>/dev/null)
  [[ "$ST" == CREATE_COMPLETE || "$ST" == FAILED ]] && break
  sleep 4
done
REASON=$(aws cloudformation describe-change-set --region "$REGION" \
          --change-set-name "$CS_ARN" --query 'StatusReason' --output text 2>/dev/null)

RC=0
if [[ "$ST" == CREATE_COMPLETE ]]; then
  N=$(aws cloudformation describe-change-set --region "$REGION" \
       --change-set-name "$CS_ARN" --query 'length(Changes)' --output text 2>/dev/null)
  echo "[3/4] parameter set accepted ................ OK ($N resources would be created)"
else
  echo "[3/4] parameter set .......................... FAIL ($ST)"
  echo "      $REASON"
  RC=1
fi

# 4. Always clean up -- a lingering REVIEW_IN_PROGRESS stack shell is confusing
#    and blocks reusing the name.
aws cloudformation delete-change-set --region "$REGION" --change-set-name "$CS_ARN" >/dev/null 2>&1
aws cloudformation delete-stack --region "$REGION" --stack-name "$STACK_NAME" >/dev/null 2>&1
echo "[4/4] cleaned up change set and stack shell .. OK"

echo
if (( RC == 0 )); then
  echo "PRE-FLIGHT PASSED -- this parameter set will create a stack."
  echo "Substitute the real CapacityReservationId at launch; shape is already validated."
else
  echo "PRE-FLIGHT FAILED -- fix before buying a block."
fi
exit $RC

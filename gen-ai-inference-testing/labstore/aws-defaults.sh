# Shared runtime lookups for the lab store. Sourced by labstore.sh and the
# labstore/bootstrap scripts so no account-specific value is kept in the repo.
#
# Every value can be set explicitly; otherwise it is resolved from the caller's
# own account at runtime:
#   AWS_REGION       explicit, else the CLI's configured region, else instance metadata
#   LAB_STORE_STACK  name of the store stack (lab-store.yaml), default dl-lab-store
#   LAB_S3_BUCKET    explicit, else the stack's <stack>-BucketName export, else the
#                    convention dl-lab-store-<account>-<region>
#   LAB_EFS_ID       explicit, else the stack's <stack>-EfsId export
#   LAB_ECR_URI      explicit, else the stack's <stack>-EcrRepositoryUri export

LAB_STORE_STACK="${LAB_STORE_STACK:-dl-lab-store}"

lab_region() {
  if [ -n "${AWS_REGION:-}" ]; then echo "$AWS_REGION"; return; fi
  local r
  r=$(aws configure get region 2>/dev/null)
  if [ -z "$r" ]; then
    local tok
    tok=$(curl -s -m 2 -X PUT http://169.254.169.254/latest/api/token \
            -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null)
    r=$(curl -s -m 2 -H "X-aws-ec2-metadata-token: $tok" \
            http://169.254.169.254/latest/meta-data/placement/region 2>/dev/null)
  fi
  echo "$r"
}

# Value of a CloudFormation export, or nothing if absent/unreadable.
lab_export() {
  local v
  v=$(aws cloudformation list-exports --region "$(lab_region)" \
        --query "Exports[?Name=='$1'].Value | [0]" --output text 2>/dev/null)
  [ -n "$v" ] && [ "$v" != "None" ] && echo "$v"
}

lab_account() { aws sts get-caller-identity --query Account --output text 2>/dev/null; }

lab_bucket() {
  if [ -n "${LAB_S3_BUCKET:-}" ]; then echo "$LAB_S3_BUCKET"; return; fi
  lab_export "${LAB_STORE_STACK}-BucketName" && return
  local acct
  acct=$(lab_account)
  [ -n "$acct" ] && echo "dl-lab-store-${acct}-$(lab_region)"
}

lab_efs_id() {
  if [ -n "${LAB_EFS_ID:-}" ]; then echo "$LAB_EFS_ID"; return; fi
  lab_export "${LAB_STORE_STACK}-EfsId"
}

lab_ecr_uri() {
  if [ -n "${LAB_ECR_URI:-}" ]; then echo "$LAB_ECR_URI"; return; fi
  lab_export "${LAB_STORE_STACK}-EcrRepositoryUri"
}

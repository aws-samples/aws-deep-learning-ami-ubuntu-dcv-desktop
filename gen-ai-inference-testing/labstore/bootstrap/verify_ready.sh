#!/usr/bin/env bash
# Poll the trn2 desktop via SSM until the Neuron driver is actually live.
#
# CREATE_COMPLETE only means CloudFormation made the resources. The advanced
# template has no wait condition on UserData, which installs aws-neuronx-dkms,
# DCV and the IDEs, then reboots. The Neuron kernel module only loads after
# that reboot, so /dev/neuron* is empty until then.
set -uo pipefail

# shellcheck source=../aws-defaults.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/aws-defaults.sh"
REGION="$(lab_region)"
IID="${IID:-${1:-}}"
[[ -n "$IID" ]] || { echo "usage: IID=<instance id> $0   (or: $0 <instance id>)"; exit 1; }
MAX_MIN="${MAX_MIN:-60}"
if [[ -z "${HOME:-}" ]]; then
  HOME="$(getent passwd "$(id -u)" 2>/dev/null | cut -d: -f6)"
  [[ -n "$HOME" ]] || HOME=/tmp
  export HOME
fi
LOG="${LOG:-$HOME/.cache/trn2-desktop/ready.log}"
mkdir -p "$(dirname "$LOG")" 2>/dev/null || LOG=/tmp/ready.log
exec > >(tee -a "$LOG") 2>&1

probe() {
  local cid
  cid=$(aws ssm send-command --region "$REGION" --instance-ids "$IID" \
    --document-name AWS-RunShellScript \
    --parameters 'commands=[
      "printf \"devnodes=\"; ls -1 /dev/neuron* 2>/dev/null | wc -l",
      "printf \"driver=\"; (lsmod 2>/dev/null | grep -qi neuron && echo yes || echo no)",
      "printf \"ready_banner=\"; (grep -qi \"Deep Learning Desktop is Ready\" /var/log/cloud-init-output.log 2>/dev/null && echo yes || echo no)",
      "printf \"cloudinit=\"; (cloud-init status 2>/dev/null | head -1 || echo unknown)",
      "printf \"uptime=\"; awk \"{printf \\\"%d\\\", \\$1/60}\" /proc/uptime; echo \" min\"",
      "printf \"dcv=\"; (systemctl is-active dcvserver 2>/dev/null || echo inactive)"
    ]' --query 'Command.CommandId' --output text 2>/dev/null) || return 1
  [[ -z "$cid" ]] && return 1
  local st=""
  for _ in $(seq 1 25); do
    st=$(aws ssm get-command-invocation --region "$REGION" --command-id "$cid" \
          --instance-id "$IID" --query 'Status' --output text 2>/dev/null)
    [[ "$st" == "Success" || "$st" == "Failed" ]] && break
    sleep 5
  done
  aws ssm get-command-invocation --region "$REGION" --command-id "$cid" \
    --instance-id "$IID" --query 'StandardOutputContent' --output text 2>/dev/null
}

echo "=== waiting for Neuron driver, started $(date -u +%H:%M:%SZ) ==="
START=$(date +%s)
while (( ($(date +%s) - START) / 60 < MAX_MIN )); do
  OUT=$(probe)
  if [[ -z "$OUT" ]]; then
    # Expected during the automatic reboot: SSM goes away briefly.
    echo "[$(date -u +%H:%M:%SZ)] no SSM response (instance may be rebooting)"
  else
    echo "[$(date -u +%H:%M:%SZ)] $(echo "$OUT" | tr '\n' ' ')"
    NODES=$(echo "$OUT" | sed -n 's/^devnodes=//p' | tr -d ' ')
    if [[ -n "$NODES" && "$NODES" -gt 0 ]]; then
      echo
      echo "=== NEURON DEVICES PRESENT: $NODES ==="
      break
    fi
  fi
  sleep 60
done

echo
echo "=== final verification ==="
cid=$(aws ssm send-command --region "$REGION" --instance-ids "$IID" \
  --document-name AWS-RunShellScript \
  --parameters 'commands=[
    "echo === neuron-ls ===",
    "/opt/aws/neuron/bin/neuron-ls 2>&1 | head -40",
    "echo \"=== O-1: nc_count (physical 8 or logical 4 per device?) ===\"",
    "/opt/aws/neuron/bin/neuron-ls --json-output 2>/dev/null | grep -m2 nc_count",
    "echo === device nodes ===",
    "ls -1 /dev/neuron* 2>/dev/null | wc -l",
    "echo === neuron pkgs ===",
    "dpkg -l | grep -i neuronx | awk \"{print \\$2, \\$3}\"",
    "echo === efs mount ===",
    "df -h /home/ubuntu/efs 2>/dev/null | tail -1",
    "echo === docker ===",
    "docker --version 2>/dev/null || echo docker-missing",
    "echo === dcv ===",
    "systemctl is-active dcvserver 2>/dev/null || echo inactive"
  ]' --query 'Command.CommandId' --output text 2>/dev/null)
for _ in $(seq 1 25); do
  st=$(aws ssm get-command-invocation --region "$REGION" --command-id "$cid" \
        --instance-id "$IID" --query 'Status' --output text 2>/dev/null)
  [[ "$st" == "Success" || "$st" == "Failed" ]] && break
  sleep 5
done
aws ssm get-command-invocation --region "$REGION" --command-id "$cid" \
  --instance-id "$IID" --query 'StandardOutputContent' --output text 2>/dev/null

if command -v osascript >/dev/null 2>&1; then
  osascript -e 'display notification "Neuron driver check finished" with title "trn2 desktop" sound name "Glass"' >/dev/null 2>&1 || true
fi

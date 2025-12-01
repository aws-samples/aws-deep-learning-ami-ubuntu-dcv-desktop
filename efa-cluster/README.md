# Deep Learning Cluster with EFA and Open MPI

This guide explains how to launch and use a deep-learning cluster enabled with [Elastic Fabric Adapter (EFA)](https://aws.amazon.com/hpc/efa/) and [Open MPI](https://www.open-mpi.org/) for distributed training workloads.

## Prerequisites

* Successfully deployed deep learning desktop using the main [README](../README.md)
* Desktop CloudFormation stack must be in `CREATE_COMPLETE` status

## Launching the Cluster

Launch the CloudFormation stack template [deep-learning-ubuntu-efa-cluster.yaml](deep-learning-ubuntu-efa-cluster.yaml) after the desktop stack is complete. See [Cluster Parameters](#cluster-parameters) for template input parameters.

## Using Open MPI on the Desktop Head Node

To run the Open MPI `mpirun` command from the desktop head node, configure password-less SSH to cluster nodes using SSH agent-forwarding (recommended to avoid storing private keys on the head node).

### Setup SSH Agent-Forwarding

1. Add your SSH private key to the SSH forwarding agent on your laptop:

```bash
ssh-add ~/.ssh/id_rsa
```

If your private key is not in the default `~/.ssh/id_rsa` location, adjust the command accordingly.

2. Add the following configuration to `~/.ssh/config` on the desktop head node:

```
Host *
    ForwardAgent yes
Host *
    StrictHostKeyChecking no
```

3. SSH to the desktop head node with agent forwarding enabled:

```bash
ssh -A ubuntu@desktop-ec2-public-address
```

Now you can run `mpirun` commands from the desktop head node targeting any deep-learning cluster.

### Creating the Hostfile

To run `mpirun`, you need a `hostfile` containing host IP addresses and slots for each cluster node. Use this bash script to generate it:

```bash
#!/bin/bash

[[ $# != 2 ]] && echo "usage: $0 aws-region ec2-autoscaling-group-name" && exit

region=$1
asg=$2

for ID in $(aws autoscaling describe-auto-scaling-instances --region $region --query "AutoScalingInstances[?AutoScalingGroupName=='$asg'].InstanceId" --output text);
do
    host=$(aws ec2 describe-instances --instance-ids $ID --region $region --query "Reservations[].Instances[].PrivateIpAddress" --output text)
    echo "$host	slots=1"
done
```

The `ec2-autoscaling-group-name` is available in the cluster stack output.

## Open MPI Example

Before running MPI jobs, SSH into each instance in your `hostfile` and verify you see the message `Cluster node is ready!`. If not, wait about 10 minutes and try again.

**NOTE:** Adjust `PATH` and `LD_LIBRARY_PATH` for your specific use case.

```bash
#!/bin/bash

NUM_PARALLEL=2
DATE=`date '+%Y-%m-%d-%H-%M-%S'`
export JOB_ID=mpirun-test-$DATE

mpirun -np $NUM_PARALLEL --verbose \
--hostfile /home/ubuntu/efs/openmpi/hostfile \
-bind-to none -map-by slot \
--mca plm_rsh_no_tree_spawn 1 -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude lo,docker0 \
--mca hwloc_base_binding_policy none --mca rmaps_base_mapping_policy slot \
--mca orte_keep_fqdn_hostnames t \
--output-filename /home/ubuntu/efs/logs/${JOB_ID} \
--display-map --tag-output --timestamp-output \
-wdir /home/ubuntu \
-x PATH='/usr/local/cuda-12.8/bin:/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin' \
-x LD_LIBRARY_PATH='/usr/local/cuda-12.8/lib64:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib' \
bash -c "source /home/ubuntu/miniconda3/etc/profile.d/conda.sh  && conda activate base && hostname && env"
```

## <a name="cluster-parameters"></a> Cluster CloudFormation Template Parameters

| Parameter Name | Parameter Description |
| --- | ----------- |
| AWSUbuntuAMIType | **Required**. Selects the AMI type. Default is [AWS Deep Learning AMI (Ubuntu 18.04)](https://aws.amazon.com/marketplace/pp/Amazon-Web-Services-AWS-Deep-Learning-AMI-Ubuntu-1/B07Y43P7X5). |
| ASGMaxSize | **Required**. Maximum size for the cluster's EC2 auto-scaling group. |
| ASGDesiredSize | **Required**. Current desired size for the cluster's EC2 auto-scaling group. |
| ClusterInstanceType | **Required**. Amazon EC2 instance type. G3, G4, P3 and P4 instance types are GPU enabled. |
| ClusterSubnetId | **Required**. Amazon VPC subnet. Must be private with NAT gateway access enabled. |
| ClusterSubnetAZ | **Required**. Availability Zone used by the `ClusterSubnetId` subnet. |
| EBSOptimized | **Required**. Enable [network optimization for EBS](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-optimized.html) (default is **true**). |
| EFSFileSystemId | *Optional* advanced parameter. Existing EFS file-system id with [existing network mount target](https://docs.aws.amazon.com/efs/latest/ug/how-it-works.html#how-it-works-ec2) accessible from your DesktopVpcSubnetId. Use with DesktopSecurityGroupId. Leave blank to create new EFS file-system. |
| EFSMountPath | Absolute path where EFS file-system is mounted (default is `/home/ubuntu/efs`). |
| EbsVolumeSize | **Required**. Size of the EBS volume (default is 200 GB). |
| EbsVolumeType | **Required**. [EBS volume type](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-volume-types.html) (default is gp3). |
| UbuntuAMIOverride | *Optional* advanced parameter to override the AMI. Leave blank to use default AMIs for your region. |

## Cluster CloudFormation Stack Outputs

| Output Key | Output Description |
| --- | ----------- |
| Asg | EC2 auto-scaling group for the cluster. |

## Deleting the Cluster

Delete the cluster CloudFormation stack from the AWS CloudFormation console when no longer needed. This will terminate EC2 instances and delete root EBS volumes.

**Note:** The EFS file system is **not** automatically deleted when you delete the stack.

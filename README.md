# AWS Deep Learning Desktop with NICE DCV

This project is a tutorial on how to launch an AWS deep learning desktop with [NICE DCV](https://aws.amazon.com/hpc/dcv/) for developing, training, testing, and visualizing deep learning models. To launch the deep learning desktop, you have a choice of two AMIs:

* Ubuntu Server Pro, 22.04 LTS, Version 20231025 (Default)
* Ubuntu Server Pro, 20.04 LTS, Version 20231030

Deep-learning desktop supports Amazon EC2 [trn1](https://aws.amazon.com/ec2/instance-types/trn1/), [inf2](https://aws.amazon.com/ec2/instance-types/inf2/), GPU enabled [g3](https://aws.amazon.com/ec2/instance-types/g3/), [g4](https://aws.amazon.com/ec2/instance-types/g4/), [g5](https://aws.amazon.com/ec2/instance-types/g5/), [p3](https://aws.amazon.com/ec2/instance-types/p3/), and [p4](https://aws.amazon.com/ec2/instance-types/p4/), and selected [m5](https://aws.amazon.com/ec2/instance-types/m5/), [c5](https://aws.amazon.com/ec2/instance-types/c5/), and [r5](https://aws.amazon.com/ec2/instance-types/r5/) instance families.

For Amazon EC2 [trn1](https://aws.amazon.com/ec2/instance-types/trn1/) and [inf2](https://aws.amazon.com/ec2/instance-types/inf2/) instance types, [AWS Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/) with PyTorch support is automatically installed in a virtual environment named `aws_neuron_venv_pytorch`.  For all other types of EC2 instances, [conda](https://docs.conda.io/en/latest/miniconda.html) environment for [Tensorflow 2.14.0](https://www.tensorflow.org/) is installed in a conda environment `tensorflow`, and [PyTorch 2.1.0](https://pytorch.org/) is installed in a conda environment named `pytorch`. Both conda environments have [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) installed. 

For Amazon EC2 GPU enabled instance types, CUDA Compatibility package compatible with the CUDA driver, [CUDA](https://developer.nvidia.com/cuda-toolkit), and [cuDNN](https://developer.nvidia.com/cudnn) are automatically installed.

**NOTE:** *Automatic or manual upgrades to CUDA driver may require new [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) package to be manually installed.*

[Visual Studio Code](https://code.visualstudio.com/) IDE is installed for code development.

The deep-learning desktop can be used for standalone development, and can be also used as a head node for working with one or more [deep-learning clusters](#launching-deep-learning-cluster-with-efa-and-open-mpi) enabled with [Elastic Fabric Adapter (EFA)](https://aws.amazon.com/hpc/efa/) and [Open MPI](https://www.open-mpi.org/). 

## Step by Step Tutorial

### <a name="Prerequisites"></a> Prerequisites
This tutorial assumes you have an [AWS Account](https://aws.amazon.com/account/), and you have [Administrator job function](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_job-functions.html) access to the AWS Management Console.

To get started:

* Select your [AWS Region](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html). The AWS Regions supported by this project include, us-east-1, us-east-2, us-west-2, eu-west-1, eu-central-1, ap-southeast-1, ap-southeast-2, ap-northeast-1, ap-northeast-2, and ap-south-1. Note that not all Amazon EC2 instance types are available in all [AWS Availability Zones](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html) in an AWS Region.
* If you do not already have an Amazon EC2 key pair, [create a new Amazon EC2 key pair](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html#prepare-key-pair). You will need the key pair name to specify the ```KeyName``` parameter when creating the CloudFormation stack below.
* You will need an [Amazon S3](https://aws.amazon.com/s3/) bucket. If you don't have one, [create a new Amazon S3 bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html) in the AWS region you selected. The S3 bucket can be empty at this point.
* Use [AWS check ip](http://checkip.amazonaws.com/) to get your public IP address. This will be the IP address you will need to specify ```DesktopAccessCIDR``` parameter in the stack. 
* Clone this Git repository on your laptop using [```git clone ```](https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository).

### Create AWS CloudFormation Stack
Use the AWS CloudFormation template [deep-learning-ubuntu-desktop.yaml](deep-learning-ubuntu-desktop.yaml) from your cloned  repository to create a new CloudFormation stack using the [ AWS Management console](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/cfn-console-create-stack.html), or using the [AWS CLI](https://docs.aws.amazon.com/cli/latest/reference/cloudformation/create-stack.html). See [Reference](#Reference) section for the [template](deep-learning-ubuntu-desktop.yaml) input parameters, and stack outputs.

The template [deep-learning-ubuntu-desktop.yaml](deep-learning-ubuntu-desktop.yaml) creates [AWS Identity and Access Management (IAM)](https://aws.amazon.com/iam/) resources. If you are creating CloudFormation Stack using the console, in the review step, you must check 
**I acknowledge that AWS CloudFormation might create IAM resources.** If you use the ```aws cloudformation create-stack``` CLI, you must use ```--capabilities CAPABILITY_NAMED_IAM```. 

### Connect to Desktop using SSH

* Once the stack status in CloudFormation console is ```CREATE_COMPLETE```, find the deep learning desktop instance launched in your stack in the Amazon EC2 console, and [connect to the instance using SSH](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html) as user ```ubuntu```, using your SSH key pair.
* When you connect using SSH, and you see the message ```"Cloud init in progress. Machine will REBOOT after cloud init is complete!!"```, disconnect and try later after about 15 minutes. The desktop installs the NICE DCV server on first-time startup, and reboots after the install is complete.
* If you see the message ```NICE DCV server is enabled!```, run the command ```sudo passwd ubuntu``` to set a new password for user ```ubuntu```. Now you are ready to connect to the desktop using the [NICE DCV client](https://docs.aws.amazon.com/dcv/latest/userguide/client.html)

#### NOTE
The deep-learning desktop uses EC2 [user-data](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html) to automatically install the required software in the desktop instance. The log output of this automatic installation is available in `/var/log/cloud-init-output.log` file. Most *transient* failures in the automatic user-data installation can be fixed by rebooting the instance.

### Connect to Desktop using NICE DCV Client
* Download and install the [NICE DCV client](https://docs.aws.amazon.com/dcv/latest/userguide/client.html) on your laptop.
* Use the NICE DCV Client to login to the desktop as user ```ubuntu```
* When you first login to the desktop using the NICE DCV client, you will be asked if you would like to upgrade the OS version. **Do not upgrade the OS version** .

## Working with Data

The deep learning desktop instance has access to the S3 bucket you specified when you create the CloudFormation stack. You can verify the access to your S3 bucket by running the command ```aws s3 ls your-bucket-name```. If you do not have access to the S3 bucket, you will see an error message. If your S3 bucket is empty, the previous command will produce no output, which is normal. 

There is an [Amazon EBS](https://aws.amazon.com/ebs/) root volume attached to the instance. In addition, an [Amazon EFS](https://aws.amazon.com/efs/) file-system is mounted on your desktop at ```EFSMountPath```, which by default is ```/home/ubuntu/efs```. Optionally, an [Amazon FSx for Lustre](https://aws.amazon.com/fsx/) file-system can be mounted on your desktop at ```FSxMountPath```, which by default is ```/home/ubuntu/fsx```.  See ```FSxForLustre``` parameter in [Reference](#Reference) section to learn how to enable FSx for Lustre file-system.

The Amazon EBS volume attached to the instance is deleted when the deep learning instance is terminated. However, the EFS file-system persists after you terminate the desktop instance. 

## Using Amazon SageMaker from deep learning desktop
The deep learning desktop is pre-configured to use [Amazon SageMaker](https://aws.amazon.com/sagemaker/) machine learning platform. To get started with [Amazon SageMaker examples](https://github.com/aws/amazon-sagemaker-examples) in a [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html) notebook, execute following steps in a desktop terminal:

	mkdir ~/git
	cd ~/git
	git clone https://github.com/aws/amazon-sagemaker-examples.git
	jupyter-lab
	
This will start a ```jupyter-lab``` notebook server in the terminal, and open a tab in your web browser. You can now explore the Amazon SageMaker examples. 

For SageMaker examples that require you to specify a [subnet](https://docs.aws.amazon.com/vpc/latest/userguide/VPC_Subnets.html#vpc-subnet-basics), and a [security group](https://docs.aws.amazon.com/vpc/latest/userguide/VPC_SecurityGroups.html), use the pre-configured environment variables  ```desktop_subnet_id``` and ```desktop_sg_id```, respectively. If FSx for Lustre is enabled, pre-configured environment variable  ```fsx_fs_id``` contains FSx for Lustre file-system id, and ```fsx_mount_name``` variable contains the mount name.

## Stopping and Restarting the Desktop

You may safely reboot, stop, and restart the desktop instance at any time. The desktop will automatically mount the EFS file-system at restart. If FSx for Luster file-system is enabled, it is automatically mounted, as well.

## Launching Deep-learning Cluster with EFA and Open MPI

The CloudFormation stack template for [deep-learning cluster enabled with EFA and Open MPI](deep-learning-ubuntu-efa-cluster.yaml) can be launched after the desktop CloudFormation Stack launch is successfully completed. See [Reference](#Reference) for deep-learning cluster CloudFormation template input parameters.

### Using Open MPI on the Desktop Head Node

To be able to run the Open MPI `mpirun` command on the desktop head node, we need to configure password-less `ssh` to the cluster nodes. Ideally, we want to do this without storing the SSH private key on the desktop head node. To accomplish this objective, we recommend using SSH agent-forwarding on your laptop. To use SSH agent-forwarding, first add your SSH private key to the SSH forwarding agent on your laptop, using the following command:

	ssh-add ~/.ssh/id_rsa


If your private key is not stored in the default `~/.ssh/id_rsa` file, change the previous command, accordingly. 

Next, add following configuration to the `~/.ssh/config` file on the desktop head node,:

	Host *
    	ForwardAgent yes
	Host *
		StrictHostKeyChecking no

To run the `mpirun` command on the desktop head node, `ssh` from your laptop to the desktop head node as follows:

	ssh -A ubuntu@desktop-ec2-public-address

Now you can run the `mpirun` command from the desktop head node, targeting any deep-learning cluster. 

To run the `mpirun` command, you will need a `hostfile` containing the host IP addresses and slots for each node in your target cluster. Following  `bash` code can help you create your `hostfile`, where the `ec2-autoscaling-group-name` argument below is the EC2 auto-scaling group name for your deep-learning cluster, and is available in cluster stack output:

	#!/bin/bash

	[[ $# != 2 ]] && echo "usage: $0 aws-region ec2-autoscaling-group-name" && exit

	region=$1
	asg=$2

	for ID in $(aws autoscaling describe-auto-scaling-instances --region $region --query "AutoScalingInstances[?AutoScalingGroupName=='$asg'].InstanceId" --output text);
	do
		host=$(aws ec2 describe-instances --instance-ids $ID --region $region --query "Reservations[].Instances[].PrivateIpAddress" --output text)
		echo "$host	slots=1"
	done

#### Open MPI Examples

Before you run the MPI examples, `ssh` into each instance in your `hostfile`, and make sure you see the message `Cluster node is ready!`. If you do not see this message, exit and `ssh` again in about 10 minutes to verify the cluster node is ready.

##### Neuron PyTorch Example

The `mpirun` example below targets the `aws_neuron_venv_pytorch` virtual environment on a 2-node `trn1.32xlarge` cluster. It is assumed that the cluster `hostfile` is stored in `/home/ubuntu/efs/openmpi/hostfile`, and the output directory `/home/ubuntu/efs/logs` exits. 

**NOTE:** The `PATH` and `LD_LIBRARY_PATH` defined below apply to  the cluster nodes.

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
	-x PATH='/opt/aws/neuron/bin:/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin' \
	-x LD_LIBRARY_PATH='/opt/aws/neuron/lib:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib' \
	bash -c "source /home/ubuntu/aws_neuron_venv_pytorch/bin/activate && hostname && env"

#### CUDA TensorFlow Example

The `mpirun` example below targets the conda `tensorFlow` environment on a 2-node CUDA GPU cluster. 

**NOTE:** The `PATH` and `LD_LIBRARY_PATH` defined below apply to  the cluster nodes.

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
	-x PATH='/usr/local/cuda-11.8/bin:/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin' \
	-x LD_LIBRARY_PATH='/usr/local/cuda-12.2/compat:/usr/local/cuda-12.1/compat:/usr/local/cuda-11.8/lib64:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib' \
	bash -c "source /home/ubuntu/miniconda3/etc/profile.d/conda.sh  && conda activate tensorflow && hostname && env"


## Deleting the Stacks

When you no longer need the stacks, you may delete the AWS CloudFormation stacks from the AWS CloudFormation console. Deleting the stack will terminate the EC2 instances, and delete the root EBS volume attached to the instances. If the FSx for Lustre file-system is enabled, it is also automatically deleted when you delete the stack.

The EFS file system is **not** automatically deleted when you delete the stack.

## <a name="Reference"></a> Reference

### Desktop CloudFormation Template Input Parameters
Below, we describe the AWS CloudFormation template input parameters for the deep-learning desktop.

| Parameter Name | Parameter Description |
| --- | ----------- |
| AWSUbuntuAMIType | This is a required parameter that selects the AMI type. Default AMI type is [AWS Deep Learning AMI (Ubuntu 18.04)](https://aws.amazon.com/marketplace/pp/Amazon-Web-Services-AWS-Deep-Learning-AMI-Ubuntu-1/B07Y43P7X5). |
| DesktopAccessCIDR | This parameter specifies the public IP CIDR range from where you need access to your deep learning desktop, e.g. 1.2.3.4/32, or 7.8.0.0/16. This parameter is *ignored* if you specify the optional parameter  DesktopSecurityGroupId.|
| DesktopHasPublicIpAddress | This is a **required** parameter whereby you specify if the desktop has a public internet address. Unless you have AWS [VPN](https://aws.amazon.com/vpn/) or [DirectConnect](https://aws.amazon.com/directconnect) access enabled, you must set this parameter to "true".
| DesktopInstanceType | This is a **required** parameter whereby you select an Amazon EC2 instance type. G3, G4, P3 and P4 instance types are GPU enabled.  |
| DesktopSecurityGroupId | This is an *optional* advanced parameter whereby you specify Amazon EC2 security group for your desktop. The specified security group must allow inbound access over ports 22 (SSH) and 8443 (DCV) from ```DesktopAccessCIDR```, access to EFS and FSx for Lustre, and allow all inbound and outbound network traffic within the security group. Leave it blank to automatically create a new security group, which enables access for SSH, DCV, EFS, FSx for Lustre, and allows all inbound and outbound network traffic within the security group. |
| DesktopVpcId | This is a **required** parameter whereby you select your Amazon VPC id.|
| DesktopVpcSubnetId | This is a **required** parameter whereby you select your Amazon VPC subnet. The specified subnet must be public with Internet Gateway access enabled if you plan to access your desktop over the Internet, or private with NAT gateway access enabled. |
| EBSOptimized | This is a **required** parameter whereby you select if you want your desktop instance to be [network optimized for EBS](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-optimized.html) (default is **true**)|
| EFSFileSystemId | This is an *optional* advanced parameter whereby you specify an existing EFS file-system id with an [existing network mount target](https://docs.aws.amazon.com/efs/latest/ug/how-it-works.html#how-it-works-ec2)  accessible from your DesktopVpcSubnetId. If you specify this parameter, do it in conjunction with DesktopSecurityGroupId. Leave it blank to create a new EFS file-system.  |
| EFSMountPath | Absolute path for the directory where EFS file-system is mounted (default is ```/home/ubuntu/efs```).   |
| EbsVolumeSize | This is a **required** parameter whereby you specify the size of the EBS volume (default size is 200 GB). Typically, the default size is sufficient.|
| EbsVolumeType | This is a **required** parameter whereby you select the [EBS volume type](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-volume-types.html) (default is gp3). |
| FSxCapacity | This is an **optional** parameter whereby you specify the capacity of the FSx for Lustre file-system. This capacity must be in multiples of 1200 GB. Default capacity is 1200 GB. See ```FSxForLustre``` parameter to enable FSx for Lustre file-system. | 
| FSxForLustre | This is an **optional** parameter whereby you  enable, disable FSx for Lustre file-system. By default, it is disabled. If enabled, a FSx for Lustre file-system is created and mounted on the desktop. The FSx for Lustre file-system automatically imports data from ```s3://S3bucket/S3Import```. See ```S3Bucket``` and ```S3Import``` parameters. |
| FSxMountPath | FSx file-system mount directory path (default is ```/home/ubuntu/fsx```).   |
| KeyName | This is a **required** parameter whereby you select the Amazon EC2 key pair name used for SSH access to the desktop. You must have access to the selected key pair's private key to connect to your desktop. |
| S3Bucket | This is a **required** parameter whereby you specify the name of the Amazon S3 bucket used to store your data. The S3 bucket may be empty at the time you create the AWS CloudFormation stack.  |
| S3Import | This is an **optional** parameter whereby you specify S3 import prefix for FSx file-system. See ```FSxForLustre``` parameter to enable FSx for Lustre file-system.  |
| UbuntuAMIOverride | This is an *optional* advanced parameter to override the AMI. Leave blank to use default AMIs for your region. See parameter ```AWSUbuntuAMIType```. |

### Desktop CloudFormation Stack Outputs
Below, we describe the Desktop CloudFormation Stack outputs. All output variables are exported prefixed with the stack name.

| Output Key | Output Description |
| --- | ----------- |
| Ami | Desktop AMI id. |
| VpcId | Desktop VPC id |
| KeyPairName | Desktop EC2 key pair name |
| InstanceProfileArn | Desktop EC2 Instance Profile ARN |
| SecurityGroupId | Desktop security group id. |
| EfsId | Desktop EFS file-system Id. |
| EfsMountPath | Desktop EFS file-system mount path. |
| FsxId | Desktop FSx Lustre file-system id, if FSx Luster is enabled. |
| FsxMountName | Desktop FSx Lustre file-system mount name, if FSx Luster is enabled. |
| FsxMountPath | Desktop FSx Lustre file-system mount path, if FSx Luster is enabled. |

### Cluster CloudFormation Template Input Parameters
Below, we describe the AWS CloudFormation template input parameters for the deep-learning cluster.

| Parameter Name | Parameter Description |
| --- | ----------- |
| AWSUbuntuAMIType | This is a **required** parameter that selects the AMI type. Default AMI type is [AWS Deep Learning AMI (Ubuntu 18.04)](https://aws.amazon.com/marketplace/pp/Amazon-Web-Services-AWS-Deep-Learning-AMI-Ubuntu-1/B07Y43P7X5). |
| ASGMaxSize | This is a **required** parameter that specifies maximum size for the cluster's EC2 auto-scaling group. |
| ASGDesiredSize | This is a **required** parameter that specifies the current desired size for the the cluster's EC2 auto-scaling group. |
| ClusterInstanceType | This is a **required** parameter whereby you select an Amazon EC2 instance type. G3, G4, P3 and P4 instance types are GPU enabled.  |
| ClusterSubnetId | This is a **required** parameter whereby you select your Amazon VPC subnet. The specified subnet must be private with NAT gateway access enabled. |
| ClusterSubnetAZ | This is a **required** parameter whereby you specify the AZ used by the `ClusterSubnetId` subnet. |
| EBSOptimized | This is a **required** parameter whereby you select if you want your desktop instance to be [network optimized for EBS](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-optimized.html) (default is **true**)|
| EFSFileSystemId | This is an *optional* advanced parameter whereby you specify an existing EFS file-system id with an [existing network mount target](https://docs.aws.amazon.com/efs/latest/ug/how-it-works.html#how-it-works-ec2)  accessible from your DesktopVpcSubnetId. If you specify this parameter, do it in conjunction with DesktopSecurityGroupId. Leave it blank to create a new EFS file-system.  |
| EFSMountPath | Absolute path for the directory where EFS file-system is mounted (default is ```/home/ubuntu/efs```).   |
| EbsVolumeSize | This is a **required** parameter whereby you specify the size of the EBS volume (default size is 200 GB). Typically, the default size is sufficient.|
| EbsVolumeType | This is a **required** parameter whereby you select the [EBS volume type](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-volume-types.html) (default is gp3). |
| UbuntuAMIOverride | This is an *optional* advanced parameter to override the AMI. Leave blank to use default AMIs for your region. See parameter ```AWSUbuntuAMIType```. |

### AWS CloudFormation Stack Outputs
Below, we describe the AWS CloudFormation Stack outputs.

| Output Key | Output Description |
| --- | ----------- |
| Asg | EC2 auto-scaling group for the cluster. |

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.


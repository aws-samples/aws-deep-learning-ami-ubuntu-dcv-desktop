# AWS Deep Learning Desktop with Amazon DCV

Launch an AWS deep learning desktop with [Amazon DCV](https://aws.amazon.com/hpc/dcv/) for developing, training, testing, and visualizing deep learning, and generative AI models.

## Overview

**Supported AMIs:**
* Ubuntu Server Pro 24.04 LTS, Version 20250516 (Default)
* Ubuntu Server Pro 22.04 LTS, Version 20250516

**Supported EC2 Instance Types:**
* **Trainium/Inferentia:** [trn1](https://aws.amazon.com/ec2/instance-types/trn1/), [trn2](https://aws.amazon.com/ec2/instance-types/trn2/), [inf2](https://aws.amazon.com/ec2/instance-types/inf2/)
* **GPU:** [g4](https://aws.amazon.com/ec2/instance-types/g4/), [g5](https://aws.amazon.com/ec2/instance-types/g5/), [g6](https://aws.amazon.com/ec2/instance-types/g6/), [p3](https://aws.amazon.com/ec2/instance-types/p3/), [p4](https://aws.amazon.com/ec2/instance-types/p4/), [p5](https://aws.amazon.com/ec2/instance-types/p5/)
* **General Purpose:** Selected [m5](https://aws.amazon.com/ec2/instance-types/m5/), [c5](https://aws.amazon.com/ec2/instance-types/c5/), [r5](https://aws.amazon.com/ec2/instance-types/r5/)

**Key Features:**
* Generative AI [Inference](#generative-ai-inference-testing) and [Training](#generative-ai-training-testing)
* [Amazon SageMaker AI](#amazon-sagemaker-ai) integration

## Getting Started

### Prerequisites

**Requirements:**
* [AWS Account](https://aws.amazon.com/account/) with [Administrator job function](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_job-functions.html) access

**Supported AWS Regions:**
us-east-1, us-east-2, us-west-2, eu-west-1, eu-central-1, ap-southeast-1, ap-southeast-2, ap-northeast-1, ap-northeast-2, ap-south-1

**Note:** Not all EC2 instance types are available in all [Availability Zones](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html).

**Setup Steps:**

1. **Select your [AWS Region](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html)** from the supported regions above

2. **VPC and Subnets:** [Create a VPC](https://docs.aws.amazon.com/vpc/latest/userguide/create-vpc.html#create-vpc-only) or use an existing one. If needed, [create three public subnets](https://docs.aws.amazon.com/vpc/latest/userguide/create-subnets.html) in three different Availability Zones

3. **EC2 Key Pair:** [Create an EC2 key pair](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html#prepare-key-pair) if you don't have one (needed for `KeyName` parameter)

4. **S3 Bucket:** [Create an S3 bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html) in your selected region (can be empty initially)

5. **Get Your Public IP:** Use [AWS check ip](http://checkip.amazonaws.com/) to find your public IP address (needed for `DesktopAccessCIDR` parameter)

6. **Clone Repository:** Clone this repository to your laptop:
   ```bash
   git clone <repository-url>
   ```

### Launch the Desktop

Create a CloudFormation stack using the [deep-learning-ubuntu-desktop.yaml](deep-learning-ubuntu-desktop.yaml) template via:
* [AWS Management Console](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/cfn-console-create-stack.html), or
* [AWS CLI](https://docs.aws.amazon.com/cli/latest/reference/cloudformation/create-stack.html)

See [CloudFormation Parameters](#desktop-cloudformation-template-parameters) for template inputs and [Stack Outputs](#desktop-cloudformation-stack-outputs) for outputs.

**Important:** The template creates [IAM](https://aws.amazon.com/iam/) resources:
* **Console:** Check "I acknowledge that AWS CloudFormation might create IAM resources" during review
* **CLI:** Use `--capabilities CAPABILITY_NAMED_IAM` flag 

### Connect via SSH

1. Wait for stack status to show `CREATE_COMPLETE` in CloudFormation console
2. Find your desktop instance in EC2 console
3. [Connect via SSH](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html) as user `ubuntu` using your key pair

**First-time Setup:**
* If you see `"Cloud init in progress! Logs: /var/log/cloud-init-output.log"`, disconnect and wait ~15 minutes. The desktop installs Amazon DCV server and reboots automatically.
* When you see `Deep Learning Desktop is Ready!`, set a password:
  ```bash
  sudo passwd ubuntu
  ```

**Troubleshooting:** The desktop uses EC2 [user-data](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html) for automatic software installation. Check logs at `/var/log/cloud-init-output.log`. Most transient failures can be fixed by rebooting the instance.

### Connect via Amazon DCV Client

1. Download and install the [Amazon DCV client](https://docs.aws.amazon.com/dcv/latest/userguide/client.html) on your laptop
2. Login to the desktop as user `ubuntu`
3. **Do not upgrade the OS version** when prompted on first login
4. Configure **Software Updater** to only apply security updates automatically (avoid non-security updates unless you're an advanced user)

## Using the Desktop

### Generative AI Inference Testing

The desktop provides comprehensive inference testing frameworks for LLMs and embedding models. See [Inference Testing Guide](./gen-ai-inference-testing/README.md) for complete documentation.

**Note :** 
Once you have successfully connected to the Deep Learning Desktop with DCV client, perform the below actions.

1. Open Visual Studio Code (it's already installed in the Desktop) 
2. Clone the project's git repository

```bash
   git clone <repository-url>
```

ALternatively, you can also choose to run the inference and training Jupyter notebooks using Kiro which is also pre-installed in the Desktop.

**Supported Inference Servers:**
* [Triton Inference Server](https://github.com/triton-inference-server) - NVIDIA's production inference server
* [DJL Serving](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/index.html) - Deep Java Library with LMI
* OpenAI-compatible Server - Standard OpenAI API interface

**Supported Backends:**
* [vLLM](https://github.com/vllm-project/vllm) - High-performance inference (GPU and Neuron)
* [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - Optimized for NVIDIA GPUs
* Custom Python backends for embeddings

**Key Features:**
* Docker containers for all server/backend combinations
* Locust-based load testing with configurable concurrency
* Automatic model caching to EFS
* Hardware auto-detection (CUDA GPUs or Neuron devices)
* Performance metrics with latency, throughput, and error rates

### Generative AI Training Testing

The desktop provides four frameworks for fine-tuning LLMs with PEFT (LoRA) or full fine-tuning. See [Training Testing Guide](./gen-ai-training-testing/README.md) for complete documentation.

**Available Frameworks:**

| Framework | Key Features |
|-----------|---------------|
| [NeMo 2.0](./gen-ai-training-testing/nemo2/README.md) | Tensor/pipeline parallelism, Megatron-LM optimizations |
| [PyTorch Lightning](./gen-ai-training-testing/ptl/README.md) |  Full control, flexible callbacks |
| [Accelerate](./gen-ai-training-testing/accelerate/README.md) |  Simple API, minimal code |
| [Ray Train](./gen-ai-training-testing/ray_train/README.md) | Distributed orchestration, auto-recovery |

**Common Features:**
* Generalized HuggingFace dataset pipeline with flexible templates
* Multi-node, multi-GPU distributed training with FSDP
* LoRA and full fine-tuning support
* Automatic checkpoint conversion to HuggingFace format
* Comprehensive testing and evaluation scripts
* Docker containers for reproducibility

### Amazon SageMaker AI

The desktop is pre-configured for [Amazon SageMaker AI](https://aws.amazon.com/sagemaker-ai/).

**Clone SageMaker AI Examples GitHUb Repository:**
```bash
mkdir ~/sagemaker-ai
cd ~/sagemaker-ai
git clone -b distributed-training-pipeline https://github.com/aws/amazon-sagemaker-examples.git
```
Install Python extension in Visual Code, and open the cloned `amazon-sagemaker-examples` repository within Visual Code.

**Inference Examples:**
1. Navigate to: `amazon-sagemaker-examples/advanced_functionality/large-model-inference-testing/large_model_inference.ipynb`
2. Use conda `base` environment as kernel
3. Skip to  **Initialize Notebook**

**Training Examples (FSx for Lustre must be enabled on the Deep Learning desktop):**
1. Navigate to: `amazon-sagemaker-examples/advanced_functionality/distributed-training-pipeline/dist_training_pipeline.ipynb`
2. Use conda `base` environment as kernel
3. Skip to **Initialize Notebook**

### Data Storage and File Systems

**S3 Access:**
The desktop has access to your specified S3 bucket. Verify access:
```bash
aws s3 ls your-bucket-name
```
No output means the bucket is empty (normal). An error indicates access issues.

**Storage Options:**
* **[Amazon EBS](https://aws.amazon.com/ebs/):** Root volume (deleted when instance terminates)
* **[Amazon EFS](https://aws.amazon.com/efs/):** Mounted at `/home/ubuntu/efs` by default (persists after termination)
* **[Amazon FSx for Lustre](https://aws.amazon.com/fsx/):** Optional, mounted at `/home/ubuntu/fsx` by default (enable via `FSxForLustre` parameter)

**Important:** EBS volumes are deleted on termination. EFS file-systems persist. 

## Managing the Desktop

### Stopping and Restarting

You can safely reboot, stop, and restart the desktop instance anytime. EFS (and FSx for Lustre, if enabled) automatically remount on restart.

### Distributed Training

For distributed training workloads, launch a deep-learning cluster with EFA and Open MPI. See the [EFA Cluster Guide](EFA-CLUSTER.md).

### Deleting Resources

Delete CloudFormation stacks from the AWS console when no longer needed.

**What Gets Deleted:**
* EC2 instances
* EBS root volumes
* FSx for Lustre file-systems (if enabled)

**What Persists:**
* **EFS file-systems are NOT automatically deleted**

## Reference

### Desktop CloudFormation Template Parameters

| Parameter Name | Parameter Description |
| --- | ----------- |
| AWSUbuntuAMIType | **Required**. Selects the AMI type. Default is [AWS Deep Learning AMI (Ubuntu 18.04)](https://aws.amazon.com/marketplace/pp/Amazon-Web-Services-AWS-Deep-Learning-AMI-Ubuntu-1/B07Y43P7X5). |
| DesktopAccessCIDR | Public IP CIDR range for desktop access, e.g. 1.2.3.4/32 or 7.8.0.0/16. Ignored if DesktopSecurityGroupId is specified. |
| DesktopHasPublicIpAddress | **Required**. Specify if desktop has a public IP address. Set to "true" unless you have AWS [VPN](https://aws.amazon.com/vpn/) or [DirectConnect](https://aws.amazon.com/directconnect) enabled.
| DesktopInstanceType | **Required**. Amazon EC2 instance type. G3, G4, P3 and P4 instance types are GPU enabled. |
| DesktopSecurityGroupId | *Optional* advanced parameter. EC2 security group for desktop. Must allow ports 22 (SSH) and 8443 (DCV) from DesktopAccessCIDR, access to EFS and FSx for Lustre, and all traffic within the security group. Leave blank to auto-create. |
| DesktopVpcId | **Required**. Amazon VPC id. |
| DesktopVpcSubnetId | **Required**. Amazon VPC subnet. Must be public with Internet Gateway (for Internet access) or private with NAT gateway. |
| EBSOptimized | **Required**. Enable [network optimization for EBS](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-optimized.html) (default is **true**). |
| EFSFileSystemId | *Optional* advanced parameter. Existing EFS file-system id with [network mount target](https://docs.aws.amazon.com/efs/latest/ug/how-it-works.html#how-it-works-ec2) accessible from DesktopVpcSubnetId. Use with DesktopSecurityGroupId. Leave blank to create new. |
| EFSMountPath | Absolute path where EFS file-system is mounted (default is `/home/ubuntu/efs`). |
| EbsVolumeSize | **Required**. Size of EBS volume (default is 500 GB). |
| EbsVolumeType | **Required**. [EBS volume type](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-volume-types.html) (default is gp3). |
| FSxCapacity | *Optional*. Capacity of FSx for Lustre file-system in multiples of 1200 GB (default is 1200 GB). See FSxForLustre parameter. | 
| FSxForLustre | *Optional*. Enable FSx for Lustre file-system (disabled by default). When enabled, automatically imports data from `s3://S3bucket/S3Import`. See S3Bucket and S3Import parameters. |
| FSxMountPath | FSx file-system mount path (default is `/home/ubuntu/fsx`). |
| KeyName | **Required**. EC2 key pair name for SSH access. You must have the private key. |
| S3Bucket | **Required**. S3 bucket name for data storage. May be empty at stack creation. |
| S3Import | *Optional*. S3 import prefix for FSx file-system. See FSxForLustre parameter. |
| UbuntuAMIOverride | *Optional* advanced parameter to override the AMI. Leave blank to use default AMIs. See AWSUbuntuAMIType. |

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the MIT-0 [License](./LICENSE).


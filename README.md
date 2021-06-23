# AWS Deep Learning High Performance Graphics Desktop

The [AWS Deep Learning AMIs](https://aws.amazon.com/machine-learning/amis/) can be used to quickly launch Amazon EC2 instances pre-configured with Conda environments for popular deep learning frameworks such as TensorFlow, PyTorch, and Apache MXNet, among others. 
[Amazon NICE DCV](https://aws.amazon.com/hpc/dcv/) enables high performance graphics desktops in Amazon EC2. 

In this project, we show how to automatically combine the AWS Deep Learning AMI with NICE DCV to run a high performance graphics desktop for developing, training, testing and visualizing deep learning models in Amazon EC2. 

## Step by Step Tutorial

### <a name="Prerequisites"></a> Prerequisites
This tutorial assumes you have an [AWS Account](https://aws.amazon.com/account/), and you have [Administrator job function](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_job-functions.html) access to the AWS Management Console.

To get started:

* Select your [AWS Region](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html). The AWS Regions supported by this project include, us-east-1, us-east-2, us-west-2, eu-west-1, eu-central-1, ap-southeast-1, ap-southeast-2, ap-northeast-1, ap-northeast-2, and ap-south-1. Note that not all Amazon EC2 instance types are available in all [AWS Availability Zones](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html) in an AWS Region.
* Subscribe to the [AWS Deep Learning AMI (Ubuntu 18.04)](https://aws.amazon.com/marketplace/pp/Amazon-Web-Services-AWS-Deep-Learning-AMI-Ubuntu-1/B07Y43P7X5).
* If you do not already have an Amazon EC2 key pair, [create a new Amazon EC2 key pair](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html#prepare-key-pair). You will need the key pair name to specify the ```KeyName``` parameter when creating the CloudFormation stack below.
* You will need an [Amazon S3](https://aws.amazon.com/s3/) bucket. If you don't have one, [create a new Amazon S3 bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html) in the AWS region you selected. The S3 bucket can be empty at this point.
* Use [AWS check ip](http://checkip.amazonaws.com/) to get your public IP address. This will be the IP address you will need to specify ```DesktopAccessCIDR``` parameter in the stack. 
* Clone this Git repository on your laptop using [```git clone ```](https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository).

### Create AWS CloudFormation Stack
Use the AWS CloudFormation template [deep-learning-ubuntu-desktop.yaml](deep-learning-ubuntu-desktop.yaml) from your cloned  repository to create a new CloudFormation stack using the [ AWS Management console](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/cfn-console-create-stack.html), or using the [AWS CLI](https://docs.aws.amazon.com/cli/latest/reference/cloudformation/create-stack.html). See [Reference](#Reference) section for the [template](deep-learning-ubuntu-desktop.yaml) input parameters, and stack outputs.

The template [deep-learning-ubuntu-desktop.yaml](deep-learning-ubuntu-desktop.yaml) creates [AWS Identity and Access Management (IAM)](https://aws.amazon.com/iam/) resources. If you are creating CloudFormation Stack using the console, you must check 
**I acknowledge that AWS CloudFormation might create IAM resources.** If you use the ```aws cloudformation create-stack``` CLI, you must use ```--capabilities CAPABILITY_NAMED_IAM```. 



### Connect to Desktop using SSH

* Once the stack status in CloudFormation console is ```CREATE_COMPLETE```, find the deep learning desktop instance launched in your stack in the Amazon EC2 console, and [connect to the instance using SSH](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html) as user ```ubuntu```, using your SSH key pair.
* When you connect using SSH, and you see the message ```"Cloud init in progress. Machine will REBOOT after cloud init is complete!!"```, disconnect and try later after about 15 minutes. The desktop installs the NICE DCV server on first-time startup, and reboots after the install is complete.
* If you see the message ```NICE DCV server is enabled!```, run the command ```sudo passwd ubuntu``` to set a new password for user ```ubuntu```. Now you are ready to connect to the desktop using the [NICE DCV client](https://docs.aws.amazon.com/dcv/latest/userguide/client.html)

### Connect to Desktop using NICE DCV Client
* Download and install the [NICE DCV client](https://docs.aws.amazon.com/dcv/latest/userguide/client.html) on your laptop.
* Use the NICE DCV Client to login to the desktop as user ```ubuntu```
* When you first login to the desktop using the NICE DCV client, you will be asked if you would like to upgrade the OS version. **Do not upgrade the OS version** .

## Developing in an IDE
Use the Ubuntu ```Software``` desktop application to install [Visual Studio Code](https://code.visualstudio.com/), or your favorite IDE to build, debug, and train deep learning models. Open a desktop terminal, and run ```conda env list``` to view the available Conda environments. 

## Working with Data

The deep learning desktop instance has access to the S3 bucket you specified when you create the CloudFormation stack. You can verify the access to your S3 bucket by running the command ```aws s3 ls your-bucket-name```. If you do not have access to the S3 bucket, you will see an error message. If your S3 bucket is empty, the previous command will produce no output, which is normal. 

There is an [Amazon EBS](https://aws.amazon.com/ebs/) root volume attached to the instance. In addition, an [Amazon EFS](https://aws.amazon.com/efs/) file-system is mounted on your desktop at ```EFSMountPath```, which by default is ```/home/ubuntu/efs```.  

The Amazon EBS volume attached to the instance is deleted when the deep learning instance is terminated. However, the EFS file-system persists after you terminate the desktop instance. 

## Using Amazon SageMaker from deep learning desktop
The deep learning desktop is pre-configured to use [Amazon SageMaker](https://aws.amazon.com/sagemaker/) machine learning platform. To get started with [Amazon SageMaker examples](https://github.com/aws/amazon-sagemaker-examples) in a [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html) notebook, execute following steps in a desktop terminal:

	mkdir ~/git
	cd ~/git
	git clone https://github.com/aws/amazon-sagemaker-examples.git
	jupyter-lab
	
This will start a ```jupyter-lab``` notebook server in the terminal, and open a tab in your web browser. You can now explore the Amazon SageMaker examples. 

For SageMaker examples that require you to specify a [subnet](https://docs.aws.amazon.com/vpc/latest/userguide/VPC_Subnets.html#vpc-subnet-basics), and a [security group](https://docs.aws.amazon.com/vpc/latest/userguide/VPC_SecurityGroups.html), use the pre-configured environment variables  ```desktop_subnet_id``` and ```desktop_sg_id```, respectively.

## Stopping and Restarting the Desktop

You may safely reboot, stop, and restart the desktop instance at any time. The desktop will automatically mount the EFS file-system at restart.

## Deleting the Stack

When you no longer need the desktop instance, you may delete the AWS CloudFormation stack from the AWS CloudFormation console. Deleting the stack will terminate the desktop instance, and delete the root EBS volume attached to the desktop. The EFS file system is **not** automatically deleted when you delete the stack.

## <a name="Reference"></a> Reference

### AWS CloudFormation Template Input Parameters
Below, we describe the AWS CloudFormation template input parameters.

| Parameter Name | Parameter Description |
| --- | ----------- |
| DeepLearningAMIUbuntu | This is an *optional* advanced parameter whereby you specify a valid Deep Learning AMI for Ubuntu 18.04 available in your AWS region. This parameter overrides the AMI version specified in the template. |
| DesktopAccessCIDR | This parameter specifies the public IP CIDR range from where you need access to your deep learning desktop, e.g. 1.2.3.4/32, or 7.8.0.0/16. This parameter is *ignored* if you specify the optional parameter  DesktopSecurityGroupId.|
| DesktopHasPublicIpAddress | This is a **required** parameter whereby you specify if the desktop has a public internet address. Unless you have AWS [VPN](https://aws.amazon.com/vpn/) or [DirectConnect](https://aws.amazon.com/directconnect) access enabled, you must set this parameter to "true".
| DesktopInstanceType | This is a **required** parameter whereby you select an Amazon EC2 instance type. G3, G4, P3 and P4 instance types are GPU enabled.  |
| DesktopSecurityGroupId | This is an *optional* advanced parameter whereby you specify Amazon EC2 security group for your desktop. The specified security group must allow inbound access over ports 22 (SSH) and 8443 (DCV) from ```DesktopAccessCIDR```, access to EFS TCP port 2049, and required network access from within the security group for running distributed SageMaker training jobs. Leave it blank to automatically create a new security group, which enables access for SSH, DCV, EFS, and running **any** distributed SageMaker training job. |
| DesktopVpcId | This is a **required** parameter whereby you select your Amazon VPC id.|
| DesktopVpcSubnetId | This is a **required** parameter whereby you select your Amazon VPC subnet. The specified subnet must be public if you plan to access your desktop over the Internet. |
| EBSOptimized | This is a **required** parameter whereby you select if you want your desktop instance to be [network optimized for EBS](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-optimized.html) (default is **true**)|
| EFSFileSystemId | This is an *optional* advanced parameter whereby you specify an existing EFS file-system id with an [existing network mount target](https://docs.aws.amazon.com/efs/latest/ug/how-it-works.html#how-it-works-ec2)  accessible from your DesktopVpcSubnetId. If you specify this parameter, do it in conjunction with DesktopSecurityGroupId. Leave it blank to create a new EFS file-system.  |
 EFSMountPath | Absolute path for the directory where EFS file-system is mounted (default is ```/home/ubuntu/efs```).   |
| EbsVolumeSize | This is a **required** parameter whereby you specify the size of the EBS volume (default size is 200 GB). Typically, the default size is sufficient.|
| EbsVolumeType | This is a **required** parameter whereby you select the [EBS volume type](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-volume-types.html) (default is gp3). |
| KeyName | This is a **required** parameter whereby you select the Amazon EC2 key pair name used for SSH access to the desktop. You must have access to the selected key pair's private key to connect to your desktop. |
| S3Bucket | This is a **required** parameter whereby you specify the name of the Amazon S3 bucket used to store your data. The S3 bucket may be empty at the time you create the AWS CloudFormation stack.  |

### AWS CloudFormation Stack Outputs
Below, we describe the AWS CloudFormation Stack outputs.

| Output Key | Output Description |
| --- | ----------- |
| DesktopInstanceId | This is the Amazon EC2 instance id for the desktop. |
| DesktopRole | This is the AWS ARN for the IAM Role automatically created and attached to the desktop instance profile. |
| DesktopSecurityGroup | This is the security group attached to the desktop instance, which is either specified in the Stack input parameters, or is automatically created. |
| EFSFileSystemId | This is the EFS file system attached to the desktop instance, which is either specified in the Stack input parameters, or is automatically created. |


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.


# AWS Deep Learning High Performance Graphics Desktop

The [AWS Deep Learning AMIs](https://aws.amazon.com/machine-learning/amis/) can be used to quickly launch Amazon EC2 instances pre-configured with Conda environments for popular deep learning frameworks such as TensorFlow, PyTorch, and Apache MXNet, among others. 
[Amazon NICE DCV](https://aws.amazon.com/hpc/dcv/) enables high performance graphics desktops in Amazon EC2. 

In this project, we show how to automatically combine the AWS Deep Learning AMI with NICE DCV to run a high performance graphics desktop for developing, training, testing and visualizing deep learning models in Amazon EC2.

## AWS CloudFormation Template 
This repository provides an [AWS CloudFormation](https://aws.amazon.com/cloudformation/) template that can be used to create a stack for launching a deep learning graphics desktop based on [AWS Deep Learning AMI (Ubuntu 18.04) Version 41.0](https://aws.amazon.com/marketplace/pp/Amazon-Web-Services-AWS-Deep-Learning-AMI-Ubuntu-1/B07Y43P7X5) and [NICE DCV Server (Ubuntu 18.04) Version 2020.2-9662](https://docs.aws.amazon.com/dcv/latest/adminguide/setting-up-installing-linux-server.html).

### AWS CloudFormation Template Input Parameters
Below, we describe the AWS CloudFormation template input parameters.

| Parameter Name | Parameter Description |
| --- | ----------- |
| DeepLearningAMIUbuntu | This is an *optional* advanced parameter whereby you specify a valid Deep Learning AMI for Ubuntu 18.04 available in your AWS region. This parameter overrides the AMI version specified in the template. |
| DesktopAccessCIDR | This parameter specifies the public IP CIDR range from where you need access to your deep learning desktop, e.g. 1.2.3.4/32, or 7.8.0.0/16. This parameter is *ignored* if you specify the optional parameter  DesktopSecurityGroupId.|
| DesktopInstanceType | This is a **required** parameter whereby you select an Amazon EC2 instance type. G3, G4, P3 and P4 instance types are GPU enabled.  |
| DesktopSecurityGroupId | This is an *optional* advanced parameter whereby you specify Amazon EC2 security group for your desktop. The specified security group must allow inbound access over ports 22 (SSH) and 8443 (DCV). Leave it blank to create a new security group. |
| DesktopVpcId | This is a **required** parameter whereby you select your Amazon VPC id.|
| DesktopVpcSubnetId | This is a **required** parameter whereby you select your Amazon VPC subnet. The specified subnet must be public if you plan to access your desktop over the Internet. |
| EBSOptimized | This is a **required** parameter whereby you select if you want your desktop instance to be [network optimized for EBS](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-optimized.html) (default is **true**)|
| EFSFileSystemId | This is an *optional* advanced parameter whereby you specify an existing EFS file-system id with an [existing network mount target](https://docs.aws.amazon.com/efs/latest/ug/how-it-works.html#how-it-works-ec2)  accessible from your DesktopVpcSubnetId. If you specify this parameter, do it in conjunction with DesktopSecurityGroupId. Leave it blank to create a new EFS file-system.  |
| EbsVolumeSize | This is a **required** parameter whereby you specify the size of the EBS volume (default size is 200 GB). Typically, the default size is sufficient.|
| EbsVolumeType | This is a **required** parameter whereby you select the [EBS volume type](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-volume-types.html) (default is gp3). |
| KeyName | This is a **required** parameter whereby you select the Amazon EC2 key pair name used for SSH access to the desktop. You must have access to the selected key pair's private key to connect to your desktop. |
| S3Bucket | This is a **required** parameter whereby you specify the name of the Amazon S3 bucket with your data. |

### AWS CloudFormation Stack Outputs
Below, we describe the AWS CloudFormation Stack outputs.

| Output Key | Output Description |
| --- | ----------- |
| DesktopInstanceId | This is the Amazon EC2 instance id for the desktop. |
| DesktopRole | This is the AWS ARN for the IAM Role automatically created and attached to the desktop instance profile. |
| DesktopSecurityGroup | This is the security group attached to the desktop instance, which is either specified in the Stack input parameters, or is automatically created. |
| EFSFileSystemId | This is the EFS file system attached to the desktop instance, which is either specified in the Stack input parameters, or is automatically created. |

## Step by Step Tutorial

### Prerequisites
This tutorial assumes you have an [AWS Account](https://aws.amazon.com/account/), and you have [Administrator job function](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies_job-functions.html) access to the AWS Management Console.

To get started:

* Select your [AWS Region](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html). The AWS Regions supported by this project include, us-east-1, us-east-2, us-west-2, eu-west-1, eu-central-1, ap-southeast-1, ap-southeast-2, ap-northeast-1, ap-northeast-2, and ap-south-1.
* Subscribe to the [AWS Deep Learning AMI (Ubuntu 18.04)](https://aws.amazon.com/marketplace/pp/Amazon-Web-Services-AWS-Deep-Learning-AMI-Ubuntu-1/B07Y43P7X5).
* If you do not already have an Amazon EC2 key pair, [create a new Amazon EC2 key pair](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html#prepare-key-pair). You will need the key pair name to specify the ```KeyName``` parameter when creating the CloudFormation stack below.
* You will need an [Amazon S3](https://aws.amazon.com/s3/) bucket. If you don't have one, [create a new Amazon S3 bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html) in the AWS region of your choice. You will use the S3 bucket name to specify the ```S3Bucket``` parameter in the stack.
* Run ```curl ifconfig.co``` on your laptop and note your public IP address. This will be the IP address you will need to specify ```DesktopAccessCIDR``` parameter in the stack. 

### Create AWS CloudFormation Stack
To create the AWS CloudFormation stack:

*  Use the [deep-learning-ubuntu-desktop.yaml](deep-learning-ubuntu-desktop.yaml) AWS CloudFormation template, to [create a new CloudFormation stack](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/cfn-console-create-stack.html). 

### Connect to Desktop using SSH

* Once the stack status in CloudFormation console is ```CREATE_COMPLETE```, find the deep learning desktop instance launched in your stack in the Amazon EC2 console, and [connect to the instance using SSH](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html) as user ```ubuntu```, using your SSH key pair.
* On the EC2 instance, run the command ```sudo passwd ubuntu``` and set a new password for user ```ubuntu```
* The EC2 instance automatically installs and configures the NICE DCV server on startup, and it may take approximately 10-15 minutes to setup the DCV server.  To monitor the progress of the desktop initialization, you can ```tail -f /var/log/cloud-init-output.log```. The instance **reboots** automatically after the NICE DCV server setup is complete. The NICE DCV server logs are available under ```/var/log/dcv``` on the EC2 instance.

### Connect to Desktop using NICE DCV Client
* Now you are ready to connect to the EC2 instance using a [NICE DCV client](https://docs.aws.amazon.com/dcv/latest/userguide/client.html)
* When you first login to the desktop using NICE DCV client, you will be asked if you would like to upgrade the OS version. **Do not upgrade the OS version** .
* Once you are connected using NICE DCV client, you can run ```conda env list``` in a terminal window to view the available Conda environments. 
* We recommend using the Ubuntu Software desktop application to install **Visual Studio Code**  IDE.

### Stopping and Restarting the Desktop

* You may safely stop and restart the desktop instance from EC2 console at any time. 

## Working with Data

The deep learning desktop instance has access to the S3 bucket you specified when you create the CloudFormation stack. You can verify the access by running the command ```aws ls your-bucket-name```. 

There is an [Amazon EBS](https://aws.amazon.com/ebs/) root volume attached to the instance. In addition, an [Amazon EFS](https://aws.amazon.com/efs/) file-system is mounted at ```/efs```. 

You can use the EFS file system to stage your data. For example, use the command ```sudo mkdir /efs/data``` to create a new directory  ```data``` to stage data on the EFS file-system, and run the command ```sudo chown -R ubuntu:ubuntu /efs/data``` to change the ownership of the ```data``` directory to user ```ubuntu```. 

The Amazon EBS volume attached to the instance is deleted when the deep learning instance is terminated. However, the EFS file-system persists after you terminate the desktop instance. 

## Deleting the Stack

When you no longer need the desktop instance, you may delete the AWS CloudFormation stack from the AWS CloudFormation console. The EFS file system is **not** automatically deleted when you delete  stack.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.


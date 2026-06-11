# Muestra de auditoría del verificador NLI — exp12_matrix (Fase 3a, N5)

50 claims estratificados (seed=42); etiquetas regeneradas determinísticamente y verificadas contra los agregados persistidos. Completar `juicio_humano` (correcto / incorrecto / dudoso) y `comentario`.

## 1. [contradicted] hibrido | granite4.1-8b — q085 (score 0.986)
**Pregunta:** How to configure networking for Azure Virtual Network?

**Claim:** Attach Secondary Network Interface: From the pull-down menu, select myVMNic2..

**Mejor chunk** (azure/Azure Functions :: Tutorial: Establish Azure Functions private site access > Configure access restrictions):

>  to provision and deploy the function app. [AZURE > Azure Functions > Tutorial: Establish Azure Functions private site access > Configure access restrictions] The next step is to configure [access restrictions](../app-service/app-service-ip-restrictions.md) to ensure only resources on the virtual network can invoke the function.
> 
> [Private site](functions-networking-options.md#private-endpoints) access is enabled by creating an Azure Virtual Network [service endpoint](../virtual-network/virtual-network-service-endpoints-overview.md) between the function app and the specified virtual network. Ac…

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 2. [contradicted] hibrido | granite4.1-8b — q085 (score 0.986)
**Pregunta:** How to configure networking for Azure Virtual Network?

**Claim:** Delete Public IP Addresses and Virtual Machine: Confirm by typing myResourceGroup in the prompt and selecting Delete..

**Mejor chunk** (azure/Azure Functions :: Tutorial: Establish Azure Functions private site access > Configure access restrictions):

>  to provision and deploy the function app. [AZURE > Azure Functions > Tutorial: Establish Azure Functions private site access > Configure access restrictions] The next step is to configure [access restrictions](../app-service/app-service-ip-restrictions.md) to ensure only resources on the virtual network can invoke the function.
> 
> [Private site](functions-networking-options.md#private-endpoints) access is enabled by creating an Azure Virtual Network [service endpoint](../virtual-network/virtual-network-service-endpoints-overview.md) between the function app and the specified virtual network. Ac…

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 3. [contradicted] hibrido | granite4.1-8b — q085 (score 0.984)
**Pregunta:** How to configure networking for Azure Virtual Network?

**Claim:** Access Function App Networking Settings: Select your target function app from the list..

**Mejor chunk** (azure/Azure Functions :: Tutorial: Establish Azure Functions private site access > Configure access restrictions):

>  to provision and deploy the function app. [AZURE > Azure Functions > Tutorial: Establish Azure Functions private site access > Configure access restrictions] The next step is to configure [access restrictions](../app-service/app-service-ip-restrictions.md) to ensure only resources on the virtual network can invoke the function.
> 
> [Private site](functions-networking-options.md#private-endpoints) access is enabled by creating an Azure Virtual Network [service endpoint](../virtual-network/virtual-network-service-endpoints-overview.md) between the function app and the specified virtual network. Ac…

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 4. [contradicted] hibrido | granite4.1-8b — q179 (score 0.972)
**Pregunta:** Compare the pricing models of AWS, Azure, and GCP for compute resources

**Claim:** For a comprehensive comparison, additional information on AWS's pricing strategies for compute resources would be necessary, such as Reserved Instances, Spot Instances, and any specific discounts or commitment programs similar to Azure Savings Plans or GCP's Committed Use Discounts.

**Mejor chunk** (gcp/Compute Engine :: ResourcesStay organized with collectionsSave and categorize content based on your preferences.):

> [GCP > Compute Engine > ResourcesStay organized with collectionsSave and categorize content based on your preferences.] VM instances pricingThis page describes pricing for VM instances on Compute Engine,
>     including pricing for machine types and running a simulated maintenance
>     event.Spot VMs pricingThis page describes pricing for Spot VMs on Compute Engine.Networking pricingThis page describes pricing for networking resources on Google Cloud.Sole-tenant nodes pricingThis page describes pricing for sole-tenant nodes on Compute Engine.GPUs pricingThis page describes pricing for GPUs on Com…

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 5. [contradicted] denso | granite4.1-8b — q002 (score 0.990)
**Pregunta:** What are the pricing tiers for AWS EC2?

**Claim:** discusses the concept of crossing pricing tiers and applying discounts, but it does not enumerate the actual tier levels or their monetary thresholds.

**Mejor chunk** (aws/EC2 :: How you are billed<a name="concepts-reserved-instances-application"></a> > Reserved Instance discount pricing tiers<a name="reserved-instances-discounts"></a> > Calculate Reserved Instance pricing discounts<a name="pricing-discounts"></a> > [ New console ]):

> year Partial Upfront `t2.small` Reserved Instance, assume the upfront price is $60\.00 and the hourly rate is $0\.007\. This provides a list value of $121\.32\.
> 
> 
> [CODE_BLOCK_1]
> 
> 
> ------ [AWS > EC2 > How you are billed<a name="concepts-reserved-instances-application"></a> > Reserved Instance discount pricing tiers<a name="reserved-instances-discounts"></a> > Calculate Reserved Instance pricing discounts<a name="pricing-discounts"></a> > [ New console ]] **To view the fixed price values for Reserved Instances using the Amazon EC2 console**
> 
> 1. Open the Amazon EC2 console at [https://console\.aw…

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 6. [contradicted] hibrido | gemma4-e4b — q164 (score 0.998)
**Pregunta:** When should you choose Azure Blob Storage over Azure Entra ID?

**Claim:** The documentation describes how to use Entra ID *with* Blob Storage, but it does not provide a scenario or condition under which one should choose Blob Storage *over* Entra ID.

**Mejor chunk** (azure/Blob Storage :: Manage blob containers using the Azure portal > Manage container and blob access > Manage Azure RBAC role assignments for the container):

> Properly managing access to containers and their blobs is key to ensuring that your data remains safe. The following sections illustrate ways in which you can meet your access requirements. [AZURE > Blob Storage > Manage blob containers using the Azure portal > Manage container and blob access > Manage Azure RBAC role assignments for the container] Microsoft Entra ID offers optimum security for Blob Storage resources. Azure role-based access control (Azure RBAC) determines what permissions a security principal has to a given resource. To grant access to a container, you'll assign an RBAC role …

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 7. [contradicted] denso | gemma4-e4b — q036 (score 0.992)
**Pregunta:** What is Azure Virtual Machines and what are its main capabilities?

**Claim:** ### Core Functionality and Networking.

**Mejor chunk** (azure/Virtual Machines :: Virtual machines in Azure > Troubleshooting > How-To Guide):

> Azure portalAzure PowerShellTerraformAzure CLI
> 
> Linux in Azure learning pathCreate a Windows virtual machine in Azure [AZURE > Virtual Machines > Virtual machines in Azure > Troubleshooting > How-To Guide] Troubleshooting GuidesTroubleshoot deployment issuesTroubleshoot allocation failuresTroubleshoot unexpected rebootsMicrosoft Q&A
> 
> Azure Well-Architected Framework Virtual Machine considerationsAzure Well-Architected Framework Disk Storage considerationsCloud adoption frameworkAzure architecture center

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 8. [contradicted] denso | gemma4-e4b — q043 (score 0.991)
**Pregunta:** What regions support Azure AKS?

**Claim:** AKS is supported in Public regions and **Azure.

**Mejor chunk** (azure/AKS :: AKS frequently asked questions > Quotas, limits, and region availability > Can I spread an AKS cluster across regions?):

> S uses Helm to deliver components to your cluster. You can safely ignoreaks-managedprefixed Helm releases. Continuously increasing revisions on these Helm releases are expected and safe.
> 
> For a complete list of available regions, seeAKS regions and availability. [AZURE > AKS > AKS frequently asked questions > Quotas, limits, and region availability > Can I spread an AKS cluster across regions?] No. AKS clusters are regional resources and can't span regions. For guidance on how to create an architecture that includes multiple regions, seebest practices for business continuity and disaster recov…

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 9. [contradicted] denso | gemma4-e4b — q135 (score 0.798)
**Pregunta:** Compare AWS VPC and AWS IAM in terms of use cases and features

**Claim:** AWS IAM

| Feature/Aspect | AWS VPC (Virtual Private Cloud)

**Mejor chunk** (aws/EC2 :: Example policies for working with the AWS CLI or an AWS SDK<a name="ExamplePolicies_EC2"></a> > Example: Work with route tables<a name="iam-example-route-tables"></a>):

>  of the IAM role and not the instance profile \(if the names are different\)\. For more information, see [Instance profiles](iam-roles-for-amazon-ec2.md#ec2-instance-profile)\.
> 
> 
> [CODE_BLOCK_55] [AWS > EC2 > Example policies for working with the AWS CLI or an AWS SDK<a name="ExamplePolicies_EC2"></a> > Example: Work with route tables<a name="iam-example-route-tables"></a>] The following policy allows users to add, remove, and replace routes for route tables that are associated with VPC `vpc-ec43eb89` only\. To specify a VPC for the `ec2:Vpc` condition key, you must specify the full ARN of the …

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 10. [contradicted] denso | gemma4-e4b — q101 (score 0.991)
**Pregunta:** How to migrate an on-premises application to Google Cloud Cloud Run?

**Claim:** This container image must be stored in Artifact Registry to ensure seamless migration between GKE and Cloud Run [Source: gcp/GKE/GKE and Cloud RunStay.

**Mejor chunk** (gcp/GKE :: GKE and Cloud RunStay organized with collectionsSave and categorize content based on your preferences.):

>  can therefore easily be migrated between runtimes.
> Here's anexamplecomparing the YAML files of a Kubernetes deployment and a
> Cloud Run service.Both GKE and Cloud Run integrate seamlessly withCloud LoggingandCloud Monitoring, providing you with a
> central [GCP > GKE > GKE and Cloud RunStay organized with collectionsSave and categorize content based on your preferences.] ized view on the Google Cloud console to observe application metrics
> regardless of their platform. You can also useservice-level objectives
> (SLO) monitoringon both
> platforms, and view a unified display of the SLOs on the Cloud M…

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 11. [contradicted] denso | mistral-7b-instruct — q156 (score 0.999)
**Pregunta:** When should you choose Azure AKS over Azure Virtual Network?

**Claim:** You want to take advantage of the overlay network model, which conserves IP addresses for virtual networks and provides maximum cluster scale support ..

**Mejor chunk** (azure/AKS :: Use kubenet networking with your own IP address ranges in Azure Kubernetes Service (AKS) > Overview of kubenet networking with your own subnet > Choose a network model to use):

> 10.0.0.0/8address range that's advertised over the ExpressRoute connection. We recommend creating your AKS clusters in Azure virtual network subnets outside this address range, such as172.16.0.0/16. [AZURE > AKS > Use kubenet networking with your own IP address ranges in Azure Kubernetes Service (AKS) > Overview of kubenet networking with your own subnet > Choose a network model to use] The following considerations help outline when each network model may be the most appropriate:
> Usekubenetwhen:
> You have limited IP address space.Most of the pod communication is within the cluster.You don't nee…

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 12. [contradicted] lexico | mistral-7b-instruct — q189 (score 0.978)
**Pregunta:** How do the Kubernetes managed services handle cluster upgrades in EKS vs AKS vs GKE?

**Claim:** Azure Kubernetes Service (AKS).

**Mejor chunk** (azure/Blob Storage :: Static website hosting in Azure Storage):

> [AZURE > Blob Storage > Static website hosting in Azure Storage] Azure Blob Storage is ideal for storing large amounts of unstructured data such as text, images, and videos. Because blob storage also provides static website hosting support, it's a great option in cases where you don't require a web server to render content. Although you're limited to hosting static content such as HTML, CSS, JavaScript, and image files, you can use serverless architectures including [Azure Functions](../../azure-functions/functions-overview.md) and other Platform as a service (PaaS) services.
> 
> [!INCLUDE [stati…

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 13. [contradicted] denso | mistral-7b-instruct — q024 (score 0.878)
**Pregunta:** What is AWS DynamoDB and what are its main features?

**Claim:** Backup and restore: DynamoDB provides automatic backups and point-in-time recovery to protect against accidental data loss or corruption.

**Mejor chunk** (aws/DynamoDB :: NOTICE):

> [AWS > DynamoDB > NOTICE] This repository is archived, read-only, and no longer updated. For more information, read [the announcement on the AWS News Blog](https://aws.amazon.com/blogs/aws/retiring-the-aws-documentation-on-github/).
> 
> You can find up-to-date AWS technical documentation on the [AWS Documentation website](https://docs.aws.amazon.com/), where you can also submit feedback and suggestions for improvement.

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 14. [contradicted] denso | mistral-7b-instruct — q192 (score 0.995)
**Pregunta:** Compare identity federation approaches across AWS, Azure, and GCP

**Claim:** Azure Active Directory (AAD) provides identity federation capabilities that allow users to access Azure services and applications using identities from external identity providers such as SAML, OIDC, or social identity providers..

**Mejor chunk** (gcp/GKE :: Authenticate to Google Cloud APIs from GKE workloadsStay organized with collectionsSave and categorize content based on your preferences. > Disable Workload Identity Federation for GKE in your organization):

>  Identity Federation for GKE for the cluster, do the following:Click theDetailstab.In theSecuritysection, next toWorkload Identity, clickeditEdit.In theEdit Workload Identitydialog, clear theEnable Workload Identitycheckbox.ClickSave changes. [GCP > GKE > Authenticate to Google Cloud APIs from GKE workloadsStay organized with collectionsSave and categorize content based on your preferences. > Disable Workload Identity Federation for GKE in your organization] The steps in thelink Kubernetes ServiceAccounts to IAMsection let Kubernetes ServiceAccounts impersonate the identity of the linked
> IAM s…

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 15. [contradicted] hibrido | mistral-7b-instruct — q068 (score 0.990)
**Pregunta:** How to set up AWS VPC for a production workload?

**Claim:** On the Inbound rules tab, create the following rules: On the Outbound rules tab, keep the default rule, which allows all outbound traffic.

**Mejor chunk** (aws/EC2 :: Set up to use Amazon EC2<a name="get-set-up-for-amazon-ec2"></a> > Create a security group<a name="create-a-base-security-group"></a> > [ Old console ]):

>  for a short time in a test environment, but it is unsafe for production environments\.
> 
> 1. For **Outbound rules**, keep the default rule, which allows all outbound traffic\.
> 
> 1. Choose **Create security group**\.
> 
> ------ [AWS > EC2 > Set up to use Amazon EC2<a name="get-set-up-for-amazon-ec2"></a> > Create a security group<a name="create-a-base-security-group"></a> > [ Old console ]] **To create a security group with least privilege**
> 
> 1. Open the Amazon EC2 console at [https://console\.aws\.amazon\.com/ec2/](https://console.aws.amazon.com/ec2/)\.
> 
> 1. In the left navigation pane, choose **Sec…

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 16. [contradicted] lexico | qwen3.5-9b — q012 (score 0.907)
**Pregunta:** What is the SLA for AWS RDS?

**Claim:** Architecture and security group configurations.

**Mejor chunk** (aws/EC2 :: Task 3: Automatically connect your EC2 instance to your RDS database<a name="option1-task3-connect-ec2-instance-to-rds-database"></a> > Task objective<a name="option1-task3-connect-ec2-instance-to-rds-database-objective"></a>):

> [AWS > EC2 > Task 3: Automatically connect your EC2 instance to your RDS database<a name="option1-task3-connect-ec2-instance-to-rds-database"></a> > Task objective<a name="option1-task3-connect-ec2-instance-to-rds-database-objective"></a>] The objective of this task is to use the automatic connection feature in the EC2 console to automatically configure the connection between your EC2 instance and your RDS database\.
> 
> Use the following steps to connect your EC2 instance and RDS database using the automatic feature in the EC2 console\.
> 
> To view an animation of these steps, see [View an animatio…

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 17. [contradicted] lexico | qwen3.5-9b — q067 (score 1.000)
**Pregunta:** How to configure auto-scaling in AWS RDS?

**Claim:** The documentation does not contain any information about:.

**Mejor chunk** (aws/EC2 :: Tutorial: Connect an Amazon EC2 instance to an Amazon RDS database<a name="tutorial-connect-ec2-instance-to-rds-database"></a> > Context<a name="tutorial-connect-ec2-rds-context"></a>):

> -rds-option3.md)
> 
>   Configure the connection between your EC2 instance to your RDS database by manually configuring and assigning the security groups to reproduce the configuration that is automatically created by the automatic connection feature in Option 1 and Option 2\. [AWS > EC2 > Tutorial: Connect an Amazon EC2 instance to an Amazon RDS database<a name="tutorial-connect-ec2-instance-to-rds-database"></a> > Context<a name="tutorial-connect-ec2-rds-context"></a>] As context for why you'd want to configure a connection between your EC2 instance and an RDS database, let's consider the follow…

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 18. [contradicted] hibrido | qwen3.5-9b — q009 (score 0.989)
**Pregunta:** What is the SLA for AWS ECS?

**Claim:** Task scheduling strategies (REPLICA and DAEMON).

**Mejor chunk** (aws/ECS :: Service Connect<a name="service-connect"></a>):

> [AWS > ECS > Service Connect<a name="service-connect"></a>] Amazon ECS Service Connect provides management of service\-to\-service communication as Amazon ECS configuration\. It does this by building both service discovery and a service mesh in Amazon ECS\. This provides  the complete configuration inside each Amazon ECS service that you manage by service deployments, a unified way to refer to your services within namespaces that doesn't depend on the Amazon VPC DNS configuration, and standardized metrics and logs to monitor all of your applications on Amazon ECS\. Amazon ECS Service Connect o…

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 19. [contradicted] hibrido | qwen3.5-9b — q155 (score 0.997)
**Pregunta:** What are the trade-offs between Azure Container Instances and Azure Virtual Network?

**Claim:** IPv6 Support: Azure Container Instances don't support IPv6 communication for containers.

**Mejor chunk** (azure/Virtual Network :: Add, change, or delete a virtual network subnet):

> [AZURE > Virtual Network > Add, change, or delete a virtual network subnet] Azure virtual network subnets are essential containers where all Azure resources are deployed within a virtual network. This comprehensive guide explains how to add, change, or delete virtual network subnets using the Azure portal, Azure CLI, or Azure PowerShell, helping you effectively manage your network infrastructure.

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 20. [contradicted] lexico | qwen3.5-9b — q172 (score 0.989)
**Pregunta:** What are the differences between AWS VPC and Azure Virtual Network?

**Claim:** ## Available Information by Provider

### AWS VPC
The context provides the following information about AWS VPC:

| Feature | Details |
|---------|---------|
| **Definition** | Enables you to define a virtual network in your own logically isolated area within the AWS cloud  |
| **IP Addressing** | You can select IP address range, create subnets, configure route tables, network gateways, and security settings  |
| **Internet Access** | By default, your instance can access the Internet.

**Mejor chunk** (aws/EC2 :: EC2\-Classic<a name="ec2-classic-platform"></a> > Differences between instances in EC2\-Classic and a VPC<a name="differences-ec2-classic-vpc"></a>):

>  hardware\.  |  You can run your instance on shared hardware or single\-tenant hardware\.  | 
> | Accessing the Internet | Your instance can access the Internet\. Your instance automatically receives a public IP address, and can access the Internet directly [AWS > EC2 > EC2\-Classic<a name="ec2-classic-platform"></a> > Differences between instances in EC2\-Classic and a VPC<a name="differences-ec2-classic-vpc"></a>]  through the AWS network edge\. | By default, your instance can access the Internet\. Your instance receives a public IP address by default\. An Internet gateway is attached to your …

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 21. [unsupported] hibrido | granite4.1-8b — q171 (score 0.004)
**Pregunta:** Compare the serverless compute offerings between AWS Lambda and Azure Functions

**Claim:** Azure Functions provides specific enhancements like the Flex Consumption plan for reduced cold starts and virtual network support, which are not directly mentioned in the AWS Lambda context provided..

**Mejor chunk** (azure/Azure Functions :: Azure Functions Flex Consumption plan hosting > Benefits):

>  <em>serverless</em> model.
> 
> You can review end-to-end samples that feature the Flex Consumption plan in the [Flex Consumption plan samples repository](https://github.com/Azure-Samples/azure-functions-flex-consumption-samples). [AZURE > Azure Functions > Azure Functions Flex Consumption plan hosting > Benefits] The Flex Consumption plan builds on the strengths of the serverless Consumption plan, which include dynamic scaling and execution-based billing. With Flex Consumption, you also get these extra features:
> 
> + **Reduced Cold Start Times**: Enable [always-ready instances](#always-ready-insta…

**Etiqueta NLI:** `unsupported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 22. [unsupported] denso | granite4.1-8b — q028 (score 0.013)
**Pregunta:** What instance types are available in AWS S3?

**Claim:** The document focuses exclusively on Amazon EC2 (Elastic Compute Cloud) instance types, categorizing them into current and previous generation instances across various families such as General Purpose, Compute Optimized, Memory Optimized, Storage Optimized, Accelerated Computing, etc.

**Mejor chunk** (aws/EC2 :: Instance types<a name="instance-types"></a> > Available instance types<a name="AvailableInstanceTypes"></a> > Current generation instances<a name="current-gen-instances"></a> > Memory optimized<a name="current-mo"></a>):

> [AWS > EC2 > Instance types<a name="instance-types"></a> > Available instance types<a name="AvailableInstanceTypes"></a> > Current generation instances<a name="current-gen-instances"></a> > Memory optimized<a name="current-mo"></a>] | Type | Sizes | 
> | --- | --- | 
> | CR1 | cr1\.8xlarge | 
> | Hpc6id | hpc6id\.32xlarge | 
> | R4 | r4\.large \| r4\.xlarge \| r4\.2xlarge \| r4\.4xlarge \| r4\.8xlarge \| r4\.16xlarge | 
> | R5 | r5\.large \| r5\.xlarge \| r5\.2xlarge \| r5\.4xlarge \| r5\.8xlarge \| r5\.12xlarge \| r5\.16xlarge \| r5\.24xlarge \| r5\.metal | 
> | R5a | r5a\.large \| r5a\.xlarge \| r5a\.2x…

**Etiqueta NLI:** `unsupported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 23. [unsupported] lexico | granite4.1-8b — q019 (score 0.005)
**Pregunta:** What are the limits and quotas for AWS EC2?

**Claim:** Target capacity per fleet: up to 10,000 instances..

**Mejor chunk** (aws/EC2 :: Fleet quotas<a name="fleet-quotas"></a>):

> [AWS > EC2 > Fleet quotas<a name="fleet-quotas"></a>] The usual Amazon EC2 quotas \(formerly referred to as limits\) apply to instances launched by an EC2 Fleet or a Spot Fleet, such as [Spot Instance limits](using-spot-limits.md) and [volume limits](volume_limits.md)\.
> 
> In addition, the following quotas apply:
> 
> 
> ****  
> 
> | Quota description | Quota | 
> | --- | --- | 
> | The number of EC2 Fleets and Spot Fleets per Region in the active, deleted\_running, and cancelled\_running states | 1,000 ¹ ² ³ ⁴ | 
> | The number of Spot capacity pools \(unique combination of instance type and subnet\) | 300 ¹ …

**Etiqueta NLI:** `unsupported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 24. [unsupported] denso | granite4.1-8b — q132 (score 0.035)
**Pregunta:** What are the differences between AWS Lambda and AWS EC2?

**Claim:** It supports custom software installations and configurations..

**Mejor chunk** (aws/EC2 :: What is Amazon EC2?<a name="concepts"></a> > Features of Amazon EC2<a name="ec2-features"></a>):

>  scale up or down to handle changes in requirements or spikes in popularity, reducing your need to forecast traffic\.
> 
> For more information about cloud computing, see [What is cloud computing?](https://aws.amazon.com/what-is-cloud-computing/) [AWS > EC2 > What is Amazon EC2?<a name="concepts"></a> > Features of Amazon EC2<a name="ec2-features"></a>] Amazon EC2 provides the following features:
> + Virtual computing environments, known as *instances*
> + Preconfigured templates for your instances, known as *Amazon Machine Images \(AMIs\)*, that package the bits you need for your server \(including t…

**Etiqueta NLI:** `unsupported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 25. [unsupported] lexico | granite4.1-8b — q108 (score 0.699)
**Pregunta:** How to set up logging and monitoring for Google Cloud Cloud Build?

**Claim:** Navigate to Logging > Logs Explorer..

**Mejor chunk** (gcp/Cloud Functions :: Build process overview > View your build image logs):

> .
> Cloud Storage does provide a free tier: review the pricing
> document for further details.For Artifact Registry pricing, see thePricingpage.Because the build process is subject to billing, your project must
> have aCloud Billing Accountattached to it. [GCP > Cloud Functions > Build process overview > View your build image logs] A key benefit of having the build image process in your user project is access
> to build logs. You can use gcloud CLI or the Google Cloud console
> to reach the logs, which are available throughCloud Logging.
> gcloudDeploy your function using thegcloud functions deploycommand…

**Etiqueta NLI:** `unsupported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 26. [unsupported] lexico | gemma4-e4b — q002 (score 0.014)
**Pregunta:** What are the pricing tiers for AWS EC2?

**Claim:** Qualification Threshold: To qualify for a discount, the list value of your Reserved Instances in the Region must be $500,000 USD or more ..

**Mejor chunk** (aws/EC2 :: How you are billed<a name="concepts-reserved-instances-application"></a> > Reserved Instance discount pricing tiers<a name="reserved-instances-discounts"></a> > Calculate Reserved Instance pricing discounts<a name="pricing-discounts"></a>):

> -discounts)
> + [Buy with a discount tier](#buying-discount-tier)
> + [Crossing pricing tiers](#crossing-pricing-tiers)
> + [Consolidated billing for pricing tiers](#consolidating-billing) [AWS > EC2 > How you are billed<a name="concepts-reserved-instances-application"></a> > Reserved Instance discount pricing tiers<a name="reserved-instances-discounts"></a> > Calculate Reserved Instance pricing discounts<a name="pricing-discounts"></a>] You can determine the pricing tier for your account by calculating the list value for all of your Reserved Instances in a Region\. Multiply the hourly recurring pri…

**Etiqueta NLI:** `unsupported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 27. [unsupported] lexico | gemma4-e4b — q065 (score 0.086)
**Pregunta:** How to set up AWS RDS for a production workload?

**Claim:** Configuration Customization (Critical Warning)
The most important step is to customize the instance settings, as the default tutorial configurations are not suitable for production.

**Mejor chunk** (aws/EC2 :: Task 1: Create an RDS database – *optional*<a name="option1-task1-create-rds-database"></a> > Steps to create an RDS database<a name="option1-task1-create-rds-database-steps"></a>):

>  that you can use, you can skip this task\.
> 
> **Important**  
> If you use an existing RDS database, make sure that it is in the same VPC as your EC2 instance so that you can use the automatic connection feature\. [AWS > EC2 > Task 1: Create an RDS database – *optional*<a name="option1-task1-create-rds-database"></a> > Steps to create an RDS database<a name="option1-task1-create-rds-database-steps"></a>] Use the following steps to create an RDS database\.
> 
> To view an animation of these steps, see [View an animation: Create an RDS database](#task1-create-rds-database-animation)\.
> 
> **RDS database c…

**Etiqueta NLI:** `unsupported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 28. [unsupported] lexico | gemma4-e4b — q092 (score 0.299)
**Pregunta:** How to implement backup and disaster recovery for Azure Virtual Machines?

**Claim:** Azure Backup [Source: azure/.

**Mejor chunk** (azure/Virtual Machines :: Introduction to Azure managed disks > Backup and disaster recovery options):

>  use to ensure their performance automatically shifts to meet the changing demands of your workload, others require manual adjustment, and other disk types can't do either.
> To learn about the options each disk type has, seeOverview of options to improve Azure managed disk performance [AZURE > Virtual Machines > Introduction to Azure managed disks > Backup and disaster recovery options] Managed disks support several backup and disaster recovery options. These options include built-in redundancy options (locally redundant storage, and zone-redundant storage), Azure Backup, managed disk snapshots…

**Etiqueta NLI:** `unsupported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 29. [unsupported] lexico | gemma4-e4b — q044 (score 0.414)
**Pregunta:** What are the service limits for Azure AKS?

**Claim:** Service limits for Azure AKS cover several areas, including the maximum number of managed clusters (quota limits) and the rate of API calls (throttling limits).

**Mejor chunk** (azure/AKS :: Quotas, virtual machine size restrictions, and region availability in Azure Kubernetes Service (AKS) > Service quotas and limits > Quota limits on AKS Managed Clusters > AKS Managed Clusters Quota Limits):

> , follow the instructions here: https://learn.microsoft.com/azure/quotas/quickstart-increase-quota-portal.
> To remedy this, customers canrequest additional quota in the Azure portal Quotas pageor via theQuotas REST API. [AZURE > AKS > Quotas, virtual machine size restrictions, and region availability in Azure Kubernetes Service (AKS) > Service quotas and limits > Quota limits on AKS Managed Clusters > AKS Managed Clusters Quota Limits] Subscription TypeDefault number of AKS clusters per subscription per region for new subscriptions1Maximum number of AKS clusters per subscription per region via …

**Etiqueta NLI:** `unsupported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 30. [unsupported] denso | gemma4-e4b — q004 (score 0.614)
**Pregunta:** What is AWS EKS and what are its main features?

**Claim:** The main features highlighted in the documentation and contribution process are:

**Documentation Features (AsciiDoc-Powered):**.

**Mejor chunk** (aws/EKS :: Amazon EKS User Guide):

> [AWS > EKS > Amazon EKS User Guide] Welcome to the Amazon EKS User Guide repository. This repository contains the open source version of the [Amazon EKS User Guide](https://docs.aws.amazon.com/eks/latest/userguide/).
> 
> You can now edit the EKS User Guide source directly. The AsciiDoc markup language meets the needs of the AWS Platform, while also being easy to learn. 
> 
> Use the "Edit this page on GitHub" links in the right sidebar of the EKS User Guide to submit changes. 
> 
> - **AsciiDoc-Powered Documentation**: The docs now use AsciiDoc, an intuitive yet powerful authoring language similar to Mar…

**Etiqueta NLI:** `unsupported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 31. [unsupported] hibrido | mistral-7b-instruct — q065 (score 0.586)
**Pregunta:** How to set up AWS RDS for a production workload?

**Claim:** Configure additional settings such as storage, backup options, maintenance windows, etc., according to your specific needs..

**Mejor chunk** (aws/EC2 :: Task 1: Create an RDS database – *optional*<a name="option1-task1-create-rds-database"></a> > Steps to create an RDS database<a name="option1-task1-create-rds-database-steps"></a>):

>  that you can use, you can skip this task\.
> 
> **Important**  
> If you use an existing RDS database, make sure that it is in the same VPC as your EC2 instance so that you can use the automatic connection feature\. [AWS > EC2 > Task 1: Create an RDS database – *optional*<a name="option1-task1-create-rds-database"></a> > Steps to create an RDS database<a name="option1-task1-create-rds-database-steps"></a>] Use the following steps to create an RDS database\.
> 
> To view an animation of these steps, see [View an animation: Create an RDS database](#task1-create-rds-database-animation)\.
> 
> **RDS database c…

**Etiqueta NLI:** `unsupported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 32. [unsupported] denso | mistral-7b-instruct — q128 (score 0.559)
**Pregunta:** When should you use AWS VPC instead of AWS RDS?

**Claim:** On the other hand, Amazon RDS (Relational Database Service) is a managed database service that makes it easier to set up, operate, and scale a relational database in the cloud [AWS > Services > Amazon RDS].

**Mejor chunk** (aws/EC2 :: Task 1: Create an RDS database – *optional*<a name="option1-task1-create-rds-database"></a> > Steps to create an RDS database<a name="option1-task1-create-rds-database-steps"></a>):

>  that you can use, you can skip this task\.
> 
> **Important**  
> If you use an existing RDS database, make sure that it is in the same VPC as your EC2 instance so that you can use the automatic connection feature\. [AWS > EC2 > Task 1: Create an RDS database – *optional*<a name="option1-task1-create-rds-database"></a> > Steps to create an RDS database<a name="option1-task1-create-rds-database-steps"></a>] Use the following steps to create an RDS database\.
> 
> To view an animation of these steps, see [View an animation: Create an RDS database](#task1-create-rds-database-animation)\.
> 
> **RDS database c…

**Etiqueta NLI:** `unsupported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 33. [unsupported] lexico | mistral-7b-instruct — q135 (score 0.032)
**Pregunta:** Compare AWS VPC and AWS IAM in terms of use cases and features

**Claim:** Features: AWS VPC : It allows you to control traffic flow between instances in your own virtual network, as well as define a networking environment that's similar to on-premises environments.

**Mejor chunk** (aws/Lambda :: License Summary):

>  [sample-apps](./sample-apps) - Sample applications that demonstrate features and use cases for the AWS Lambda service and managed runtimes.
> - [templates](./templates) - AWS CloudFormation templates for creating functions and VPC network resources. [AWS > Lambda > License Summary] The sample code within this repo is made available under a modified MIT license. See the [LICENSE](./LICENSE) file.

**Etiqueta NLI:** `unsupported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 34. [unsupported] denso | mistral-7b-instruct — q079 (score 0.109)
**Pregunta:** How to deploy a containerized application on AWS EC2?

**Claim:** Verify that you can run Docker commands without [CODE]..

**Mejor chunk** (aws/ECS :: Getting started with Amazon ECS using the AWS CDK<a name="tutorial-ecs-web-server-cdk"></a> > Step 2: Use the AWS CDK to define a containerized web server on Fargate<a name="ecs-web-server-cdk-step-2"></a>):

> [CODE_BLOCK_11]
> 
> 
> ------ [AWS > ECS > Getting started with Amazon ECS using the AWS CDK<a name="tutorial-ecs-web-server-cdk"></a> > Step 2: Use the AWS CDK to define a containerized web server on Fargate<a name="ecs-web-server-cdk-step-2"></a>] Use the container image [https://hub.docker.com/r/amazon/amazon-ecs-sample](https://hub.docker.com/r/amazon/amazon-ecs-sample) from DockerHub\. This image contains a PHP web app that runs on Amazon Linux 2\.
> 
> In the AWS CDK project that you created, edit the file that contains the stack definition to resemble one of the following examples\.
> 
> **Note**  
> …

**Etiqueta NLI:** `unsupported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 35. [unsupported] hibrido | mistral-7b-instruct — q035 (score 0.502)
**Pregunta:** What are the service limits for Azure Functions?

**Claim:** For a comprehensive list of service limits for Azure Functions, refer to the [Functions service limits](functions-scale.md#service-limits) documentation.

**Mejor chunk** (azure/Virtual Network :: Azure Virtual Network frequently asked questions (FAQ) > Migration of classic network resources to Resource Manager > What is Azure Service Manager, and what does the term "classic" mean?):

> nets that you use for securing the resource. The following table shows some example limits:
> 
> 
> [TABLE_0]
> 
> >[!NOTE]
> > The limits are subject to change at the discretion of the Azure services. Refer to the respective service documentation for details. [AZURE > Virtual Network > Azure Virtual Network frequently asked questions (FAQ) > Migration of classic network resources to Resource Manager > What is Azure Service Manager, and what does the term "classic" mean?] Azure Service Manager is the old deployment model of Azure that was responsible for creating, managing, and deleting resources. The wor…

**Etiqueta NLI:** `unsupported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 36. [unsupported] lexico | qwen3.5-9b — q011 (score 0.274)
**Pregunta:** What is AWS VPC and what are its main features?

**Claim:** A default subnet for each Availability Zone.

**Mejor chunk** (aws/EC2 :: Dedicated Instances<a name="dedicated-instance"></a> > Supported features<a name="features"></a>):

>  and manually specify a tenancy of `dedicated` for the instances to run as Dedicated Instances\. For more information, see [Launch Dedicated Instances into a VPC](dedicated-usage-overview.md#dedicatedinstancesintovpc)\. [AWS > EC2 > Dedicated Instances<a name="dedicated-instance"></a> > Supported features<a name="features"></a>] Dedicated Instances support the following features and AWS service integrations:
> 
> **Topics**
> + [Reserved Instances](#dedicatedreservedinstances)
> + [Automatic scaling](#dedicated-instance-autoscaling)
> + [Automatic recovery](#dedicated-instance-recovery)
> + [Dedicated Spo…

**Etiqueta NLI:** `unsupported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 37. [unsupported] lexico | qwen3.5-9b — q026 (score 0.530)
**Pregunta:** What are the limits and quotas for AWS CloudWatch?

**Claim:** CloudWatch Service Quotas: The context states that calls to [CODE] count toward CloudWatch service quotas associated with your account .

**Mejor chunk** (aws/EC2 :: Spot Instance quotas<a name="using-spot-limits"></a> > Monitor Spot Instance quotas and usage<a name="monitoring-spot-limits"></a>):

>  Spot Instances \(16 x 16 vCPUs\)\. 
> 
> **Topics**
> + [Monitor Spot Instance quotas and usage](#monitoring-spot-limits)
> + [Request a quota increase](#spot-vcpu-limits-request-increase) [AWS > EC2 > Spot Instance quotas<a name="using-spot-limits"></a> > Monitor Spot Instance quotas and usage<a name="monitoring-spot-limits"></a>] You can view and manage your Spot Instance quotas using the following:
> + The Amazon EC2 [Services quotas page](https://console.aws.amazon.com/servicequotas/#!/services/ec2/quotas) in the Service Quotas console
> + The [get\-service\-quota](https://docs.aws.amazon.com/cli/lat…

**Etiqueta NLI:** `unsupported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 38. [unsupported] hibrido | qwen3.5-9b — q137 (score 0.687)
**Pregunta:** What are the trade-offs between AWS Lambda and AWS EC2 for production workloads?

**Claim:** Amazon ECS (Elastic Container Service, container orchestration, Fargate, ECS Anywhere).

**Mejor chunk** (aws/EC2 :: Optimize CPU options<a name="instance-optimize-cpu"></a>):

> [AWS > EC2 > Optimize CPU options<a name="instance-optimize-cpu"></a>] Amazon EC2 instances support multithreading, which enables multiple threads to run concurrently on a single CPU core\. Each thread is represented as a virtual CPU \(vCPU\) on the instance\. An instance has a default number of CPU cores, which varies according to instance type\. For example, an `m5.xlarge` instance type has two CPU cores and two threads per core by default—four vCPUs in total\.
> 
> **Note**  
> Each vCPU is a thread of a CPU core, except for T2 instances and instances powered by AWS Graviton2 processors\.
> 
> In mos…

**Etiqueta NLI:** `unsupported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 39. [unsupported] denso | qwen3.5-9b — q054 (score 0.618)
**Pregunta:** What are the quotas for Google Cloud Compute Engine?

**Claim:** If your project lacks preemptible quota, standard quota is consumed.

**Mejor chunk** (gcp/GKE :: Run GPUs in GKE Standard node poolsStay organized with collectionsSave and categorize content based on your preferences. > Ensure sufficient GPU quota > Requesting GPU quota):

>  quota at least equivalent to the number of GPUs per node multiplied by your cluster's maximum number of
> nodes.
> For example, if you create a cluster with three nodes that runs two GPUs
> per node, your project requires at least six GPU quota. [GCP > GKE > Run GPUs in GKE Standard node poolsStay organized with collectionsSave and categorize content based on your preferences. > Ensure sufficient GPU quota > Requesting GPU quota] To request GPU quota, use the Google Cloud console. For more information about
> requesting quotas, refer toGPU quotasin the
> Compute Engine documentation.
> To search for GPU …

**Etiqueta NLI:** `unsupported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 40. [unsupported] denso | qwen3.5-9b — q071 (score 0.246)
**Pregunta:** How to monitor AWS DynamoDB with CloudWatch?

**Claim:** To answer this question accurately, I would need documentation specifically covering AWS DynamoDB monitoring capabilities with CloudWatch.

**Mejor chunk** (aws/EC2 :: Monitor your instances using CloudWatch<a name="using-cloudwatch"></a>):

> [AWS > EC2 > Monitor your instances using CloudWatch<a name="using-cloudwatch"></a>] You can monitor your instances using Amazon CloudWatch, which collects and processes raw data from Amazon EC2 into readable, near real\-time metrics\. These statistics are recorded for a period of 15 months, so that you can access historical information and gain a better perspective on how your web application or service is performing\.
> 
> By default, Amazon EC2 sends metric data to CloudWatch in 5\-minute periods\. To send metric data for your instance to CloudWatch in 1\-minute periods, you can enable detailed…

**Etiqueta NLI:** `unsupported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 41. [supported] denso | granite4.1-8b — q031 (score 0.962)
**Pregunta:** What regions support Azure Entra ID?

**Claim:** The supported Azure clouds for Azure Entra ID are:.

**Mejor chunk** (azure/Entra ID :: Sign in to Windows virtual machine in Azure or Arc-enabled Windows Server, using Microsoft Entra ID and Azure Roles Based Access Control > Requirements > Network requirements):

> Azure GlobalAzure GovernmentMicrosoft Azure operated by 21Vianet
> NoteCIS hardened images support Microsoft Entra ID authentication for Microsoft Windows Enterprise and Microsoft Windows Server offerings. For more information, see:CIS Hardened Images on Microsoft Windows Enterprise. [AZURE > Entra ID > Sign in to Windows virtual machine in Azure or Arc-enabled Windows Server, using Microsoft Entra ID and Azure Roles Based Access Control > Requirements > Network requirements] To enable Microsoft Entra authentication to virtual machines in Azure or Arc-enabled Windows Servers, you need to ensure …

**Etiqueta NLI:** `supported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 42. [supported] lexico | granite4.1-8b — q108 (score 0.879)
**Pregunta:** How to set up logging and monitoring for Google Cloud Cloud Build?

**Claim:** Access the build logs via the URL provided in the terminal output or through the Google Cloud Console's Logs Explorer..

**Mejor chunk** (gcp/Cloud Functions :: Build process overview > View your build image logs):

> .
> Cloud Storage does provide a free tier: review the pricing
> document for further details.For Artifact Registry pricing, see thePricingpage.Because the build process is subject to billing, your project must
> have aCloud Billing Accountattached to it. [GCP > Cloud Functions > Build process overview > View your build image logs] A key benefit of having the build image process in your user project is access
> to build logs. You can use gcloud CLI or the Google Cloud console
> to reach the logs, which are available throughCloud Logging.
> gcloudDeploy your function using thegcloud functions deploycommand…

**Etiqueta NLI:** `supported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 43. [supported] lexico | gemma4-e4b — q132 (score 0.970)
**Pregunta:** What are the differences between AWS Lambda and AWS EC2?

**Claim:** State Changes: EC2 instances generate events when their state changes (e.g., an instance is launched or changes state within an EC2 Fleet or Spot Fleet)..

**Mejor chunk** (aws/EC2 :: Create Amazon EventBridge rules to monitor EC2 Fleet events<a name="ec2-fleet-using-eventbridge"></a> > Create an EventBridge rule to trigger a Lambda function<a name="eventbridge-trigger-lambda"></a>):

> docs.aws.amazon.com/eventbridge/latest/userguide/eb-rules.html) and [Amazon EventBridge event patterns](https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-event-patterns.html) in the *Amazon EventBridge User Guide* [AWS > EC2 > Create Amazon EventBridge rules to monitor EC2 Fleet events<a name="ec2-fleet-using-eventbridge"></a> > Create an EventBridge rule to trigger a Lambda function<a name="eventbridge-trigger-lambda"></a>] The following example creates an EventBridge rule to trigger a Lambda function every time that Amazon EC2 emits an EC2 Fleet instance change notification for whe…

**Etiqueta NLI:** `supported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 44. [supported] hibrido | gemma4-e4b — q041 (score 0.972)
**Pregunta:** What regions support Azure Virtual Network?

**Claim:** Virtual Network TAP**
The supported regions for Virtual Network TAP include:.

**Mejor chunk** (azure/Virtual Network :: Create a virtual network routing appliance > Register for the public preview (AFEC) and get approval > Register your subscription by using Azure Feature Exposure Control (AFEC)):

> 2).
> 
> The virtual network routing appliance public preview is limited to the following regions:
> 
> - West US  
> - East US  
> - East Asia  
> - North Europe  
> - West Europe  
> - East US 2  
> - West Central US  
> - UK South [AZURE > Virtual Network > Create a virtual network routing appliance > Register for the public preview (AFEC) and get approval > Register your subscription by using Azure Feature Exposure Control (AFEC)] AFEC controls preview access to the virtual network routing appliance. The AFEC feature name for enabling the virtual network routing appliance preview is:  
> - Microsoft.network/Allow…

**Etiqueta NLI:** `supported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 45. [supported] denso | mistral-7b-instruct — q166 (score 0.866)
**Pregunta:** Compare Azure App Service and Azure Blob Storage in terms of capabilities and pricing

**Claim:** Azure App Service Pricing: [Azure App Services Pricing](https://azure.microsoft.com/services/app-service/pricing/).

**Mejor chunk** (azure/Blob Storage :: Scalability and performance targets for premium page blob storage accounts):

> [AZURE > Blob Storage > Scalability and performance targets for premium page blob storage accounts] [!INCLUDE [storage-scalability-intro-include](../../../includes/storage-scalability-intro-include.md)]
> 
> The service-level agreement (SLA) for Azure Storage accounts is available at [SLA for Storage Accounts](https://azure.microsoft.com/support/legal/sla/storage/v1_5/).

**Etiqueta NLI:** `supported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 46. [supported] lexico | mistral-7b-instruct — q076 (score 0.932)
**Pregunta:** How to configure auto-scaling in AWS EC2?

**Claim:** The IAM policy associated with the IAM user or role using the scripts needs to have permissions to call the EC2 action [https://docs.aws.amazon.com/AWSEC2/latest/APIReference/ApiReference-query-DescribeTags.html](https://docs.aws.amazon.com/AWSEC2/latest/APIReference/ApiReference-query-DescribeTags.html).

**Mejor chunk** (aws/EC2 :: Configure instance metadata options for new instances<a name="configuring-IMDS-new-instances"></a> > Turn off access to instance metadata<a name="configure-IMDS-new-instances--turn-off-instance-metadata"></a> > [ AWS CLI ]):

> Configure Instance Details** page:
>   + Under **Advanced Details**, for **Metadata accessible**, select **Disabled**\.
> 
> For more information, see [Step 3: Configure Instance Details](launching-instance.md#configure_instance_details_step)\.
> 
> ------ [AWS > EC2 > Configure instance metadata options for new instances<a name="configuring-IMDS-new-instances"></a> > Turn off access to instance metadata<a name="configure-IMDS-new-instances--turn-off-instance-metadata"></a> > [ AWS CLI ]] **To turn off access to instance metadata**  
> Launch the instance with `--metadata-options` set to `HttpEndpoint=dis…

**Etiqueta NLI:** `supported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 47. [supported] lexico | qwen3.5-9b — q198 (score 0.874)
**Pregunta:** How does the concept of availability zones differ across AWS, Azure, and GCP?

**Claim:** **AWS Availability Zone Information (from context):**

| Aspect | AWS Details | Source |
|--------|-------------|--------|
| **AZ Structure** | Regions provide multiple physically separated and isolated Availability Zones, connected through low-latency, high-throughput, and highly redundant networking |  |
| **AZ Naming** | AZ names differ across accounts (e.g., [CODE] for your account might not have the same location as [CODE] for another account) |  |
| **AZ Identification** | Use Availability Zone ID (AZ ID) like [CODE] which is unique and consistent across all AWS accounts |  |
| **AZ Visibility** | View AZ IDs via AWS RAM console under **Your AZ ID** panel |  |
| **Local Zones** | AWS Local Zones are extensions of AWS Regions in geographic proximity to users, completely isolated from other AWS Zones |  |
| **Task Placement** | ECS task placement strategies can distribute tasks evenly across Availability Zones |  |

**Missing Information:**.

**Mejor chunk** (aws/EC2 :: Work with shared Dedicated Hosts<a name="dh-sharing"></a> > Share across Availability Zones<a name="dh-sharing-azs"></a>):

>  them\. Consumers can be individual AWS accounts, or organizational units or an entire organization from AWS Organizations\.
> 
> For more information about AWS RAM, see the *[AWS RAM User Guide](https://docs.aws.amazon.com/ram/latest/userguide/)*\. [AWS > EC2 > Work with shared Dedicated Hosts<a name="dh-sharing"></a> > Share across Availability Zones<a name="dh-sharing-azs"></a>] To ensure that resources are distributed across the Availability Zones for a Region, we independently map Availability Zones to names for each account\. This could lead to Availability Zone naming differences across acc…

**Etiqueta NLI:** `supported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 48. [supported] denso | qwen3.5-9b — q062 (score 0.985)
**Pregunta:** How to deploy a containerized application on AWS RDS?

**Claim:** AWS App2Container containerizes applications for Amazon ECS.

**Mejor chunk** (aws/ECS :: Amazon ECS developer tools overview<a name="ecs-developer-tools"></a> > Amazon ECS CLI<a name="developer-tools-ecscli"></a>):

>  see [Getting started with AWS App2Container](https://docs.aws.amazon.com/app2container/latest/UserGuide/start-intro.html)\.
> 
> Use App2Container if you have applications that are hosted on premises or on Amazon EC2 instances or both\. [AWS > ECS > Amazon ECS developer tools overview<a name="ecs-developer-tools"></a> > Amazon ECS CLI<a name="developer-tools-ecscli"></a>] The Amazon ECS CLI allows you to run your applications on Amazon ECS and AWS Fargate using the Docker Compose file format\. You can quickly provision resources, push and pull images using [Amazon ECR](https://docs.aws.amazon.com…

**Etiqueta NLI:** `supported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 49. [supported] lexico | qwen3.5-9b — q142 (score 0.978)
**Pregunta:** What are the differences between AWS RDS and AWS VPC?

**Claim:** VPC Migration: "You can move your EC2-Classic DB instance to a VPC in the same Region, in the same account.".

**Mejor chunk** (aws/EC2 :: Migrate from EC2\-Classic to a VPC<a name="vpc-migrate"></a> > Migrate your resources to a VPC<a name="full-migrate"></a> > Amazon RDS DB instances<a name="vpc-migrate-rds"></a>):

>  **Launch** to specify a key pair and launch your instance\.
> 
> ------
> 
> For more information about the parameters that you can configure in each step of the wizard, see [Launch an instance using the old launch instance wizard](launching-instance.md)\. [AWS > EC2 > Migrate from EC2\-Classic to a VPC<a name="vpc-migrate"></a> > Migrate your resources to a VPC<a name="full-migrate"></a> > Amazon RDS DB instances<a name="vpc-migrate-rds"></a>] You can move your EC2\-Classic DB instance to a VPC in the same Region, in the same account\. For more information, see [Updating the VPC for a DB Instance](h…

**Etiqueta NLI:** `supported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 50. [supported] denso | gemma4-e4b — q107 (score 0.975)
**Pregunta:** How to set up logging and monitoring for Google Cloud BigQuery?

**Claim:** Logs Covered: Scheduler and Controller Manager logs ..

**Mejor chunk** (gcp/GKE :: Compare features in Autopilot and Standard clustersStay organized with collectionsSave and categorize content based on your preferences. > Feature comparison of Autopilot and Standard clusters):

> Logging and monitoringPre-configured:System-level logsWorkload-level logsSystem-level monitoringGoogle Cloud Managed Service for PrometheusOptional:Control plane metricsControl plane logsDefault:System-level logsWorkload-level logsSystem-level monitoringGoogle Cloud Managed Service for PrometheusOptional:Control [GCP > GKE > Compare features in Autopilot and Standard clustersStay organized with collectionsSave and categorize content based on your preferences. > Feature comparison of Autopilot and Standard clusters]  plane metricsControl plane logsAdd-onsPre-configured:HTTP load balancingComput…

**Etiqueta NLI:** `supported`  |  **Juicio humano:** ______  |  **Comentario:** ______

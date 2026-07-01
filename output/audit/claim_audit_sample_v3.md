# Muestra de auditoría del verificador NLI — exp12_matrix (Fase 3a, N5)

50 claims estratificados (seed=42); etiquetas regeneradas determinísticamente y verificadas contra los agregados persistidos. Completar `juicio_humano` (correcto / incorrecto / dudoso) y `comentario`.

## 1. [contradicted] hibrido | granite4.1-8b — q085 (score 0.994)
**Pregunta:** How to configure networking for Azure Virtual Network?

**Claim:** Stop the Virtual Machine (if running): Choose myVM from the list..

**Mejor chunk** (azure/Virtual Network :: Quickstart: Create an Azure Virtual Network > Create virtual machines):

>     | Tier | Select **Developer**. |
>     | **Configure virtual networks** |  |
>     | Virtual network | Select **vnet-1**. |
> 
> 1. Select **Review + create**.
> 
> 1. Select **Create**. [AZURE > Virtual Network > Quickstart: Create an Azure Virtual Network > Create virtual machines] The following procedure creates two VMs named **vm-1** and **vm-2** in the virtual network:
> 
> 1. In the portal, search for and select **Virtual machines**.
> 
> 1. In **Virtual machines**, select **+ Create**, and then select **Azure virtual machine**.
> 
> 1. On the **Basics** tab of **Create a virtual machine**, enter or select …

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 2. [contradicted] hibrido | granite4.1-8b — q085 (score 0.995)
**Pregunta:** How to configure networking for Azure Virtual Network?

**Claim:** Delete Public IP Addresses and Virtual Machine: In the search box at the top of the portal, enter Resource group..

**Mejor chunk** (azure/Virtual Network :: Quickstart: Create an Azure Virtual Network > Create virtual machines):

>     | Tier | Select **Developer**. |
>     | **Configure virtual networks** |  |
>     | Virtual network | Select **vnet-1**. |
> 
> 1. Select **Review + create**.
> 
> 1. Select **Create**. [AZURE > Virtual Network > Quickstart: Create an Azure Virtual Network > Create virtual machines] The following procedure creates two VMs named **vm-1** and **vm-2** in the virtual network:
> 
> 1. In the portal, search for and select **Virtual machines**.
> 
> 1. In **Virtual machines**, select **+ Create**, and then select **Azure virtual machine**.
> 
> 1. On the **Basics** tab of **Create a virtual machine**, enter or select …

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 3. [contradicted] hibrido | granite4.1-8b — q085 (score 0.993)
**Pregunta:** How to configure networking for Azure Virtual Network?

**Claim:** Configure Access Restrictions: Under the Settings section, click on the Networking link..

**Mejor chunk** (azure/Virtual Network :: Tutorial: Configure both routing preference options for a virtual machine using the Azure portal > Clean up resources):

>  **Settings**.
> 
> 5. In **Networking** of **myVM**, select **Attach network interface**.
> 
> 6. In **Attach network interface**, select **myVMNic2** in the pull-down box.
> 
> 7. Select **OK**. [AZURE > Virtual Network > Tutorial: Configure both routing preference options for a virtual machine using the Azure portal > Clean up resources] If you're not going to continue to use this application, delete the public IP addresses and virtual machine with the following steps:
> 
> 1. In the search box at the top of the portal, enter **Resource group**.
> 
> 2. In the search results, select **Resource groups**.
> 
> 3. Se…

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 4. [contradicted] denso | granite4.1-8b — q002 (score 0.990)
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

## 5. [contradicted] denso | granite4.1-8b — q153 (score 0.992)
**Pregunta:** Compare AWS EC2 and AWS Lambda in terms of use cases and features

**Claim:** Pay-per-use model; you are charged based on the number of requests served and the compute time consumed (GB-seconds)..

**Mejor chunk** (aws/ECS :: Common use cases in Amazon ECS<a name="common_use_cases"></a> > Additional resources<a name="additional-resources"></a>):

> ancing Application Load Balancers\. This provides service discovery for a microservice architecture\.
> + [Amazon Elastic Container Service \- Reference Architecture: Service Discovery](https://github.com/awslabs/ecs-refarch-service-discovery/): This Amazon [AWS > ECS > Common use cases in Amazon ECS<a name="common_use_cases"></a> > Additional resources<a name="additional-resources"></a>]  ECS reference architecture provides service discovery to containers using CloudWatch Events, Lambda, and Route 53 private hosted zones\. 
> + [Metrics and traces collection from Amazon ECS using AWS Distro for O…

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 6. [contradicted] denso | gemma4-e4b — q101 (score 0.998)
**Pregunta:** How to migrate an on-premises application to Google Cloud Cloud Run?

**Claim:** Package your application into a standard container image..

**Mejor chunk** (gcp/Compute Engine :: Ubuntu Pro FAQStay organized with collectionsSave and categorize content based on your preferences. > Migration FAQs > I plan to migrate Ubuntu Pro instances from AWS or Azure to Compute Engine. What is the recommended toolset or method to do so?):

> start using a Linux VMguide. When creating the VM, on the public images tab, choose a Ubuntu Pro
> version.
> To migrate an instance running Ubuntu Pro on-premise or on other clouds
> to Google Cloud, seeChoose a migration path. [GCP > Compute Engine > Ubuntu Pro FAQStay organized with collectionsSave and categorize content based on your preferences. > Migration FAQs > I plan to migrate Ubuntu Pro instances from AWS or Azure to Compute Engine. What is the recommended toolset or method to do so?] You can add VM instances, custom images, or disks that are available
> in an existing environment, such as …

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 7. [contradicted] denso | gemma4-e4b — q055 (score 0.982)
**Pregunta:** What are the best practices for Google Cloud Cloud SQL?

**Claim:** For detailed information, you should refer to the "Cloud SQL best practices" documentation  .

**Mejor chunk** (gcp/Cloud Functions :: Cloud Functions client librariesStay organized with collectionsSave and categorize content based on your preferences. > Additional resources > PHP):

>  libraries best practicesIssue trackergoogle-cloud-functionson Stack OverflowSource code
> 
> The following list contains links to more resources related to the
> client library for Node.js:
> API referenceClient libraries best practicesIssue trackergoogle-cloud-functionson Stack OverflowSource code [GCP > Cloud Functions > Cloud Functions client librariesStay organized with collectionsSave and categorize content based on your preferences. > Additional resources > PHP] The following list contains links to more resources related to the
> client library for PHP:
> API referenceClient libraries best practice…

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 8. [contradicted] hibrido | gemma4-e4b — q085 (score 0.985)
**Pregunta:** How to configure networking for Azure Virtual Network?

**Claim:** Start VM Creation: In the portal, search for and select Virtual machines.

**Mejor chunk** (azure/Azure Functions :: Tutorial: Establish Azure Functions private site access > Configure access restrictions):

>  to provision and deploy the function app. [AZURE > Azure Functions > Tutorial: Establish Azure Functions private site access > Configure access restrictions] The next step is to configure [access restrictions](../app-service/app-service-ip-restrictions.md) to ensure only resources on the virtual network can invoke the function.
> 
> [Private site](functions-networking-options.md#private-endpoints) access is enabled by creating an Azure Virtual Network [service endpoint](../virtual-network/virtual-network-service-endpoints-overview.md) between the function app and the specified virtual network. Ac…

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 9. [contradicted] hibrido | gemma4-e4b — q138 (score 0.966)
**Pregunta:** Compare AWS EC2 and AWS EKS in terms of use cases and features

**Claim:** | The context only mentions EKS in the context of GuardDuty using the instance identity role for [EKS Runtime Monitoring](https://docs.aws.amazon.com/guardduty/latest/ug/guardduty-eks-runtime-monitoring.html) .

**Mejor chunk** (aws/EC2 :: Instance identity roles<a name="ec2-instance-identity-roles"></a> > Supported services<a name="iir-supported-services"></a>):

>  whether a service or feature that makes use of instance identity roles is enabled on the instance\.
> 
> Instance identity roles are automatically created when an instance is launched, have no role\-trust policy document, and are not subject to any identity or resource policy\. [AWS > EC2 > Instance identity roles<a name="ec2-instance-identity-roles"></a> > Supported services<a name="iir-supported-services"></a>] The following AWS services use the instance identity role:
> + **Amazon EC2** – [EC2 Instance Connect](Connect-using-EC2-Instance-Connect.md) uses the instance identity role to update an E…

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 10. [contradicted] hibrido | gemma4-e4b — q197 (score 0.950)
**Pregunta:** Compare the event-driven architectures: AWS EventBridge vs Azure Event Grid vs Google Eventarc

**Claim:** | Event Hubs: Reads per partition from a single point in the stream, allowing for high-throughput and replay-ability [Source

**Mejor chunk** (aws/ECS :: Amazon ECS events<a name="ecs_cwe_events"></a>):

> [AWS > ECS > Amazon ECS events<a name="ecs_cwe_events"></a>] Amazon ECS sends the following types of events to EventBridge: container instance state change events, task state change events, service action, and service deployment state change events\. If these resources change, an event is generated\. These events and their possible causes are described in greater detail in the following sections\.
> 
> **Note**  
> Amazon ECS may add other event types, sources, and details in the future\. If you are deserializing event JSON data in code, make sure that your application is prepared to handle unknown …

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 11. [contradicted] denso | mistral-7b-instruct — q156 (score 0.998)
**Pregunta:** When should you choose Azure AKS over Azure Virtual Network?

**Claim:** You don't need advanced AKS features, such as virtual nodes or Azure Network Policy ..

**Mejor chunk** (azure/AKS :: Use kubenet networking with your own IP address ranges in Azure Kubernetes Service (AKS) > Overview of kubenet networking with your own subnet > Choose a network model to use):

> 10.0.0.0/8address range that's advertised over the ExpressRoute connection. We recommend creating your AKS clusters in Azure virtual network subnets outside this address range, such as172.16.0.0/16. [AZURE > AKS > Use kubenet networking with your own IP address ranges in Azure Kubernetes Service (AKS) > Overview of kubenet networking with your own subnet > Choose a network model to use] The following considerations help outline when each network model may be the most appropriate:
> Usekubenetwhen:
> You have limited IP address space.Most of the pod communication is within the cluster.You don't nee…

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 12. [contradicted] lexico | mistral-7b-instruct — q189 (score 0.999)
**Pregunta:** How do the Kubernetes managed services handle cluster upgrades in EKS vs AKS vs GKE?

**Claim:** AWS Elastic Kubernetes Service (EKS).

**Mejor chunk** (azure/AKS :: Quickstart: Create an Azure Kubernetes Service (AKS) Automatic cluster > Create Automatic Kubernetes cluster):

> 000001   Ready    agent   2m26s   v1.28.5
> aks-nodepool1-13213685-vmss000002   Ready    agent   2m26s   v1.28.5 [AZURE > AKS > Quickstart: Create an Azure Kubernetes Service (AKS) Automatic cluster > Create Automatic Kubernetes cluster] To create an AKS Automatic cluster, search forKubernetes Services, and selectAutomatic Kubernetes clusterfrom the drop-down options.On theBasicstab, fill in all the mandatory fields (Subscription, Resource group, Kubernetes cluster name, and Region) required to get started:On theMonitoringtab, choose your monitoring configurations from Azure Monitor, Managed Pro…

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 13. [contradicted] denso | mistral-7b-instruct — q024 (score 0.812)
**Pregunta:** What is AWS DynamoDB and what are its main features?

**Claim:** High performance: DynamoDB provides single-digit millisecond latency at any scale.

**Mejor chunk** (aws/DynamoDB :: NOTICE):

> [AWS > DynamoDB > NOTICE] This repository is archived, read-only, and no longer updated. For more information, read [the announcement on the AWS News Blog](https://aws.amazon.com/blogs/aws/retiring-the-aws-documentation-on-github/).
> 
> You can find up-to-date AWS technical documentation on the [AWS Documentation website](https://docs.aws.amazon.com/), where you can also submit feedback and suggestions for improvement.

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 14. [contradicted] denso | mistral-7b-instruct — q192 (score 0.998)
**Pregunta:** Compare identity federation approaches across AWS, Azure, and GCP

**Claim:** In Autopilot clusters, it is always enabled..

**Mejor chunk** (gcp/GKE :: Authenticate to Google Cloud APIs from GKE workloadsStay organized with collectionsSave and categorize content based on your preferences. > Troubleshooting):

>  if you plan todisable service account creationordisable service account key creation,
> disabling Workload Identity Federation for GKE prevents the exchange of ServiceAccount tokens
> for IAM service account access tokens.
> For more information, seeDisable workload identity cluster creation. [GCP > GKE > Authenticate to Google Cloud APIs from GKE workloadsStay organized with collectionsSave and categorize content based on your preferences. > Troubleshooting] For troubleshooting information, refer toTroubleshooting Workload Identity Federation for GKE.
> 
> Learn more aboutWorkload Identity Federation …

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

## 16. [contradicted] hibrido | qwen3.5-9b — q009 (score 0.996)
**Pregunta:** What is the SLA for AWS ECS?

**Claim:** None of these sections contain information about the Service Level Agreement (SLA) for AWS ECS.

**Mejor chunk** (aws/ECS :: Custom log routing<a name="using_firelens"></a> > Considerations<a name="firelens-considerations"></a>):

> )\. We provide the AWS for Fluent Bit image or you can use your own Fluentd or Fluent Bit image\.
> 
> Creating Amazon ECS task definitions with a FireLens configuration is supported using the AWS SDKs, AWS CLI, and AWS Management Console\. [AWS > ECS > Custom log routing<a name="using_firelens"></a> > Considerations<a name="firelens-considerations"></a>] Consider the following when using FireLens for Amazon ECS:
> + FireLens for Amazon ECS is supported for tasks that are hosted on both AWS Fargate on Linux and Amazon EC2 on Linux\. Windows containers don't support FireLens\.
> 
>   For information abou…

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 17. [contradicted] hibrido | qwen3.5-9b — q155 (score 0.997)
**Pregunta:** What are the trade-offs between Azure Container Instances and Azure Virtual Network?

**Claim:** IPv6 Support: Azure Container Instances don't support IPv6 communication for containers.

**Mejor chunk** (azure/Virtual Network :: Add, change, or delete a virtual network subnet):

> [AZURE > Virtual Network > Add, change, or delete a virtual network subnet] Azure virtual network subnets are essential containers where all Azure resources are deployed within a virtual network. This comprehensive guide explains how to add, change, or delete virtual network subnets using the Azure portal, Azure CLI, or Azure PowerShell, helping you effectively manage your network infrastructure.

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 18. [contradicted] lexico | qwen3.5-9b — q172 (score 0.991)
**Pregunta:** What are the differences between AWS VPC and Azure Virtual Network?

**Claim:** Direct feature comparisons between AWS VPC and Azure Virtual Network.

**Mejor chunk** (azure/Virtual Network :: Custom IP address prefix (BYOIP) > Limitations):

> IMPORTANT]
> > There are several differences between how custom IPv4 and IPv6 prefixes are onboarded and utilized. For more information, see [Differences between using BYOIPv4 and BYOIPv6](create-custom-ip-address-prefix-ipv6 [AZURE > Virtual Network > Custom IP address prefix (BYOIP) > Limitations] -powershell.md#differences-between-using-byoipv4-and-byoipv6).

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 19. [contradicted] hibrido | qwen3.5-9b — q062 (score 0.999)
**Pregunta:** How to deploy a containerized application on AWS RDS?

**Claim:** None of the provided documentation contains information about AWS RDS (Relational Database Service) or how to deploy containerized applications with RDS.

**Mejor chunk** (aws/ECS :: Application architecture<a name="application_architecture"></a>):

> [AWS > ECS > Application architecture<a name="application_architecture"></a>] You can follow one of the two following models to run your containers:
> + Fargate launch type \- This is a serverless pay\-as\-you\-go option\. You can run containers without having to manage your infrastructure\.
> + EC2 launch type \- Configure and deploy EC2 instances in your cluster to run your containers\.
> 
> How you architect your application on Amazon ECS depends on several factors, with the launch type that you're using being a key differentiator\. We give the following guidance, broken down by launch type, which …

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 20. [contradicted] hibrido | qwen3.5-9b — q065 (score 0.968)
**Pregunta:** How to set up AWS RDS for a production workload?

**Claim:** The tutorial documentation is designed for learning purposes only and explicitly states that production environments require different configurations to meet specific needs.

**Mejor chunk** (aws/EC2 :: Task 2: Create an RDS database and automatically connect it to your EC2 instance<a name="option2-task2-create-rds-database"></a> > Steps to create an RDS database<a name="option2-task2-create-rds-database-steps"></a>):

> The objective of this task is to create an RDS database and use the automatic connection feature in the RDS console to automatically configure the connection between your EC2 instance and your RDS database\. [AWS > EC2 > Task 2: Create an RDS database and automatically connect it to your EC2 instance<a name="option2-task2-create-rds-database"></a> > Steps to create an RDS database<a name="option2-task2-create-rds-database-steps"></a>] Use the following steps to create an RDS database and connect it to your EC2 instance using the automatic feature in the RDS console\.
> 
> To view an animation of t…

**Etiqueta NLI:** `contradicted`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 21. [unsupported] hibrido | granite4.1-8b — q171 (score 0.312)
**Pregunta:** Compare the serverless compute offerings between AWS Lambda and Azure Functions

**Claim:** Flex Consumption plan enhancements: Per-function scaling for more efficient resource allocation..

**Mejor chunk** (azure/Azure Functions :: Use AI tools and models in Azure Functions):

> [AZURE > Azure Functions > Use AI tools and models in Azure Functions] Azure Functions provides serverless compute resources that integrate with AI and Azure services to streamline building cloud-hosted intelligent applications. This article provides a survey of the breadth of AI-related scenarios, integrations, and other AI resources that you can use in your function apps. 
> 
> Consider using Azure Functions in your AI-enabled experiences for these scenarios:
> 
> 
> [TABLE_0]
>  
> Select one of these scenarios to learn more in this article. 
> 
> This article is language-specific, so make sure you choose yo…

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

## 24. [unsupported] denso | granite4.1-8b — q132 (score 0.577)
**Pregunta:** What are the differences between AWS Lambda and AWS EC2?

**Claim:** AWS Lambda offers a fully managed, event-driven compute service with automatic scaling and pay-per-use pricing, ideal for flexible, scalable applications..

**Mejor chunk** (aws/EC2 :: What is Amazon EC2?<a name="concepts"></a> > Features of Amazon EC2<a name="ec2-features"></a>):

>  scale up or down to handle changes in requirements or spikes in popularity, reducing your need to forecast traffic\.
> 
> For more information about cloud computing, see [What is cloud computing?](https://aws.amazon.com/what-is-cloud-computing/) [AWS > EC2 > What is Amazon EC2?<a name="concepts"></a> > Features of Amazon EC2<a name="ec2-features"></a>] Amazon EC2 provides the following features:
> + Virtual computing environments, known as *instances*
> + Preconfigured templates for your instances, known as *Amazon Machine Images \(AMIs\)*, that package the bits you need for your server \(including t…

**Etiqueta NLI:** `unsupported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 25. [unsupported] lexico | granite4.1-8b — q108 (score 0.394)
**Pregunta:** How to set up logging and monitoring for Google Cloud Cloud Build?

**Claim:** Use the provided URL or search for [CODE] to view specific build logs..

**Mejor chunk** (gcp/GKE :: GKE overviewStay organized with collectionsSave and categorize content based on your preferences. > When to use GKE > Benefits of GKE):

> infrastructure that runs your containerized apps, such as networking, scaling,
> hardware, and security. GKE provides the operational power of
> Kubernetes while managing many of the underlying components, such as the
> control plane and nodes, for you. [GCP > GKE > GKE overviewStay organized with collectionsSave and categorize content based on your preferences. > When to use GKE > Benefits of GKE] The following table describes some of the benefits of using GKE
> as your managed Kubernetes platform:
> GKE benefitsPlatform managementFully-managed nodes in GKEAutopilot modewith built-in hardening and best…

**Etiqueta NLI:** `unsupported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 26. [unsupported] lexico | gemma4-e4b — q002 (score 0.497)
**Pregunta:** What are the pricing tiers for AWS EC2?

**Claim:** Calculating the Tier: You determine the pricing tier by.

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

## 29. [unsupported] lexico | gemma4-e4b — q044 (score 0.444)
**Pregunta:** What are the service limits for Azure AKS?

**Claim:** Managed Cluster Quota Limits (Maximum Cluster Count)

AKS quota represents a limit on the maximum number of managed clusters that an Azure subscription can create per region .

**Mejor chunk** (azure/AKS :: Quotas, virtual machine size restrictions, and region availability in Azure Kubernetes Service (AKS) > Service quotas and limits > Quota limits on AKS Managed Clusters > AKS Managed Clusters Quota Limits):

> , follow the instructions here: https://learn.microsoft.com/azure/quotas/quickstart-increase-quota-portal.
> To remedy this, customers canrequest additional quota in the Azure portal Quotas pageor via theQuotas REST API. [AZURE > AKS > Quotas, virtual machine size restrictions, and region availability in Azure Kubernetes Service (AKS) > Service quotas and limits > Quota limits on AKS Managed Clusters > AKS Managed Clusters Quota Limits] Subscription TypeDefault number of AKS clusters per subscription per region for new subscriptions1Maximum number of AKS clusters per subscription per region via …

**Etiqueta NLI:** `unsupported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 30. [unsupported] lexico | gemma4-e4b — q026 (score 0.221)
**Pregunta:** What are the limits and quotas for AWS CloudWatch?

**Claim:** General Information: For more details on configuring alarms and monitoring quotas, you can refer to the *Service Quotas User Guide* section on [Service Quotas and Amazon CloudWatch alarms](https://docs.aws.amazon.com/servicequotas/latest/userguide/configure-cloudwatch.html)..

**Mejor chunk** (aws/EC2 :: Spot Instance quotas<a name="using-spot-limits"></a> > Monitor Spot Instance quotas and usage<a name="monitoring-spot-limits"></a>):

>  Spot Instances \(16 x 16 vCPUs\)\. 
> 
> **Topics**
> + [Monitor Spot Instance quotas and usage](#monitoring-spot-limits)
> + [Request a quota increase](#spot-vcpu-limits-request-increase) [AWS > EC2 > Spot Instance quotas<a name="using-spot-limits"></a> > Monitor Spot Instance quotas and usage<a name="monitoring-spot-limits"></a>] You can view and manage your Spot Instance quotas using the following:
> + The Amazon EC2 [Services quotas page](https://console.aws.amazon.com/servicequotas/#!/services/ec2/quotas) in the Service Quotas console
> + The [get\-service\-quota](https://docs.aws.amazon.com/cli/lat…

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

## 33. [unsupported] lexico | mistral-7b-instruct — q135 (score 0.001)
**Pregunta:** Compare AWS VPC and AWS IAM in terms of use cases and features

**Claim:** This includes setting up a network topology, configuring route tables, network gateways, and security groups for your resources..

**Mejor chunk** (aws/Lambda :: AWS Lambda Developer Guide):

> [AWS > Lambda > AWS Lambda Developer Guide] This repository contains additional resources for the AWS Lambda developer guide.
> 
> - [iam-policies](./iam-policies) - Sample permissions policies for cross-service use cases.
> - [sample-apps](./sample-apps) - Sample applications that demonstrate features and use cases for the AWS Lambda service and managed runtimes.
> - [templates](./templates) - AWS CloudFormation templates for creating functions and VPC network resources.

**Etiqueta NLI:** `unsupported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 34. [unsupported] denso | mistral-7b-instruct — q079 (score 0.087)
**Pregunta:** How to deploy a containerized application on AWS EC2?

**Claim:** Install Docker Engine and the Docker CLI on your EC2 instance.

**Mejor chunk** (aws/ECS :: Amazon ECS developer tools overview<a name="ecs-developer-tools"></a> > Amazon ECS CLI<a name="developer-tools-ecscli"></a>):

>  see [Getting started with AWS App2Container](https://docs.aws.amazon.com/app2container/latest/UserGuide/start-intro.html)\.
> 
> Use App2Container if you have applications that are hosted on premises or on Amazon EC2 instances or both\. [AWS > ECS > Amazon ECS developer tools overview<a name="ecs-developer-tools"></a> > Amazon ECS CLI<a name="developer-tools-ecscli"></a>] The Amazon ECS CLI allows you to run your applications on Amazon ECS and AWS Fargate using the Docker Compose file format\. You can quickly provision resources, push and pull images using [Amazon ECR](https://docs.aws.amazon.com…

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

## 38. [unsupported] hibrido | qwen3.5-9b — q137 (score 0.768)
**Pregunta:** What are the trade-offs between AWS Lambda and AWS EC2 for production workloads?

**Claim:** AWS EC2 (burstable performance instances, CPU options, instance types).

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

## 40. [unsupported] denso | qwen3.5-9b — q071 (score 0.700)
**Pregunta:** How to monitor AWS DynamoDB with CloudWatch?

**Claim:** DynamoDB monitoring configurations or code examples.

**Mejor chunk** (aws/ECS :: Monitoring tools<a name="monitoring-automated-manual"></a> > Automated monitoring tools<a name="monitoring-automated_tools"></a>):

> AWS provides various tools that you can use to monitor Amazon ECS\. You can configure some of these tools to do the monitoring for you, while some of the tools require manual intervention\. We recommend that you automate monitoring tasks as much as possible\. [AWS > ECS > Monitoring tools<a name="monitoring-automated-manual"></a> > Automated monitoring tools<a name="monitoring-automated_tools"></a>] You can use the following automated monitoring tools to watch Amazon ECS and report when something is wrong:
> + Amazon CloudWatch alarms – Watch a single metric over a time period that you specify, …

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

## 43. [supported] lexico | gemma4-e4b — q132 (score 0.963)
**Pregunta:** What are the differences between AWS Lambda and AWS EC2?

**Claim:** This can be achieved using Instance Scheduler or by using AWS Lambda and an Amazon EventBridge rule  [Source: aws/EC2/Stop and start your instance<a name="Stop_Start"></a> > Stop and start your instances on a schedule<a name=".

**Mejor chunk** (aws/EC2 :: Create Amazon EventBridge rules to monitor EC2 Fleet events<a name="ec2-fleet-using-eventbridge"></a> > Create an EventBridge rule to trigger a Lambda function<a name="eventbridge-trigger-lambda"></a>):

> docs.aws.amazon.com/eventbridge/latest/userguide/eb-rules.html) and [Amazon EventBridge event patterns](https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-event-patterns.html) in the *Amazon EventBridge User Guide* [AWS > EC2 > Create Amazon EventBridge rules to monitor EC2 Fleet events<a name="ec2-fleet-using-eventbridge"></a> > Create an EventBridge rule to trigger a Lambda function<a name="eventbridge-trigger-lambda"></a>] The following example creates an EventBridge rule to trigger a Lambda function every time that Amazon EC2 emits an EC2 Fleet instance change notification for whe…

**Etiqueta NLI:** `supported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 44. [supported] denso | gemma4-e4b — q107 (score 0.975)
**Pregunta:** How to set up logging and monitoring for Google Cloud BigQuery?

**Claim:** Logs Covered: Scheduler and Controller Manager logs ..

**Mejor chunk** (gcp/GKE :: Compare features in Autopilot and Standard clustersStay organized with collectionsSave and categorize content based on your preferences. > Feature comparison of Autopilot and Standard clusters):

> Logging and monitoringPre-configured:System-level logsWorkload-level logsSystem-level monitoringGoogle Cloud Managed Service for PrometheusOptional:Control plane metricsControl plane logsDefault:System-level logsWorkload-level logsSystem-level monitoringGoogle Cloud Managed Service for PrometheusOptional:Control [GCP > GKE > Compare features in Autopilot and Standard clustersStay organized with collectionsSave and categorize content based on your preferences. > Feature comparison of Autopilot and Standard clusters]  plane metricsControl plane logsAdd-onsPre-configured:HTTP load balancingComput…

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

## 47. [supported] denso | qwen3.5-9b — q062 (score 0.989)
**Pregunta:** How to deploy a containerized application on AWS RDS?

**Claim:** EC2 instances can connect TO RDS databases: Option 1: Create RDS database, launch EC2 instance, then connect EC2 to RDS.

**Mejor chunk** (aws/ECS :: Amazon ECS developer tools overview<a name="ecs-developer-tools"></a> > Amazon ECS CLI<a name="developer-tools-ecscli"></a>):

>  see [Getting started with AWS App2Container](https://docs.aws.amazon.com/app2container/latest/UserGuide/start-intro.html)\.
> 
> Use App2Container if you have applications that are hosted on premises or on Amazon EC2 instances or both\. [AWS > ECS > Amazon ECS developer tools overview<a name="ecs-developer-tools"></a> > Amazon ECS CLI<a name="developer-tools-ecscli"></a>] The Amazon ECS CLI allows you to run your applications on Amazon ECS and AWS Fargate using the Docker Compose file format\. You can quickly provision resources, push and pull images using [Amazon ECR](https://docs.aws.amazon.com…

**Etiqueta NLI:** `supported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 48. [supported] lexico | qwen3.5-9b — q151 (score 0.929)
**Pregunta:** What are the trade-offs between AWS RDS and AWS EC2 for production workloads?

**Claim:** Configuration requirements (same VPC, same Region).

**Mejor chunk** (aws/EC2 :: Task 1: Create an RDS database – *optional*<a name="option1-task1-create-rds-database"></a> > Steps to create an RDS database<a name="option1-task1-create-rds-database-steps"></a>):

>  that you can use, you can skip this task\.
> 
> **Important**  
> If you use an existing RDS database, make sure that it is in the same VPC as your EC2 instance so that you can use the automatic connection feature\. [AWS > EC2 > Task 1: Create an RDS database – *optional*<a name="option1-task1-create-rds-database"></a> > Steps to create an RDS database<a name="option1-task1-create-rds-database-steps"></a>] Use the following steps to create an RDS database\.
> 
> To view an animation of these steps, see [View an animation: Create an RDS database](#task1-create-rds-database-animation)\.
> 
> **RDS database c…

**Etiqueta NLI:** `supported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 49. [supported] hibrido | gemma4-e4b — q002 (score 0.991)
**Pregunta:** What are the pricing tiers for AWS EC2?

**Claim:** If a purchase crosses into a discounted pricing tier, the service generates multiple Reserved Instance IDs, as there is an ID for each set of reservations in a tier .

**Mejor chunk** (aws/EC2 :: How you are billed<a name="concepts-reserved-instances-application"></a> > Reserved Instance discount pricing tiers<a name="reserved-instances-discounts"></a> > Crossing pricing tiers<a name="crossing-pricing-tiers"></a>):

>  **Two discount rates**—Your purchase within a Region crosses from a lower discount tier to a higher discount tier\. You are charged two different rates: one or more reservations at the lower discounted rate, and the remaining reservations at the higher discounted rate\. [AWS > EC2 > How you are billed<a name="concepts-reserved-instances-application"></a> > Reserved Instance discount pricing tiers<a name="reserved-instances-discounts"></a> > Crossing pricing tiers<a name="crossing-pricing-tiers"></a>] If your purchase crosses into a discounted pricing tier, you see multiple entries for that pu…

**Etiqueta NLI:** `supported`  |  **Juicio humano:** ______  |  **Comentario:** ______

## 50. [supported] lexico | mistral-7b-instruct — q177 (score 0.855)
**Pregunta:** What are the differences between AWS Lambda cold starts and Azure Functions cold starts?

**Claim:** Cold Start Causes: In Azure Functions (), cold starts are more frequent in the Consumption plan because function apps get shut down during periods of inactivity.

**Mejor chunk** (azure/Azure Functions :: [PowerShell](#tab/powershell) > Considerations for PowerShell functions > Cold Start > Avoid using Install-Module):

>  starts are a reality. *Cold start* refers to period of time it takes for your function app to start running to process a request. Cold start happens more frequently in the Consumption plan because your function app gets shut down during periods of inactivity. [AZURE > Azure Functions > [PowerShell](#tab/powershell) > Considerations for PowerShell functions > Cold Start > Avoid using Install-Module] Running `Install-Module` in your function script on each invocation can cause performance issues. Instead, use `Save-Module` or `Save-PSResource` before publishing your function app to bundle the n…

**Etiqueta NLI:** `supported`  |  **Juicio humano:** ______  |  **Comentario:** ______

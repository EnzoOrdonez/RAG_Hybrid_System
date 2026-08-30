# Gold v4 — auditoría humana del verificador (Tier 3, N≈200)

150 claims NUEVOS (seed 42), disjuntos de los 50 de v3. **Juicio ciego**: no se muestran las etiquetas de los verificadores para evitar anclaje. Completar `juicio_humano` con: `correcto` (el claim está respaldado por el chunk), `incorrecto` (contradicho o no respaldado), `dudoso` (evidencia insuficiente).

Estratos (ocultos al anotador): {'false_contr': 30, 'near_threshold': 40, 'disagreement': 50, 'random_anchor': 30}

## 1. denso | granite4.1-8b — q178
**Pregunta:** How does auto-scaling work differently in AWS EC2 vs Azure Virtual Machine Scale Sets?

**Claim:** Azure relies on pod scheduling failures and resource constraints to trigger scale-up actions..

**Mejor evidencia** (aws/ECS):

> **  
> Amazon ECS cluster auto scaling is only supported with Auto Scaling group capacity providers\. For Amazon ECS workloads that are hosted on AWS Fargate, see [AWS Fargate capacity providers](fargate-capacity-providers.md)\. [AWS > ECS > Amazon ECS cluster Auto Scaling<a name="cluster-auto-scaling"></a> > How cluster Auto Scaling works<a name="how-it-works"></a>] The following is the workflow used for Amazon ECS cluster auto scaling\. For more information, see [Turn on cluster Auto Scaling](turn-on-cluster-auto-scaling.md)\.
> 
> 1. Create an Auto Scaling group
> 
> 1. Create a capacity provider that uses that Auto Scaling group
> 
> 1. Turn on managed scaling for the capacity provider
> 
> 1. Associate the capacity provider with a cluster
> 
> 1. Create a capacity provider strategy that uses the capacity p

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 2. lexico | granite4.1-8b — q170
**Pregunta:** Compare Azure Cosmos DB and Azure Virtual Machines in terms of capabilities and pricing

**Claim:** Provides hardware acceleration of initial connection setup, enhancing performance and reducing latency..

**Mejor evidencia** (azure/Cosmos DB):

>  Cosmos DB for MongoDB accounts with Customer Managed Keys (CMK).
> Note2This capability cannot be enabled on an Azure Cosmos DB for MongoDB accounts with continuous backup.
> ImportantChanging capabilities using Azure Resource Manager is not available for Azure Cosmos DB for MongoDB accounts. [AZURE > Cosmos DB > Configure your Azure Cosmos DB for MongoDB account capabilities > Enable a capability] Retrieve your existing account capabilities by usingaz cosmosdb show:az cosmosdb show \
>     --resource-group <azure_resource_group> \
>     --name <azure_cosmos_db_account_name>You should see a capability section that's similar to this example output:"capabilities": [
>   {
>     "name": "EnableMongo"
>   }
> ]Review the default capability. In this example, the only capability that's set isEnableMongo.Set th

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 3. lexico | granite4.1-8b — q183
**Pregunta:** What are the differences between AWS DynamoDB and Azure Cosmos DB?

**Claim:** Getting Started with Azure Cosmos DB for Table ().

**Mejor evidencia** (azure/Cosmos DB):

> From Apache Cassandra to API for CassandraConnect to API for Cassandra from SparkChange feed support with query predicates
> 
> Connect to SparkAccess from Azure DatabricksAccess from HDInsight [AZURE > Cosmos DB > Azure Cosmos DB for Apache Cassandra documentation > Migrate existing Apache Cassandra data > Architecture] Differences between platforms
> 
> Migrate using Kafka ConnectMigrate using Azure Databricks

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 4. lexico | granite4.1-8b — q022
**Pregunta:** What is AWS Lambda and what are its main features?

**Claim:** Sample applications: The [CODE] directory provides example applications that demonstrate various features and use cases of AWS Lambda, including managed runtimes in multiple programming languages such as Node.js, Python, Ruby, Java, Go, C#, and PowerShell.

**Mejor evidencia** (aws/Lambda):

> [AWS > Lambda > AWS Lambda Developer Guide] This repository contains additional resources for the AWS Lambda developer guide.
> 
> - [iam-policies](./iam-policies) - Sample permissions policies for cross-service use cases.
> - [sample-apps](./sample-apps) - Sample applications that demonstrate features and use cases for the AWS Lambda service and managed runtimes.
> - [templates](./templates) - AWS CloudFormation templates for creating functions and VPC network resources.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 5. hibrido | granite4.1-8b — q189
**Pregunta:** How do the Kubernetes managed services handle cluster upgrades in EKS vs AKS vs GKE?

**Claim:** Control Over Upgrades: EKS requires upgrading the control plane first and allows node pool upgrades alongside it if needed..

**Mejor evidencia** (azure/AKS):

> S clusters via Azure Kubernetes Fleet ManagerUse planned maintenance to schedule and control upgradesStop AKS cluster upgrades automatically on API breaking changes (preview)Automatically upgrade AKS cluster node operating system imagesApply security updates to AKS nodes automatically by using GitHub actions [AZURE > AKS > Upgrade options and recommendations for Azure Kubernetes Service (AKS) clusters > Upgrade options > Special considerations for node pools that span multiple availability zones] AKS uses best-effort zone balancing in node groups. During an upgrade surge, the zones for surge nodes in virtual machine scale sets are unknown ahead of time, which can temporarily cause an unbalanced zone configuration. AKS deletes surge nodes after the upgrade and restores the original zone bal

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 6. denso | mistral-7b-instruct — q101
**Pregunta:** How to migrate an on-premises application to Google Cloud Cloud Run?

**Claim:** If you're using Ubuntu Pro, start by creating a VM and choosing the desired Ubuntu Pro version from the public images tab.

**Mejor evidencia** (gcp/GKE):

>  can therefore easily be migrated between runtimes.
> Here's anexamplecomparing the YAML files of a Kubernetes deployment and a
> Cloud Run service.Both GKE and Cloud Run integrate seamlessly withCloud LoggingandCloud Monitoring, providing you with a
> central [GCP > GKE > GKE and Cloud RunStay organized with collectionsSave and categorize content based on your preferences.] ized view on the Google Cloud console to observe application metrics
> regardless of their platform. You can also useservice-level objectives
> (SLO) monitoringon both
> platforms, and view a unified display of the SLOs on the Cloud Monitoring
> dashboard.You can implement continuous delivery to either GKE resources
> or Cloud Run services by usingCloud Deploy. Or, if
> you prefer, simultaneously deploy your application to both GKE
> and 

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 7. hibrido | qwen3.5-9b — q082
**Pregunta:** How to configure networking for Azure App Service?

**Claim:** Automated Deployment Configuration: Create the file share as part of your deployment.

**Mejor evidencia** (azure/Azure Functions):

>  to provision and deploy the function app. [AZURE > Azure Functions > Tutorial: Establish Azure Functions private site access > Configure access restrictions] The next step is to configure [access restrictions](../app-service/app-service-ip-restrictions.md) to ensure only resources on the virtual network can invoke the function.
> 
> [Private site](functions-networking-options.md#private-endpoints) access is enabled by creating an Azure Virtual Network [service endpoint](../virtual-network/virtual-network-service-endpoints-overview.md) between the function app and the specified virtual network. Access restrictions are implemented via service endpoints. Service endpoints ensure only traffic originating from within the specified virtual network can access the designated resource. In this case, t

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 8. lexico | granite4.1-8b — q196
**Pregunta:** What are the equivalent networking concepts across AWS, Azure, and GCP for VPCs, subnets, and security groups?

**Claim:** Subnets: All three providers support subnets within their respective VPC/VNet structures..

**Mejor evidencia** (aws/ECS):

>  service to the previous deployment and configuration\. If you use another service discovery system that is based on DNS instead of Service Connect, any frontend or client applications begin using new endpoints and changed endpoint configuration after the local DNS cache expires, commonly taking multiple hours\. [AWS > ECS > Service Connect<a name="service-connect"></a> > Service Connect concepts<a name="service-connect-concepts"></a> > Networking<a name="service-connect-concepts-network"></a>] In the default configuration, the Service Connect proxy listens on the `containerPort` from the port mapping in the task definition\. If you have rules in your security group to allow this port, then you don't need to change your Amazon VPC security groups to use Service Connect\.
> 
> Even if you set a

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 9. lexico | mistral-7b-instruct — q004
**Pregunta:** What is AWS EKS and what are its main features?

**Claim:** If a potential security issue is discovered, it should be reported via AWS's vulnerability reporting page and not on GitHub..

**Mejor evidencia** (aws/EKS):

> ferencing
>   - Enhanced security controls
>   - Enterprise-grade documentation features
> 
> - **Streamlined Contribution Process**:
>   - Direct editing of documentation source files
>   - Faster pull request processing
>   - Increased automation
>   - Reduced manual steps [AWS > EKS > Stay Tuned] For more information about the new experience, see [Contribute](https://docs.aws.amazon.com/eks/latest/userguide/contribute.html) in the Amazon EKS User Guide.
> 
> We look forward to your contributions with the new GitHub experience. The improved platform makes it easier than ever to help us enhance the Amazon EKS documentation.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 10. lexico | qwen3.5-9b — q175
**Pregunta:** What is the equivalent of AWS IAM in Azure and GCP?

**Claim:** - Reserved Instance usage examples.

**Mejor evidencia** (aws/Lambda):

> [AWS > Lambda > AWS Lambda Empty Function Project] This starter project consists of:
> * Function.cs - class file containing a class with a single function handler method
> * aws-lambda-tools-defaults.json - default argument settings for use with Visual Studio and command line deployment tools for AWS
> 
> You may also have a test project depending on the options selected.
> 
> The generated function handler is a simple method accepting a string argument that returns the uppercase equivalent of the input string. Replace the body of this method, and parameters, to suit your needs.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 11. hibrido | granite4.1-8b — q178
**Pregunta:** How does auto-scaling work differently in AWS EC2 vs Azure Virtual Machine Scale Sets?

**Claim:** Azure VMSS: Relies on AKS APIs or stopping the cluster for zero-node configurations..

**Mejor evidencia** (azure/Virtual Network):

>  all the VMs in the availability set, and [confirm that Accelerated Networking is enabled](./create-virtual-machine-accelerated-networking.md#confirm-that-accelerated-networking-is-enabled).
> 
>    
> [CODE_BLOCK_9] [AZURE > Virtual Network > Disable SR-IOV on a virtual network adapter > Enable Accelerated Networking on Virtual Machine Scale Sets] Azure Virtual Machine Scale Sets is slightly different, but follows the same workflow.
> 
> 1. First, stop the VMs:
> 
>    
> [CODE_BLOCK_10]
> 
> 
> 1. Once the VMs are stopped, update the Accelerated Networking property under the network interface.
> 
>    
> [CODE_BLOCK_11]
> 
> 
> 1. Virtual Machine Scale Sets has an upgrade policy that applies updates by using automatic, rolling, or manual settings. The following instructions set the policy to automatic so Virtual Machine 

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 12. denso | qwen3.5-9b — q006
**Pregunta:** What are the pricing tiers for AWS S3?

**Claim:** Burstable performance instances (T instances).

**Mejor evidencia** (aws/EC2):

> -discounts)
> + [Buy with a discount tier](#buying-discount-tier)
> + [Crossing pricing tiers](#crossing-pricing-tiers)
> + [Consolidated billing for pricing tiers](#consolidating-billing) [AWS > EC2 > How you are billed<a name="concepts-reserved-instances-application"></a> > Reserved Instance discount pricing tiers<a name="reserved-instances-discounts"></a> > Calculate Reserved Instance pricing discounts<a name="pricing-discounts"></a>] You can determine the pricing tier for your account by calculating the list value for all of your Reserved Instances in a Region\. Multiply the hourly recurring price for each reservation by the total number of hours for the term and add the undiscounted upfront price \(also known as the fixed price\) at the time of purchase\. Because the list value is based on 

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 13. denso | qwen3.5-9b — q123
**Pregunta:** How to migrate an on-premises application to Google Cloud GKE?

**Claim:** GKE Multi-Cloud - Extends GKE for use outside Google Cloud, including: GKE attached clusters (add CNCF-conformant Kubernetes clusters to your fleet).

**Mejor evidencia** (gcp/GKE):

> [GCP > GKE > Deploy and update from an IDEStay organized with collectionsSave and categorize content based on your preferences.] AutopilotIf you develop applications to run in Google Kubernetes Engine, you can use an IDE and
> the Cloud Code plugin to simplify key development tasks, such as
> testing and debugging locally and deploying to a test cluster.ObjectivesLearn how to do the following development tasks in Cloud Shell Editor,
> an online demonstration IDE:Create a sample Kubernetes app.Run, update, and view logs for the app in your development environment.Deploy the app to a GKE cluster.You can follow a similar process in your own IDE if youinstall Cloud Code.To follow step-by-step guidance for this task directly in the
>   Google Cloud console, clickGuide me:Guide meCostsIn this document, 

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 14. lexico | granite4.1-8b — q116
**Pregunta:** How to set up logging and monitoring for Google Cloud Cloud Functions?

**Claim:** Enable Data Access Audit Logs.

**Mejor evidencia** (gcp/Cloud Functions):

> The following audit logs are associated with methods belonging togoogle.cloud.functions.v2.FunctionService. [GCP > Cloud Functions > Functions audit logging > API interface audit logs > google.cloud.functions.v2.FunctionService > AbortFunctionUpgrade] Method:google.cloud.functions.v2.FunctionService.AbortFunctionUpgradeAudit log type:Admin activityPermissions:cloudfunctions.functions.generationUpgrade - ADMIN_WRITEMethod is a long-running or streaming operation:Long-running operationFilter for this method:protoPayload.methodName="google.cloud.functions.v2.FunctionService.AbortFunctionUpgrade"

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 15. denso | qwen3.5-9b — q166
**Pregunta:** Compare Azure App Service and Azure Blob Storage in terms of capabilities and pricing

**Claim:** Use Case Selection: Blob Storage: Choose when you don't require a web server to render content and need to store unstructured data.

**Mejor evidencia** (azure/Blob Storage):

> [AZURE > Blob Storage > Scalability and performance targets for premium page blob storage accounts] [!INCLUDE [storage-scalability-intro-include](../../../includes/storage-scalability-intro-include.md)]
> 
> The service-level agreement (SLA) for Azure Storage accounts is available at [SLA for Storage Accounts](https://azure.microsoft.com/support/legal/sla/storage/v1_5/).

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 16. hibrido | granite4.1-8b — q190
**Pregunta:** Compare DNS services: AWS Route 53 vs Azure DNS vs Google Cloud DNS

**Claim:** Important note: Increase forwarding timeout values for Windows DNS servers forwarding to Azure DNS to avoid resolution issues..

**Mejor evidencia** (aws/EC2):

>  enable Route 53 Resolver DNS Firewall \(see [Route 53 Resolver DNS Firewall](https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/resolver-dns-firewall.html) in the *Amazon Route 53 Developer Guide*\)\. [AWS > EC2 > Security group connection tracking<a name="security-group-connection-tracking"></a> > Untracked connections<a name="untracked-connections"></a>] Not all flows of traffic are tracked\. If a security group rule permits TCP or UDP flows for all traffic \(0\.0\.0\.0/0 or ::/0\) and there is a corresponding rule in the other direction that permits all response traffic \(0\.0\.0\.0/0 or ::/0\) for all ports \(0\-65535\), then that flow of traffic is not tracked, unless it is part of an [automatically tracked connection](#automatic-tracking)\. The response traffic for an untrack

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 17. hibrido | qwen3.5-9b — q082
**Pregunta:** How to configure networking for Azure App Service?

**Claim:** Automated Deployment Configuration: Set the [CODE] app setting to the name of the file share ].

**Mejor evidencia** (azure/Azure Functions):

>  to provision and deploy the function app. [AZURE > Azure Functions > Tutorial: Establish Azure Functions private site access > Configure access restrictions] The next step is to configure [access restrictions](../app-service/app-service-ip-restrictions.md) to ensure only resources on the virtual network can invoke the function.
> 
> [Private site](functions-networking-options.md#private-endpoints) access is enabled by creating an Azure Virtual Network [service endpoint](../virtual-network/virtual-network-service-endpoints-overview.md) between the function app and the specified virtual network. Access restrictions are implemented via service endpoints. Service endpoints ensure only traffic originating from within the specified virtual network can access the designated resource. In this case, t

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 18. lexico | qwen3.5-9b — q119
**Pregunta:** How to deploy a workload on Google Cloud Cloud Run?

**Claim:** Using Terraform for GKE setup.

**Mejor evidencia** (gcp/GKE):

>  in GKEtells you how to ensure that your application's Pods run on the same or
> different underlying machines.GKE Sandboxexplains
> how to protect your host kernel by using sandbox Pods when you deploy unknown
> or untrusted workloads. [GCP > GKE > Deploying workloadsStay organized with collectionsSave and categorize content based on your preferences. > Observing your workloads] GKE provides a range of features for observing your workloads and
> their health, including at-a-glance overviews of workload state and metrics in
> the Google Cloud console, as well as more in-depth metrics, logs, and
> alerting.
> Learn more about using the GKE pages in the Google Cloud console
> inGKE in the
> Google Cloud console.Learn more about usingApp Hubto view
> your workloads and Services.Learn more about GKE and Google Cl

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 19. lexico | qwen3.5-9b — q021
**Pregunta:** What is AWS CloudWatch and what are its main features?

**Claim:** Can monitor Data Lifecycle Manager (DLM) policies.

**Mejor evidencia** (aws/EC2):

> [AWS > EC2 > Monitoring Capacity Reservations<a name="cr-monitor"></a>] You can use the following features to monitor your Capacity Reservations:
> 
> **Topics**
> + [CloudWatch metrics](capacity-reservation-cw-metrics.md)
> + [EventBridge events](cr-eventbridge.md)
> + [Utilization notifications](monitor-cr-utilization.md)

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 20. denso | mistral-7b-instruct — q108
**Pregunta:** How to set up logging and monitoring for Google Cloud Cloud Build?

**Claim:** Access the build logs using [CODE] or the Google Cloud console.

**Mejor evidencia** (gcp/Cloud Functions):

> 380d50d4f5e8%0AlogName%3Dprojects%2F%
> 2Flogs%2Fcloudbuild
> Deployingfunction(maytakeawhile-upto2minutes)...done. [GCP > Cloud Functions > Build process overview > Google Cloud console] To view function logs in the Cloud Run page:
> Go to Cloud RunClick the chosen function in the displayed list.Click theLOGStab to get the request and container logs for all
> revisions of this function. You can filter by log severity level.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 21. lexico | gemma4-e4b — q170
**Pregunta:** Compare Azure Cosmos DB and Azure Virtual Machines in terms of capabilities and pricing

**Claim:** | A compute service used to host middle-tier services and provide enhanced networking capabilities.

**Mejor evidencia** (azure/Cosmos DB):

>  Cosmos DB for MongoDB accounts with Customer Managed Keys (CMK).
> Note2This capability cannot be enabled on an Azure Cosmos DB for MongoDB accounts with continuous backup.
> ImportantChanging capabilities using Azure Resource Manager is not available for Azure Cosmos DB for MongoDB accounts. [AZURE > Cosmos DB > Configure your Azure Cosmos DB for MongoDB account capabilities > Enable a capability] Retrieve your existing account capabilities by usingaz cosmosdb show:az cosmosdb show \
>     --resource-group <azure_resource_group> \
>     --name <azure_cosmos_db_account_name>You should see a capability section that's similar to this example output:"capabilities": [
>   {
>     "name": "EnableMongo"
>   }
> ]Review the default capability. In this example, the only capability that's set isEnableMongo.Set th

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 22. denso | granite4.1-8b — q089
**Pregunta:** How to configure networking for Azure Blob Storage?

**Claim:** Click Create virtual network..

**Mejor evidencia** (azure/Blob Storage):

>  subresource select `dfs`. Select Virtual network and select virtual network and Subnet. Select DNS. Select Yes for Integrate with private DNS. Select the Subscription and Resource Group for your private link DNS. Select Next, Next, and select Create. [AZURE > Blob Storage > Troubleshoot issues in BlobFuse > Common mount problems > Failed to initialize new pipeline [configuration error in azstorage [account name not provided]]] Make sure the configuration file has `azstorage` section in your configuration file.
> 
> The [BlobFuse base configuration file](https://github.com/Azure/azure-storage-fuse/blob/main/setup/baseConfig.yaml) contains a list of all settings and a brief explanation of each setting. Use the [sample file cache configuration file](https://github.com/Azure/azure-storage-fuse/bl

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 23. hibrido | qwen3.5-9b — q094
**Pregunta:** How to configure networking for Azure Container Instances?

**Claim:** Azure Bastion for connecting to VMs.

**Mejor evidencia** (azure/Virtual Network):

>  install and configure the CNI plugin for a standalone Linux Docker host.
> 
> - An Azure account with an active subscription. [Create an account for free](https://azure.microsoft.com/pricing/purchase-options/azure-account?cid=msft_learn). [AZURE > Virtual Network > Deploy container networking for a stand-alone Linux Docker host > Create a resource group] 1. In the portal, search for and select **Resource groups**.
> 
> 1. Select **+ Create**.
> 
> 1. In the **Basics** tab of **Create a resource group**, enter, or select the following information:
> 
>     | Setting | Value |
>     | ------- | ----- |
>     | Subscription | Select your subscription. |
>     | Resource group | Enter **test-rg**. |
>     | Region | Select **East US 2**. |
> 
> 1. Select **Review + create**.
> 
> 1. Select **Create**.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 24. lexico | granite4.1-8b — q048
**Pregunta:** What is Azure Cosmos DB and what are its main capabilities?

**Claim:** EnableMongo: Enables support for native MongoDB features..

**Mejor evidencia** (azure/Cosmos DB):

>  Cosmos DB for MongoDB accounts with Customer Managed Keys (CMK).
> Note2This capability cannot be enabled on an Azure Cosmos DB for MongoDB accounts with continuous backup.
> ImportantChanging capabilities using Azure Resource Manager is not available for Azure Cosmos DB for MongoDB accounts. [AZURE > Cosmos DB > Configure your Azure Cosmos DB for MongoDB account capabilities > Enable a capability] Retrieve your existing account capabilities by usingaz cosmosdb show:az cosmosdb show \
>     --resource-group <azure_resource_group> \
>     --name <azure_cosmos_db_account_name>You should see a capability section that's similar to this example output:"capabilities": [
>   {
>     "name": "EnableMongo"
>   }
> ]Review the default capability. In this example, the only capability that's set isEnableMongo.Set th

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 25. hibrido | gemma4-e4b — q195
**Pregunta:** Compare the CI/CD pipeline offerings: AWS CodePipeline vs Azure DevOps vs Google Cloud Build

**Claim:** Functionality: Google Cloud Build is recommended as part of the best practice for configuring an application CI/CD pipeline for GKE.

**Mejor evidencia** (azure/Azure Functions):

>  sure to set `--build-remote true` to perform a [remote build].
> ::: zone-end [AZURE > Azure Functions > Create and manage function apps in the Flex Consumption plan > Deploy your code project > [Continuous Deployment](#tab/continuous-deployment)] Azure Functions has both a custom GitHub Action and a custom Azure Pipelines Task to support continuous deployment. Refer the following guides to incorporate these tools in your CI/CD pipelines:
> 
> - [Build and deploy using Azure Pipelines](./functions-how-to-azure-devops.md)
> - [Build and deploy using GitHub Actions](./functions-how-to-github-actions.md)

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 26. lexico | granite4.1-8b — q076
**Pregunta:** How to configure auto-scaling in AWS EC2?

**Claim:** [CODE]: This optional parameter specifies that only aggregated metrics for the Auto Scaling group should be reported..

**Mejor evidencia** (gcp/GKE):

>  to manually
> recreate these resources:1.17.12-gke.1501 and higher1.18.9-gke.1501 and higherThe issue only occurs during upgrades, so new clusters created in earlier
> versions are unaffected. [GCP > GKE > GKE release notesStay organized with collectionsSave and categorize content based on your preferences. > October 16, 2020] IssueThere is a known issue impacting both LTSC and SAC Windows Server images on GKE
> versions 1.17.x and 1.18.x. New Windows nodes take longer to join the cluster
> which may cause node pool creation, auto-scaling, and auto-repair operations to
> time out.We recommend not upgrading clusters with Windows Server nodes pools to the
> following versions:1.17.12-gke.15011.18.9-gke.1501GKE versions beyond the affected versions will not have the startup time
> regression. Upgrade to v

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 27. denso | granite4.1-8b — q173
**Pregunta:** How do managed Kubernetes services compare across AWS EKS, Azure AKS, and GKE?

**Claim:** Users do not need to manage or maintain the underlying Kubernetes components such as the API server, controller manager, scheduler, etc..

**Mejor evidencia** (gcp/GKE):

> , and optimize
> container workloads depending on where they're running. GKE Multi-Cloud
> and Google Distributed Cloud both extend GKE for use outside
> Google Cloud, letting you create and manage hybrid or entirely
> on-premises deployments. [GCP > GKE > Explore GKE documentationStay organized with collectionsSave and categorize content based on your preferences. > GKE outside Google Cloud > GKE Multi-Cloud] GKE on AWS: Work with
> GKE clusters running on AWS infrastructure.GKE on Azure: Work
> with GKE clusters running on Azure infrastructure.GKE attached clusters:
> Add CNCF-conformant Kubernetes clusters to your fleet to view and manage
> along with your GKE clusters, with instructions for EKS, AKS,
> and other conformant cluster types.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 28. denso | granite4.1-8b — q012
**Pregunta:** What is the SLA for AWS RDS?

**Claim:** For precise SLA information, one would need to refer directly to Amazon's official AWS Service Level Agreements documentation or contact AWS support.

**Mejor evidencia** (aws/EC2):

>  the CloudWatch agent\. The agent enables you to select individual metrics and control publication\.
> 
> You can also use the ethtool to retrieve the metrics for each network interface, such as eth0, as follows\.
> 
> 
> [CODE_BLOCK_1] [AWS > EC2 > Monitor network performance for your EC2 instance<a name="monitoring-network-performance-ena"></a> > Metrics for ENA Express<a name="network-performance-metrics-ena-express"></a>] ENA Express is powered by AWS Scalable Reliable Datagram \(SRD\) technology\. SRD is a high performance network transport protocol that uses dynamic routing to increase throughput and minimize tail latency\. You can use ENA Express metrics to help ensure that your instances take full advantage of the performance improvements that SRD technology provides, for example:
> + Evaluate

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 29. lexico | qwen3.5-9b — q064
**Pregunta:** How to set up AWS EKS for a production workload?

**Claim:** AsciiDoc markup language used for documentation.

**Mejor evidencia** (aws/EKS):

> .aws.amazon.com/eks/latest/userguide/contribute.html) in the Amazon EKS User Guide.
> 
> We look forward to your contributions with the new GitHub experience. The improved platform makes it easier than ever to help us enhance the Amazon EKS documentation. [AWS > EKS > License Summary] The documentation is made available under the Creative Commons Attribution-ShareAlike 4.0 International License. See the LICENSE file.
> 
> The sample code within this documentation is made available under a modified MIT license. See the LICENSE-SAMPLECODE file.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 30. lexico | granite4.1-8b — q048
**Pregunta:** What is Azure Cosmos DB and what are its main capabilities?

**Claim:** Multi-Model Support: Supports various data models through different APIs: API for Table.

**Mejor evidencia** (azure/Cosmos DB):

>  Cosmos DB for MongoDB accounts with Customer Managed Keys (CMK).
> Note2This capability cannot be enabled on an Azure Cosmos DB for MongoDB accounts with continuous backup.
> ImportantChanging capabilities using Azure Resource Manager is not available for Azure Cosmos DB for MongoDB accounts. [AZURE > Cosmos DB > Configure your Azure Cosmos DB for MongoDB account capabilities > Enable a capability] Retrieve your existing account capabilities by usingaz cosmosdb show:az cosmosdb show \
>     --resource-group <azure_resource_group> \
>     --name <azure_cosmos_db_account_name>You should see a capability section that's similar to this example output:"capabilities": [
>   {
>     "name": "EnableMongo"
>   }
> ]Review the default capability. In this example, the only capability that's set isEnableMongo.Set th

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 31. denso | qwen3.5-9b — q097
**Pregunta:** How to scale Azure Monitor horizontally?

**Claim:** Azure Functions-specific monitoring metrics.

**Mejor evidencia** (azure/Azure Functions):

>  the durable orchestration client to terminate the monitors. You can change a monitor's `wait` interval based on a specific condition (for example, exponential backoff.) 
> 
> The following code implements a basic monitor:
> 
> ::: zone pivot="csharp" [AZURE > Azure Functions > [Python (V2 model)](#tab/v2-model) > <a name="monitoring"></a>Pattern #4: Monitor > [In-process](#tab/in-process)] [CODE_BLOCK_17]
> 
> [CODE_BLOCK_18]
> 
> 
> ::: zone-end
> ::: zone pivot="javascript"

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 32. hibrido | qwen3.5-9b — q041
**Pregunta:** What regions support Azure Virtual Network?

**Claim:** Virtual Network TAP - Supported in these regions: UK South.

**Mejor evidencia** (azure/Virtual Network):

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
> - Microsoft.network/AllowVirtualNetworkAppliance
> 
> Register for the preview by activating the AFEC flag in the Azure portal.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 33. hibrido | mistral-7b-instruct — q175
**Pregunta:** What is the equivalent of AWS IAM in Azure and GCP?

**Claim:** Similar to AWS IAM, it allows administrators to control who can access which resources in their GCP project.

**Mejor evidencia** (gcp/GKE):

> .io/gcp-service-account=scalable-apps@PROJECT_ID.iam.gserviceaccount.comThis configuration allows Pods that use thedefaultKubernetes service
> account in thedefaultnamespace to access the same Google Cloud
> resources as the IAM service account. [GCP > GKE > Learning Path: Scalable applications - ScalingStay organized with collectionsSave and categorize content based on your preferences. > Set up custom metrics collection] You can configure the Horizontal Pod Autoscaler to use basic built-in Kubernetes
> CPU and memory metrics, or you can use custom metrics from Cloud Monitoring
> like HTTP requests per second or the quantity ofSELECTstatements. Custom
> metrics can work without application changes, and give your cluster more insight
> into the overall performance and needs of the application. In this

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 34. denso | qwen3.5-9b — q064
**Pregunta:** How to set up AWS EKS for a production workload?

**Claim:** AWS ECS (Elastic Container Service) -.

**Mejor evidencia** (aws/ECS):

> \.
> 
> 1. For **Production listener port**, choose the listener port and protocol of the listener that you created in [Creating an Application Load Balancer](create-application-load-balancer.md) \(if applicable\), or choose **create new** to [AWS > ECS > Configuring a load balancer for the blue/green deployment type<a name="service-create-loadbalancer-bluegreen"></a>]  create a new listener and then enter a port number and choose a port protocol for **Production listener protocol**\.
> 
> 1. \(Optional\) Select **Test listener** if you want to configure a listener port and protocol on your load balancer to test updates to your service before routing traffic to your new task set\. Complete the following step:
> 
>    1. For **Test listener port**, choose the listener port and protocol of the listener 

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 35. denso | mistral-7b-instruct — q122
**Pregunta:** How to implement auto-scaling with Google Cloud Cloud Storage?

**Claim:** For Amazon ECS, auto-scaling is implemented using an Auto Scaling group capacity provider with managed scaling turned on.

**Mejor evidencia** (aws/ECS):

> **  
> Amazon ECS cluster auto scaling is only supported with Auto Scaling group capacity providers\. For Amazon ECS workloads that are hosted on AWS Fargate, see [AWS Fargate capacity providers](fargate-capacity-providers.md)\. [AWS > ECS > Amazon ECS cluster Auto Scaling<a name="cluster-auto-scaling"></a> > How cluster Auto Scaling works<a name="how-it-works"></a>] The following is the workflow used for Amazon ECS cluster auto scaling\. For more information, see [Turn on cluster Auto Scaling](turn-on-cluster-auto-scaling.md)\.
> 
> 1. Create an Auto Scaling group
> 
> 1. Create a capacity provider that uses that Auto Scaling group
> 
> 1. Turn on managed scaling for the capacity provider
> 
> 1. Associate the capacity provider with a cluster
> 
> 1. Create a capacity provider strategy that uses the capacity p

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 36. lexico | qwen3.5-9b — q117
**Pregunta:** How to implement auto-scaling with Google Cloud Cloud Run?

**Claim:** Infrastructure Management: Cloud Run manages the underlying infrastructure automatically.

**Mejor evidencia** (gcp/GKE):

>  custom infrastructure or network
> configurations, and scenarios where deep control over Kubernetes is essential.
> Cloud Run is best suited for event-driven apps. It's ideal for
> stateless web services, APIs, batch jobs, and other workloads that benefit from [GCP > GKE > GKE and Cloud RunStay organized with collectionsSave and categorize content based on your preferences.] 
> pay-per-use pricing.The preceding example demonstrates how combining GKE and
> Cloud Run can provide a powerful and flexible solution for your
> ecommerce platform. You gain the benefits of both platforms; serverless
> efficiency for stateless workloads, and Kubernetes control for complex
> microservices and stateful components.For a unified view of the disparate components that are deployed with a hybrid
> approach and to see how t

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 37. denso | qwen3.5-9b — q082
**Pregunta:** How to configure networking for Azure App Service?

**Claim:** Connect the back end of your apps to your virtual networks by using virtual network integration.

**Mejor evidencia** (azure/Virtual Network):

> 's in the deallocated state. The MAC address remains assigned to the network adapter until you do one of these tasks:
> 
> * Delete the network adapter.
> * Change the private IP address that's assigned to the primary IP configuration of the primary network adapter. [AZURE > Virtual Network > Azure Virtual Network frequently asked questions (FAQ) > Azure services that connect to virtual networks > Can I use Web Apps with a virtual network?] Yes. You can deploy the Web Apps feature of Azure App Service inside a virtual network by using an App Service Environment. You can then:
> 
> * Connect the back end of your apps to your virtual networks by using virtual network integration.
> * Lock down inbound traffic to your app by using service endpoints.
> 
> For more information, see the following articles:
> 
> * [

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 38. denso | mistral-7b-instruct — q082
**Pregunta:** How to configure networking for Azure App Service?

**Claim:** In your Function App, expand Settings, and then select Networking..

**Mejor evidencia** (azure/Virtual Network):

> 's in the deallocated state. The MAC address remains assigned to the network adapter until you do one of these tasks:
> 
> * Delete the network adapter.
> * Change the private IP address that's assigned to the primary IP configuration of the primary network adapter. [AZURE > Virtual Network > Azure Virtual Network frequently asked questions (FAQ) > Azure services that connect to virtual networks > Can I use Web Apps with a virtual network?] Yes. You can deploy the Web Apps feature of Azure App Service inside a virtual network by using an App Service Environment. You can then:
> 
> * Connect the back end of your apps to your virtual networks by using virtual network integration.
> * Lock down inbound traffic to your app by using service endpoints.
> 
> For more information, see the following articles:
> 
> * [

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 39. denso | mistral-7b-instruct — q019
**Pregunta:** What are the limits and quotas for AWS EC2?

**Claim:** Fleet quotas for EC2 Fleets and Spot Fleets: An EC2 Fleet request or a Spot Fleet request can't span Regions or different subnets from the same Availability Zone ..

**Mejor evidencia** (aws/EC2):

>  20 running On\-Demand Instances, and you purchase 20 zonal Reserved Instances, you can launch a further 20 On\-Demand Instances that match the specifications of your zonal Reserved Instances, giving you a total of 40 running instances\. [AWS > EC2 > Reserved Instance quotas<a name="ri-limits"></a> > View your Reserved Instance quotas and request a quota increase<a name="view-ri-quotas"></a>] The Amazon EC2 console provides quota information\. You can also request an increase in your quotas\. For more information, see [View your current quotas](ec2-resource-limits.md#view-limits) and [Request an increase](ec2-resource-limits.md#request-increase)\.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 40. lexico | mistral-7b-instruct — q187
**Pregunta:** Compare the machine learning platforms: AWS SageMaker vs Azure Machine Learning vs Google Vertex AI

**Claim:** Through their partnership with NVIDIA, they deliver the latest GPUs while optimizing the software stack with a wide.

**Mejor evidencia** (gcp/Compute Engine):

> [GCP > Compute Engine > TPU locationsStay organized with collectionsSave and categorize content based on your preferences.] Accelerators, such as Tensor Processing
> Units (TPUs), are specialized hardware components that can significantly speed
> up machine learning and high performance computing workloads. When you create
> Compute Engine (VM) instances or other resources on Google Cloud, you can
> attach accelerators to them. For an overview of these accelerators, see theCloud TPUdocumentation.The availability of specific TPU models varies by region and zone. This
> document provides a comprehensive list of accelerator availability to help you
> plan your deployments and select the appropriate locations for your resources.This document focuses on accelerator availability for Compute Engine,Google Ku

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 41. lexico | qwen3.5-9b — q164
**Pregunta:** When should you choose Azure Blob Storage over Azure Entra ID?

**Claim:** These services serve different purposes:.

**Mejor evidencia** (azure/Blob Storage):

>  accounts in all of the resource groups in all of the subscriptions in the management group.
> 
> For more information about scope for Azure RBAC role assignments, see [Understand scope for Azure RBAC](../../role-based-access-control/scope-overview.md). [AZURE > Blob Storage > Authorize access to blobs using Microsoft Entra ID > Assign Azure roles for access rights > Azure built-in roles for blobs] Azure RBAC provides several built-in roles for authorizing access to blob data using Microsoft Entra ID and OAuth. Some examples of roles that provide permissions to data resources in Azure Storage include:
> 
> - [Storage Blob Data Owner](../../role-based-access-control/built-in-roles.md#storage-blob-data-owner): Use to set ownership and manage POSIX access control for Azure Data Lake Storage. For more

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 42. hibrido | granite4.1-8b — q075
**Pregunta:** How to monitor AWS RDS with CloudWatch?

**Claim:** Create Alarms (Optional): To set up notifications for specific thresholds, create CloudWatch alarms based on the RDS metrics..

**Mejor evidencia** (aws/EC2):

> \.com/ec2/](https://console.aws.amazon.com/ec2/)\.
> 
> 1. In the navigation pane, choose **Lifecycle Manager**\.
> 
> 1. Select a policy in the grid and then choose the **Monitoring** tab\.
> 
> ------ [AWS > EC2 > Monitor your policies using Amazon CloudWatch<a name="monitor-dlm-cw-metrics"></a> > View CloudWatch metrics for your policies<a name="view-metrics"></a> > [ CloudWatch console ]] **To view metrics using the Amazon CloudWatch console**
> 
> 1. Open the CloudWatch console at [https://console\.aws\.amazon\.com/cloudwatch/](https://console.aws.amazon.com/cloudwatch/)\.
> 
> 1. In the navigation pane, choose **Metrics**\.
> 
> 1. Select the **EBS** namespace and then select **Data Lifecycle Manager metrics**\.
> 
> ------

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 43. hibrido | mistral-7b-instruct — q167
**Pregunta:** Compare the scaling options of Azure Virtual Machines vs Azure Blob Storage

**Claim:** Scaling Blob Storage: In Azure Blob Storage [1], the performance of your storage requests can be increased by using more clients.

**Mejor evidencia** (azure/Blob Storage):

>  the performance of your storage requests. To learn more about NFS 3.0 support for Azure Blob Storage, see [Network File System (NFS) 3.0 protocol support for Azure Blob storage](network-file-system-protocol-support.md). [AZURE > Blob Storage > Network File System (NFS) 3.0 performance considerations in Azure Blob storage > Add clients to increase throughput] Azure Blob Storage scales linearly until it reaches the maximum storage account egress and ingress limit. Therefore, your applications can achieve higher throughput by using more clients. To view storage account egress and ingress limits, see [Scalability and performance targets for standard storage accounts](../common/scalability-targets-standard-account.md).
> 
> The following chart shows how bandwidth increases as you add more clients.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 44. hibrido | qwen3.5-9b — q070
**Pregunta:** How to set up AWS ECS for a production workload?

**Claim:** Container instances require external network access to communicate with the Amazon ECS service endpoint.

**Mejor evidencia** (aws/ECS):

> _started_prereqs.html)\.
> + The IAM ecsInstanceRole and ecsTaskExecutionRole permissions are associated with your account\. These service\-linked roles allow services to make API calls and access containers, secrets, directories and file servers on your behalf\. [AWS > ECS > Tutorial: Using FSx for Windows File Server file systems with Amazon ECS<a name="tutorial-wfsx-volumes"></a> > Step 1: Create IAM access roles<a name="iam-roles"></a>] **Create a cluster with the AWS Management Console\.**
> 
> 1. See [Amazon ECS container instance IAM role](instance_IAM_role.md) to check whether you have an ecsInstanceRole and to see how you can create one if you don't have one\.
> 
> 1. We recommend that role policies are customized for minimum permissions in an actual production environment\. For the purpose

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 45. hibrido | mistral-7b-instruct — q088
**Pregunta:** How to deploy an application using Azure Blob Storage?

**Claim:** In the storage account, select Containers under Data storage..

**Mejor evidencia** (azure/Azure Functions):

>  ensure you enable [Always On](dedicated-plan.md#always-on).
> + You can use [Azure Storage Explorer](../vs-azure-tools-storage-manage-with-storage-explorer.md) to upload package files to blob containers in your storage account. [AZURE > Azure Functions > Run your functions from a package file in Azure > Use WEBSITE_RUN_FROM_PACKAGE = URL > Manually uploading a package to Blob Storage] To deploy a zipped package when using the URL option, you must create a .zip compressed deployment package and upload it to the destination. The following procedure deploys to a container in Blob Storage:
> 
> 1. Create a .zip package for your project using the utility of your choice.
> 
> 1. In the [Azure portal](https://portal.azure.com), search for your storage account name or browse for it in the storage accounts 

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 46. hibrido | granite4.1-8b — q169
**Pregunta:** Compare Azure App Service and Azure AKS in terms of capabilities and pricing

**Claim:** The choice of tier determines the features available for managing your cluster, with Standard being the default for Automatic SKU clusters ((azure/AKS/Core concepts for Azure Kubernetes Service (AKS) > Pricing tiers))..

**Mejor evidencia** (azure/AKS):

>  nodes, scaling, security, and other preconfigured settings. AKS Standard provides more control over the cluster configuration, including the ability to manage node pools, scaling, and other settings.
> For more information, seeAKS Automatic and Standard feature comparison. [AZURE > AKS > Core concepts for Azure Kubernetes Service (AKS) > Pricing tiers] AKS offers three pricing tiers for cluster management: Free, Standard, and Premium. The pricing tier you choose determines the features that are available for managing your cluster.
> For more information, seePricing tiers for AKS cluster management.
> 
> For more information, seeSupported Kubernetes versions in AKS.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 47. hibrido | gemma4-e4b — q029
**Pregunta:** What is AWS ECS and what are its main features?

**Claim:** Amazon Elastic Container Service (Amazon ECS) is a fully managed container orchestration service designed to help users easily deploy, manage, and scale containerized applications [Source: Source: Source: Source: Source: Source: Source: Source: Source: Source: Source: Source: Source: Source: Source: Source: Source: Source: Source: Source: Source: Source: Source

**Mejor evidencia** (aws/EC2):

>  scale up or down to handle changes in requirements or spikes in popularity, reducing your need to forecast traffic\.
> 
> For more information about cloud computing, see [What is cloud computing?](https://aws.amazon.com/what-is-cloud-computing/) [AWS > EC2 > What is Amazon EC2?<a name="concepts"></a> > Features of Amazon EC2<a name="ec2-features"></a>] Amazon EC2 provides the following features:
> + Virtual computing environments, known as *instances*
> + Preconfigured templates for your instances, known as *Amazon Machine Images \(AMIs\)*, that package the bits you need for your server \(including the operating system and additional software\)
> + Various configurations of CPU, memory, storage, and networking capacity for your instances, known as *instance types*
> + Secure login information for you

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 48. denso | granite4.1-8b — q115
**Pregunta:** How to configure IAM permissions for Google Cloud Pub/Sub?

**Claim:** For precise steps and configurations specific to Pub/Sub, refer directly to the [Google Cloud IAM documentation for Pub/Sub](https://cloud.google.com/pubsub/iam) or similar authoritative sources outside the provided context.

**Mejor evidencia** (gcp/Compute Engine):

> manage access to custom imageswith IAM.Learn more aboutService accounts.Learn more aboutCompute Engine IAM roles.Learn more about the permissions that are included inpredefined
> Compute Engine IAM roles.Learn how to create and managecustom roles.
> 
> Send feedback [GCP > Compute Engine > Managing access to Compute Engine resourcesStay organized with collectionsSave and categorize content based on your preferences. > Before you begin] Review theIAM overview.Read the Compute Engineaccess control overview.Familiarize yourself withIAM roles for Compute Engine.If you haven't already, set upauthentication.
>   Authentication verifies your identity for access to Google Cloud services and APIs. To run
>   code or samples from a local development environment, you can authenticate to
>   Compute Engine by sel

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 49. denso | granite4.1-8b — q104
**Pregunta:** How to implement auto-scaling with Google Cloud GKE?

**Claim:** Deploy the Custom Metrics Stackdriver Adapter: This adapter allows the HPA to read custom metrics from Monitoring..

**Mejor evidencia** (gcp/GKE):

>  configuration, autoscaling, and security constraints.Optional:You create and manage nodes in node pools.You let GKE manage some nodes by using anAutopilot ComputeClassin your Standard cluster.Node provisioning and scalingPre-configured: Autopilot automatically
>  [GCP > GKE > Compare features in Autopilot and Standard clustersStay organized with collectionsSave and categorize content based on your preferences.]         scales the quantity and size of nodes based on Pods in the cluster.Optional:Horizontal Pod autoscaling.Vertical Pod autoscaling:
>           enabled in the cluster, but GKE doesn't configure any
>           Pod autoscaling parameters by default.Default:Manually provision new nodesManually specify node resourcesOptional:Node auto-provisioningCluster autoscalerHorizontal Pod autosc

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 50. denso | mistral-7b-instruct — q075
**Pregunta:** How to monitor AWS RDS with CloudWatch?

**Claim:** In the "CloudWatch alarms" section, create a new CloudWatch alarm for your RDS instance by clicking on "Create Alarm".

**Mejor evidencia** (aws/ECS):

> AWS provides various tools that you can use to monitor Amazon ECS\. You can configure some of these tools to do the monitoring for you, while some of the tools require manual intervention\. We recommend that you automate monitoring tasks as much as possible\. [AWS > ECS > Monitoring tools<a name="monitoring-automated-manual"></a> > Automated monitoring tools<a name="monitoring-automated_tools"></a>] You can use the following automated monitoring tools to watch Amazon ECS and report when something is wrong:
> + Amazon CloudWatch alarms – Watch a single metric over a time period that you specify, and perform one or more actions based on the value of the metric relative to a given threshold over a number of time periods\. The action is a notification sent to an Amazon Simple Notification Servic

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 51. hibrido | gemma4-e4b — q195
**Pregunta:** Compare the CI/CD pipeline offerings: AWS CodePipeline vs Azure DevOps vs Google Cloud Build

**Claim:** It is used alongside Cloud Deploy and Artifact Registry [Source: gcp/GKE/GKE overviewStay organized with collectionsSave.

**Mejor evidencia** (gcp/GKE):

>  deprecations.Multi-cluster Service capabilities.
> Best practice:Configure the application CI/CD pipeline to use Cloud Build,
> Cloud Deploy, and Artifact Registry.
> By using  managed build and deployment services,
> you can optimize for security, scale, and simplicity. [GCP > GKE > GKE overviewStay organized with collectionsSave and categorize content based on your preferences. > When to use GKE > Use cases for GKE] GKE and Kubernetes are used in a variety of industries, including
> robotics, healthcare, retail, education, gaming, and financial services. Examples of workloads you can run include:
> AI and ML operationsData processing at scaleScalableonline games platformsReliable applications under heavy load
> For case studies by industry and application, refer toGoogle Cloud customers.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 52. denso | granite4.1-8b — q030
**Pregunta:** What is Azure Blob Storage and what are its main capabilities?

**Claim:** Scalability and Durability: Azure Blob Storage supports both standard and premium storage accounts with replicated data to ensure durability and high availability.

**Mejor evidencia** (azure/Blob Storage):

>  direct access from multiple clients, simplifies the classic deployment model by eliminating the need to attach/detach disks, and eliminates the risk of issues on the VM. And, it provides the same level of performance for random read/write operations as a disk [AZURE > Blob Storage > Overview of Azure page blobs > Page blob features > Durability and high availability] Both standard and premium storage are durable storage where the page blob data is always replicated to ensure durability and high availability.Azure has consistently delivered enterprise-grade durability for IaaS disks and page blobs, with an industry-leading zero percent [Annualized Failure Rate](https://en.wikipedia.org/wiki/Annualized_failure_rate).
> 
> For more information about Azure Storage redundancy for standard and prem

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 53. denso | qwen3.5-9b — q021
**Pregunta:** What is AWS CloudWatch and what are its main features?

**Claim:** Graphs of alarms and resources.

**Mejor evidencia** (aws/ECS):

> AWS provides various tools that you can use to monitor Amazon ECS\. You can configure some of these tools to do the monitoring for you, while some of the tools require manual intervention\. We recommend that you automate monitoring tasks as much as possible\. [AWS > ECS > Monitoring tools<a name="monitoring-automated-manual"></a> > Automated monitoring tools<a name="monitoring-automated_tools"></a>] You can use the following automated monitoring tools to watch Amazon ECS and report when something is wrong:
> + Amazon CloudWatch alarms – Watch a single metric over a time period that you specify, and perform one or more actions based on the value of the metric relative to a given threshold over a number of time periods\. The action is a notification sent to an Amazon Simple Notification Servic

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 54. denso | qwen3.5-9b — q146
**Pregunta:** Compare AWS S3 and AWS CloudWatch in terms of use cases and features

**Claim:** AWS ECS CloudWatch metrics.

**Mejor evidencia** (aws/EC2):

>  by the type of instance running\. For example, you might compare data from an m1\.small instance and an m1\.large instance to determine which has the better business value for your application\. Available for instances with Detailed Monitoring enabled\.  | [AWS > EC2 > List the available CloudWatch metrics for your instances<a name="viewing_metrics_with_cloudwatch"></a> > Amazon EC2 usage metrics<a name="service-quota-metrics"></a>] You can use CloudWatch usage metrics to provide visibility into your account's usage of resources\. Use these metrics to visualize your current service usage on CloudWatch graphs and dashboards\.
> 
> Amazon EC2 usage metrics correspond to AWS service quotas\. You can configure alarms that alert you when your usage approaches a service quota\. For more information

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 55. hibrido | granite4.1-8b — q072
**Pregunta:** How to secure AWS VPC using IAM policies?

**Claim:** [Example IAM policies for ClassicLink](#iam-example-classiclink).

**Mejor evidencia** (aws/EC2):

> 1. In the navigation pane, choose **Your VPCs**\.
> 
> 1. Select your VPC\.
> 
> 1. Choose **Actions**, **Edit VPC settings**\.
> 
> 1. For **ClassicLink**, deselect **Enable**\. [AWS > EC2 > ClassicLink<a name="vpc-classiclink"></a> > Example IAM policies for ClassicLink<a name="iam-example-classiclink"></a>] You can enable a VPC for ClassicLink and then link an EC2\-Classic instance to the VPC\. You can also view your ClassicLink\-enabled VPCs, and all of your EC2\-Classic instances that are linked to a VPC\. You can create policies with resource\-level permission for the `ec2:EnableVpcClassicLink`, `ec2:DisableVpcClassicLink`, `ec2:AttachClassicLinkVpc`, and `ec2:DetachClassicLinkVpc` actions to control how users are able to use those actions\. Resource\-level permissions are not supported for `ec2

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 56. lexico | granite4.1-8b — q044
**Pregunta:** What are the service limits for Azure AKS?

**Claim:** Upcoming Changes: Starting September 2025, a rollout will enable quota for all current and new AKS customers, enforcing limits on the maximum number of managed clusters per region..

**Mejor evidencia** (azure/AKS):

> /etcd instances.Free tierLimited resources withinflight requests limit. Recommended node limit of 10 nodes per cluster. Best for experimenting, learning, and simple testing.Not advised for production/critical workloads.
> LINKS - Internal
> LINKS - External [AZURE > AKS > Quotas, virtual machine size restrictions, and region availability in Azure Kubernetes Service (AKS) > Service quotas and limits > Quota limits on AKS Managed Clusters] Starting in September 2025, Azure Kubernetes Service will begin rolling out a change to enable quota for all current and new AKS customers. This rollout is expected to take place between September 1-30, 2025.
> AKS quota will represent a limit of the maximum number of managed clusters (AKS clusters) that an Azure subscription can create per region. Once managed 

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 57. hibrido | qwen3.5-9b — q181
**Pregunta:** Compare monitoring solutions: AWS CloudWatch vs Azure Monitor vs Google Cloud Monitoring

**Claim:** AWS CloudWatch: The provided context contains detailed information about AWS CloudWatch monitoring solutions, including: CloudWatch Logs for container and container instance logs.

**Mejor evidencia** (aws/ECS):

> [AWS > ECS > Logging and Monitoring in Amazon Elastic Container Service<a name="ecs-logging-monitoring"></a>] Monitoring is an important part of maintaining the reliability, availability, and performance of Amazon Elastic Container Service and your AWS solutions\. You should collect monitoring data from all of the parts of your AWS solution so that you can more easily debug a multi\-point failure if one occurs\. AWS provides several tools for monitoring your Amazon ECS resources and responding to potential incidents:
> 
> **Amazon CloudWatch Alarms**  
> Watch a single metric over a time period that you specify, and perform one or more actions based on the value of the metric relative to a given threshold over a number of time periods\. The action is a notification sent to an Amazon Simple Notif

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 58. denso | mistral-7b-instruct — q118
**Pregunta:** How to implement auto-scaling with Google Cloud VPC?

**Claim:** Create a new GKE cluster using Autopilot mode: Choose a name for your cluster, set the node pool configuration, and click on "Create"..

**Mejor evidencia** (aws/ECS):

> **  
> Amazon ECS cluster auto scaling is only supported with Auto Scaling group capacity providers\. For Amazon ECS workloads that are hosted on AWS Fargate, see [AWS Fargate capacity providers](fargate-capacity-providers.md)\. [AWS > ECS > Amazon ECS cluster Auto Scaling<a name="cluster-auto-scaling"></a> > How cluster Auto Scaling works<a name="how-it-works"></a>] The following is the workflow used for Amazon ECS cluster auto scaling\. For more information, see [Turn on cluster Auto Scaling](turn-on-cluster-auto-scaling.md)\.
> 
> 1. Create an Auto Scaling group
> 
> 1. Create a capacity provider that uses that Auto Scaling group
> 
> 1. Turn on managed scaling for the capacity provider
> 
> 1. Associate the capacity provider with a cluster
> 
> 1. Create a capacity provider strategy that uses the capacity p

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 59. hibrido | granite4.1-8b — q076
**Pregunta:** How to configure auto-scaling in AWS EC2?

**Claim:** Purchasing Option: Select "Request Spot instances" if you want to use Spot Instances with adjustable maximum prices and interruption behaviors..

**Mejor evidencia** (aws/EC2):

> /ec2/)\.
> 
> 1. In the navigation pane, choose **Spot Requests**\.
> 
> 1. Select your Spot Fleet request and choose **Auto Scaling**\.
> 
> 1. If automatic scaling is not configured, choose **Configure**\.
> 
> 1 [AWS > EC2 > Scale Spot Fleet using step scaling policies<a name="spot-fleet-step-scaling"></a>] . Use **Scale capacity between** to set the minimum and maximum capacity for your fleet\. Automatic scaling does not scale your fleet below the minimum capacity or above the maximum capacity\.
> 
> 1. Initially, **Scaling policies** contains policies named ScaleUp and ScaleDown\. You can complete these policies, or choose **Remove policy** to delete them\. You can also choose **Add policy**\.
> 
> 1. To define a policy, do the following:
> 
>    1. For **Policy name**, enter a name for the policy\.
> 
>    1. For *

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 60. denso | mistral-7b-instruct — q070
**Pregunta:** How to set up AWS ECS for a production workload?

**Claim:** Create an Amazon ECS service to run and maintain the specified number of instances of your task definition simultaneously in the cluster .

**Mejor evidencia** (aws/ECS):

>  `1` for *capacityProviderA* and a weight of `4` for *capacityProviderB*\. Then, for every one task that's run using *capacityProviderA*, there are four tasks that use *capacityProviderB*\. [AWS > ECS > Amazon ECS capacity providers<a name="cluster-capacity-providers"></a> > Capacity provider types<a name="capacity-providers-types"></a>] The infrastructure that your Amazon ECS workloads are run on determines the type of capacity provider that you can use\.
> 
> For Amazon ECS workloads that are hosted on Fargate, the following predefined capacity providers are available:
> + Fargate
> + Fargate Spot
> 
> For Amazon ECS workloads that are hosted on Amazon EC2 instances, you must create and maintain a capacity provider that consists of the following components:
> + A name
> + An Auto Scaling group
> + The set

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 61. denso | qwen3.5-9b — q128
**Pregunta:** When should you use AWS VPC instead of AWS RDS?

**Claim:** RDS provides database hosting.

**Mejor evidencia** (aws/EC2):

>  that you can use, you can skip this task\.
> 
> **Important**  
> If you use an existing RDS database, make sure that it is in the same VPC as your EC2 instance so that you can use the automatic connection feature\. [AWS > EC2 > Task 1: Create an RDS database – *optional*<a name="option1-task1-create-rds-database"></a> > Steps to create an RDS database<a name="option1-task1-create-rds-database-steps"></a>] Use the following steps to create an RDS database\.
> 
> To view an animation of these steps, see [View an animation: Create an RDS database](#task1-create-rds-database-animation)\.
> 
> **RDS database configuration**
> 
> The steps in this task configure the RDS database as follows:
> + Engine type: MySQL
> + Template: Free tier
> + DB instance identifier: **tutorial\-database\-1**
> + DB instance class: `db.t3

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 62. denso | granite4.1-8b — q136
**Pregunta:** What are the differences between AWS CloudWatch and AWS EKS?

**Claim:** Metrics are filtered by [CODE], [CODE], [CODE], etc., reflecting only resources with running tasks..

**Mejor evidencia** (aws/ECS):

> [AWS > ECS > Viewing Amazon ECS metrics<a name="viewing_cloudwatch_metrics"></a>] After you have turned on CloudWatch metrics for Amazon ECS, you can view those metrics on the Amazon ECS and CloudWatch consoles\. The Amazon ECS console provides a 24\-hour maximum, minimum, and average view of your cluster and service metrics\. The CloudWatch console provides a fine\-grained and customizable display of your resources, as well as the number of running tasks in a service\.
> 
> **Topics**
> + [Viewing cluster metrics using the Amazon ECS console](#viewing_cluster_metrics)
> + [Viewing service metrics using the Amazon ECS console](#viewing_service_metrics)
> + [Viewing Amazon ECS metrics using the CloudWatch console](#viewing_metrics_console)

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 63. denso | granite4.1-8b — q029
**Pregunta:** What is AWS ECS and what are its main features?

**Claim:** External Instance Option with ECS Anywhere: Allows management of on-premises container workloads using the same tools as AWS-managed clusters, extending ECS capabilities to hybrid environments..

**Mejor evidencia** (aws/ECS):

> [AWS > ECS > AWS services integrated with Amazon ECS<a name="ecs-integrations"></a>] Amazon ECS works with other AWS services to provide additional solutions for your business challenges\. This topic identifies services that either use Amazon ECS to add functionality, or services that Amazon ECS uses to perform tasks\.
> 
> **Topics**
> + [Using Amazon ECR with Amazon ECS](ecr-repositories.md)
> + [Creating Amazon ECS resources with AWS CloudFormation](creating-resources-with-cloudformation.md)
> + [Amazon Elastic Container Service on AWS Outposts](ecs-on-outposts.md)
> + [Use App Mesh with Amazon ECS](gs-app-mesh.md)
> + [AWS Deep Learning Containers on Amazon ECS](deep-learning-containers.md)

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 64. hibrido | granite4.1-8b — q116
**Pregunta:** How to set up logging and monitoring for Google Cloud Cloud Functions?

**Claim:** [GCP > Cloud Functions > Functions audit logging > API interface audit logs](https://cloud.google.com/functions/docs/audit-logging#api_interface).

**Mejor evidencia** (gcp/Cloud Functions):

> FunctionsService.ListFunctionsAudit log type:Data accessPermissions:cloudfunctions.functions.list - ADMIN_READMethod is a long-running or streaming operation:
> No.Filter for this method:protoPayload.methodName="google.cloud.functions.v1.CloudFunctionsService.ListFunctions" [GCP > Cloud Functions > Functions audit logging > API interface audit logs > google.cloud.functions.v1.CloudFunctionsService > SetIamPolicy] Method:SetIamPolicyAudit log type:Admin activityPermissions:cloudfunctions.functions.setIamPolicy - ADMIN_WRITEMethod is a long-running or streaming operation:
> No.Filter for this method:protoPayload.methodName="SetIamPolicy"
> 
> Method:google.cloud.functions.v1.CloudFunctionsService.UpdateFunctionAudit log type:Admin activityPermissions:cloudfunctions.functions.update - ADMIN_WRITEMeth

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 65. hibrido | qwen3.5-9b — q021
**Pregunta:** What is AWS CloudWatch and what are its main features?

**Claim:** Watch a single metric over a time period that you specify.

**Mejor evidencia** (aws/EC2):

>  scale up or down to handle changes in requirements or spikes in popularity, reducing your need to forecast traffic\.
> 
> For more information about cloud computing, see [What is cloud computing?](https://aws.amazon.com/what-is-cloud-computing/) [AWS > EC2 > What is Amazon EC2?<a name="concepts"></a> > Features of Amazon EC2<a name="ec2-features"></a>] Amazon EC2 provides the following features:
> + Virtual computing environments, known as *instances*
> + Preconfigured templates for your instances, known as *Amazon Machine Images \(AMIs\)*, that package the bits you need for your server \(including the operating system and additional software\)
> + Various configurations of CPU, memory, storage, and networking capacity for your instances, known as *instance types*
> + Secure login information for you

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 66. denso | granite4.1-8b — q041
**Pregunta:** What regions support Azure Virtual Network?

**Claim:** Azure Virtual Network Routing Appliance (Public Preview) supports the following regions: East US.

**Mejor evidencia** (azure/Virtual Network):

>                           |
>     | **Resource group**     | Enter *myResourceGroup*                           |
>     | **Region**             | Select **(US) East US**                           |
> 
> 1. Select **Review + create**, and then select **Create**. [AZURE > Virtual Network > [Azure portal](#tab/azureportal) > Create a virtual network] 1. In the portal, search for and select **Virtual networks**.
> 
> 1. Select **+ Create**.
> 
> 1. On the **Basics** tab of **Create virtual network**, enter or select the following values:
> 
>     | Setting                | Value                                             |
>     |------------------------|---------------------------------------------------|
>     | **Subscription**       | Select your subscription                          |
>     | **Resource group**  

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 67. lexico | qwen3.5-9b — q079
**Pregunta:** How to deploy a containerized application on AWS EC2?

**Claim:** AWS CDK for ECS Deployments
****.

**Mejor evidencia** (gcp/GKE):

> Learn how to deploy and manage a containerized agentic AI application on GKE, using the Agent Development Kit (ADK) and vLLM for scalable inference with Llama 3.1.
> TutorialAI/ML InferenceAgentic AI [GCP > GKE > AI/ML orchestration on GKE documentation > Deploy an agentic AI application on GKE with the Agent Development Kit (ADK) and Vertex AI] Learn how to deploy and manage a containerized agentic AI application on GKE, using the Agent Development Kit (ADK) and Vertex AI for scalable inference with Gemini 2.0 Flash.
> TutorialAI/ML InferenceAgentic AI
> 
> Learn how to deploy LLMs using Tensor Processing Units (TPUs) on GKE with the Optimum TPU serving framework from Hugging Face.
> TutorialAI/ML InferenceTPU

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 68. hibrido | granite4.1-8b — q125
**Pregunta:** How to implement auto-scaling with Google Cloud BigQuery?

**Claim:** Utilize BigQuery's Automatic Scaling: Ensure that you have sufficient quota limits set in your Google Cloud project to accommodate high-throughput queries..

**Mejor evidencia** (gcp/Compute Engine):

> healing.
> Autoscaling lets you deal with spikes in traffic by scaling the number of VMs up
> or down based on specific signals. Autohealing performs health checking and, if
> necessary, automatically recreates unhealthy VMs.MIGs [GCP > Compute Engine > Designing resilient systemsStay organized with collectionsSave and categorize content based on your preferences.]  are also available for regions, so you can create a group of VMs
> distributed across multiple zones within a single region. For more information,
> seeCreating and managing regional MIGs.Use load balancingGoogle Cloud offers a load balancing service that helps you support periods of
> heavy traffic so that you don't overload your compute instances. WithCloud Load Balancing, you can
> do the following:Deploy your application on VMs within mu

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 69. denso | granite4.1-8b — q182
**Pregunta:** How do load balancing services differ across AWS ELB, Azure Load Balancer, and Google Cloud Load Balancing?

**Claim:** AWS ELB offers specialized ALBs and NLBs with distinct features tailored for application layer routing and high-performance TCP/UDP traffic, respectively..

**Mejor evidencia** (aws/ECS):

>  Network Load Balancers](https://docs.aws.amazon.com/elasticloadbalancing/latest/network/)\.
> 
> ![\[Network Load Balancer\]](http://docs.aws.amazon.com/AmazonECS/latest/developerguide/images/alb.png) [AWS > ECS > Load balancer types<a name="load-balancer-types"></a> > Application Load Balancer and Network Load Balancer considerations<a name="alb-considerations"></a>] The following considerations are specific to Amazon ECS services using Application Load Balancers or Network Load Balancers:
> + Amazon ECS requires the service\-linked IAM role which provides the permissions needed to register and deregister targets with your load balancer when tasks are created and stopped\. For more information, see [Using service\-linked roles for Amazon ECS](using-service-linked-roles.md)\.
> + For services tha

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 70. lexico | mistral-7b-instruct — q080
**Pregunta:** How to set up monitoring and alerts for Azure AKS?

**Claim:** Navigate to the Azure portal and create a new Log Analytics workspace.

**Mejor evidencia** (azure/AKS):

> Some services in Azure have a built-in monitoring dashboard in the Azure portal that provides a starting point for monitoring your service. These dashboards are calledinsights, and you can find them in theInsights Hubof Azure Monitor in the Azure portal. [AZURE > AKS > Monitor Azure Kubernetes Service (AKS) > AKS monitoring data: metrics, logs, integrations] AKS generates the same kinds of monitoring data as other Azure resources as described inMonitor data from Azure resources. For detailed information on the metrics and logs created by AKS, see theAKS monitoring data reference.
> Other Azure services and featurescollect other data and enable other analysis options as shown in the following diagram and table.
> 
> SourceDescriptionPlatform metricsPlatform metricsare automatically collected for 

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 71. hibrido | qwen3.5-9b — q130
**Pregunta:** Compare AWS EKS and AWS EC2 in terms of use cases and features

**Claim:** The documentation contains detailed information about AWS ECS (Elastic Container Service) and AWS EC2, but not EKS.

**Mejor evidencia** (aws/EC2):

>  whether a service or feature that makes use of instance identity roles is enabled on the instance\.
> 
> Instance identity roles are automatically created when an instance is launched, have no role\-trust policy document, and are not subject to any identity or resource policy\. [AWS > EC2 > Instance identity roles<a name="ec2-instance-identity-roles"></a> > Supported services<a name="iir-supported-services"></a>] The following AWS services use the instance identity role:
> + **Amazon EC2** – [EC2 Instance Connect](Connect-using-EC2-Instance-Connect.md) uses the instance identity role to update an EC2 instance’s host keys\.
> + **Amazon GuardDuty** – [EKS Runtime Monitoring](https://docs.aws.amazon.com/guardduty/latest/ug/guardduty-eks-runtime-monitoring.html) uses the instance identity role to al

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 72. denso | granite4.1-8b — q178
**Pregunta:** How does auto-scaling work differently in AWS EC2 vs Azure Virtual Machine Scale Sets?

**Claim:** AWS EC2 (via ECS): Specifically designed for ECS workloads, not directly applicable to EC2 instances without additional configuration..

**Mejor evidencia** (aws/ECS):

> **  
> Amazon ECS cluster auto scaling is only supported with Auto Scaling group capacity providers\. For Amazon ECS workloads that are hosted on AWS Fargate, see [AWS Fargate capacity providers](fargate-capacity-providers.md)\. [AWS > ECS > Amazon ECS cluster Auto Scaling<a name="cluster-auto-scaling"></a> > How cluster Auto Scaling works<a name="how-it-works"></a>] The following is the workflow used for Amazon ECS cluster auto scaling\. For more information, see [Turn on cluster Auto Scaling](turn-on-cluster-auto-scaling.md)\.
> 
> 1. Create an Auto Scaling group
> 
> 1. Create a capacity provider that uses that Auto Scaling group
> 
> 1. Turn on managed scaling for the capacity provider
> 
> 1. Associate the capacity provider with a cluster
> 
> 1. Create a capacity provider strategy that uses the capacity p

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 73. lexico | granite4.1-8b — q103
**Pregunta:** How to deploy a workload on Google Cloud Compute Engine?

**Claim:** Select Create an Autopilot cluster..

**Mejor evidencia** (gcp/GKE):

>  how to deploy a containerized application with Kubernetes Engine in less than 30 minutes.
> 
> This tutorial demonstrates how to use graphical processinng units (GPUs) on GKE to run large language models (LLMs) for AI/ML inference. [GCP > GKE > Google Kubernetes Engine documentation > Create a cluster and deploy a workload in the Google Cloud console] Learn how to create a Kubernetes cluster and deploy a 'hello world' web app in Google Cloud console.
> 
> This tutorial shows how to run a web application behind an external HTTP(S) load balancer by configuring the Ingress resource.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 74. lexico | mistral-7b-instruct — q196
**Pregunta:** What are the equivalent networking concepts across AWS, Azure, and GCP for VPCs, subnets, and security groups?

**Claim:** VPC (Virtual Private Cloud): A VPC (Virtual Private Cloud) network is a logically isolated section of the Google Cloud Platform that you can use to launch and manage resources..

**Mejor evidencia** (aws/ECS):

>  service to the previous deployment and configuration\. If you use another service discovery system that is based on DNS instead of Service Connect, any frontend or client applications begin using new endpoints and changed endpoint configuration after the local DNS cache expires, commonly taking multiple hours\. [AWS > ECS > Service Connect<a name="service-connect"></a> > Service Connect concepts<a name="service-connect-concepts"></a> > Networking<a name="service-connect-concepts-network"></a>] In the default configuration, the Service Connect proxy listens on the `containerPort` from the port mapping in the task definition\. If you have rules in your security group to allow this port, then you don't need to change your Amazon VPC security groups to use Service Connect\.
> 
> Even if you set a

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 75. lexico | granite4.1-8b — q173
**Pregunta:** How do managed Kubernetes services compare across AWS EKS, Azure AKS, and GKE?

**Claim:** Azure AKS: Offers advanced networking services that extend beyond Azure, including integration with other cloud providers through features like Advanced Container Networking Services..

**Mejor evidencia** (gcp/GKE):

> , and optimize
> container workloads depending on where they're running. GKE Multi-Cloud
> and Google Distributed Cloud both extend GKE for use outside
> Google Cloud, letting you create and manage hybrid or entirely
> on-premises deployments. [GCP > GKE > Explore GKE documentationStay organized with collectionsSave and categorize content based on your preferences. > GKE outside Google Cloud > GKE Multi-Cloud] GKE on AWS: Work with
> GKE clusters running on AWS infrastructure.GKE on Azure: Work
> with GKE clusters running on Azure infrastructure.GKE attached clusters:
> Add CNCF-conformant Kubernetes clusters to your fleet to view and manage
> along with your GKE clusters, with instructions for EKS, AKS,
> and other conformant cluster types.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 76. denso | granite4.1-8b — q182
**Pregunta:** How do load balancing services differ across AWS ELB, Azure Load Balancer, and Google Cloud Load Balancing?

**Claim:** Global Load Balancing: Routes traffic to the nearest location using Cloud DNS geolocation routing policies, enhancing latency reduction and redundancy across regions..

**Mejor evidencia** (gcp/Compute Engine):

> healing.
> Autoscaling lets you deal with spikes in traffic by scaling the number of VMs up
> or down based on specific signals. Autohealing performs health checking and, if
> necessary, automatically recreates unhealthy VMs.MIGs [GCP > Compute Engine > Designing resilient systemsStay organized with collectionsSave and categorize content based on your preferences.]  are also available for regions, so you can create a group of VMs
> distributed across multiple zones within a single region. For more information,
> seeCreating and managing regional MIGs.Use load balancingGoogle Cloud offers a load balancing service that helps you support periods of
> heavy traffic so that you don't overload your compute instances. WithCloud Load Balancing, you can
> do the following:Deploy your application on VMs within mu

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 77. lexico | gemma4-e4b — q110
**Pregunta:** How to configure IAM permissions for Google Cloud Compute Engine?

**Claim:** Configuring IAM permissions for Google Cloud Compute Engine involves understanding the principle of least privilege and assigning specific roles to identities at the appropriate scope (project or resource).

**Mejor evidencia** (gcp/GKE):

>  in the Google Cloud consoleGoogle Cloud CLI overviewInstallkubectland configure cluster accessProvision GKE resources with TerraformLearn how to simplify deployment from your IDE with Cloud Code in
> ourDeploy and update from an
> IDEtutorial.
> 
> Send feedback [GCP > GKE > Deploying workloadsStay organized with collectionsSave and categorize content based on your preferences. > Required roles] If you are not a project owner, you must have the following
> Identity and Access Management (IAM) role at minimum to deploy workloads:
> Kubernetes Engine Cluster Viewer (roles/container.clusterViewer): This
> provides thecontainer.clusters.getpermission, which is required to
> authenticate to clusters in a Google Cloud project. This does not authorize
> you to perform any actions inside those clusters. Your clust

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 78. hibrido | qwen3.5-9b — q092
**Pregunta:** How to implement backup and disaster recovery for Azure Virtual Machines?

**Claim:** Built-in redundancy options: Locally redundant storage (LRS) and zone-redundant storage (ZRS).

**Mejor evidencia** (azure/Virtual Machines):

>  use to ensure their performance automatically shifts to meet the changing demands of your workload, others require manual adjustment, and other disk types can't do either.
> To learn about the options each disk type has, seeOverview of options to improve Azure managed disk performance [AZURE > Virtual Machines > Introduction to Azure managed disks > Backup and disaster recovery options] Managed disks support several backup and disaster recovery options. These options include built-in redundancy options (locally redundant storage, and zone-redundant storage), Azure Backup, managed disk snapshots, restore points, and Azure Site Recovery. The ideal configuration of backup and disaster recovery options for your needs can vary. To decide which works best for your needs, seeBackup and disaster re

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 79. lexico | gemma4-e4b — q061
**Pregunta:** What is Google Cloud BigQuery and what problems does it solve?

**Claim:** Big Data Processing: It helps in analyzing big datasets on GKE by providing necessary data storage and.

**Mejor evidencia** (gcp/GKE):

>  between different namespaces on GKE.
> TutorialAI/ML Batch
> 
> This tutorial shows you how to integrate a Large Language Model application based on retrieval-augmented generation with PDF files that you upload to a Cloud Storage bucket.
> TutorialAI/ML Data Loading [GCP > GKE > AI/ML orchestration on GKE documentation > Analyze data on GKE using BigQuery, Cloud Run, and Gemma] This tutorial shows you how to analyze big datasets on GKE by leveraging BigQuery for data storage and processing, Cloud Run for request handling, and a Gemma LLM for data analysis and predictions.
> TutorialAI/ML Data Loading
> 
> Learn how to leverage GKE and Ray to efficiently preprocess large datasets for machine learning.
> MLOpsTrainingRay

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 80. hibrido | granite4.1-8b — q175
**Pregunta:** What is the equivalent of AWS IAM in Azure and GCP?

**Claim:** Azure: Azure AD + RBAC.

**Mejor evidencia** (gcp/GKE):

> .io/gcp-service-account=scalable-apps@PROJECT_ID.iam.gserviceaccount.comThis configuration allows Pods that use thedefaultKubernetes service
> account in thedefaultnamespace to access the same Google Cloud
> resources as the IAM service account. [GCP > GKE > Learning Path: Scalable applications - ScalingStay organized with collectionsSave and categorize content based on your preferences. > Set up custom metrics collection] You can configure the Horizontal Pod Autoscaler to use basic built-in Kubernetes
> CPU and memory metrics, or you can use custom metrics from Cloud Monitoring
> like HTTP requests per second or the quantity ofSELECTstatements. Custom
> metrics can work without application changes, and give your cluster more insight
> into the overall performance and needs of the application. In this

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 81. hibrido | mistral-7b-instruct — q122
**Pregunta:** How to implement auto-scaling with Google Cloud Cloud Storage?

**Claim:** In this example, replace [CODE] with the YAML file containing your application's deployment configuration and [CODE] with the name of your deployment.

**Mejor evidencia** (aws/ECS):

>  *Amazon EC2 Auto Scaling User Guide*\.
> + You have a VPC and security group created to use\. For more information, see [Create a virtual private cloud](get-set-up-for-amazon-ecs.md#create-a-vpc)\. [AWS > ECS > Tutorial: Using cluster auto scaling with the AWS Management Console and the Amazon ECS console<a name="tutorial-cluster-auto-scaling-console"></a> > Step 1: Create an Amazon ECS cluster<a name="console-tutorial-cluster"></a>] Use the following steps to create an Amazon ECS cluster\. 
> 
> Amazon ECS creates an Amazon EC2 Auto Scaling launch template and Auto Scaling group on your behalf as part of the AWS CloudFormation stack\. 
> 
> 1. Open the console at [https://console\.aws\.amazon\.com/ecs/v2](https://console.aws.amazon.com/ecs/v2)\.
> 
> 1. From the navigation bar, select the Region to us

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 82. denso | mistral-7b-instruct — q019
**Pregunta:** What are the limits and quotas for AWS EC2?

**Claim:** Fleet quotas for EC2 Fleets and Spot Fleets: The target capacity per EC2 Fleet or Spot Fleet: 10,000.

**Mejor evidencia** (aws/EC2):

>  20 running On\-Demand Instances, and you purchase 20 zonal Reserved Instances, you can launch a further 20 On\-Demand Instances that match the specifications of your zonal Reserved Instances, giving you a total of 40 running instances\. [AWS > EC2 > Reserved Instance quotas<a name="ri-limits"></a> > View your Reserved Instance quotas and request a quota increase<a name="view-ri-quotas"></a>] The Amazon EC2 console provides quota information\. You can also request an increase in your quotas\. For more information, see [View your current quotas](ec2-resource-limits.md#view-limits) and [Request an increase](ec2-resource-limits.md#request-increase)\.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 83. denso | mistral-7b-instruct — q038
**Pregunta:** What is Azure AKS and what are its main capabilities?

**Claim:** Networking: Leverage our networking options for your needs.

**Mejor evidencia** (azure/AKS):

>  process real-time data streams with millions of data points collected via sensors, and perform fast analyses and computations to develop insights into complex scenarios.Using Windows containers on AKS: Run Windows Server containers on AKS to modernize your Windows applications and infrastructure. [AZURE > AKS > What is Azure Kubernetes Service (AKS)? > Features of AKS] The following table lists some of the key features of AKS:
> FeatureDescriptionIdentity and security managementâ¢ Enforceregulatory compliance controls using Azure Policywith built-in guardrails and internet security benchmarks.â¢ Integrate withKubernetes RBACto limit access to cluster resources.â¢ UseMicrosoft Entra IDto set up Kubernetes access based on existing identity and group membership.Logging and monitoringâ¢ Int

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 84. lexico | qwen3.5-9b — q132
**Pregunta:** What are the differences between AWS Lambda and AWS EC2?

**Claim:** Comparative Answer:

Based on the provided documentation context, I cannot find sufficient information to fully answer this question about the differences between AWS Lambda and AWS EC2.

**Mejor evidencia** (aws/EC2):

> docs.aws.amazon.com/eventbridge/latest/userguide/eb-rules.html) and [Amazon EventBridge event patterns](https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-event-patterns.html) in the *Amazon EventBridge User Guide* [AWS > EC2 > Create Amazon EventBridge rules to monitor EC2 Fleet events<a name="ec2-fleet-using-eventbridge"></a> > Create an EventBridge rule to trigger a Lambda function<a name="eventbridge-trigger-lambda"></a>] The following example creates an EventBridge rule to trigger a Lambda function every time that Amazon EC2 emits an EC2 Fleet instance change notification for when an instance is launched\. The signal in this example is emitted as an `EC2 Fleet Instance Change` event, sub\-type `launched`, which triggers the action defined by the rule\.
> 
> Before creating the Ev

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 85. denso | mistral-7b-instruct — q089
**Pregunta:** How to configure networking for Azure Blob Storage?

**Claim:** Select Next, Next, and select Create..

**Mejor evidencia** (azure/Azure Functions):

>  to communicate with the newly secured storage account. [AZURE > Azure Functions > How to use a secured storage account with Azure Functions > Secure storage for an existing function app > 3. Enable application and configuration routing] > [!NOTE]
> > These configuration steps are required only for the [Elastic Premium](./functions-premium-plan.md) and [Dedicated (App Service)](./dedicated-plan.md) hosting plans.
> > The [Flex Consumption plan](./flex-consumption-plan.md) doesn't require site settings to configure networking.
> 
> You're now ready to route your function app's traffic to go through the virtual network:
> 
> 1. Enable [application routing](../app-service/overview-vnet-integration.md#application-routing) to route your app's traffic to the virtual network:
> 
>     1. In your function app, ex

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 86. hibrido | gemma4-e4b — q150
**Pregunta:** Compare the pricing models of AWS CloudWatch vs AWS IAM

**Claim:** While the context provides details regarding:.

**Mejor evidencia** (aws/ECS):

>  [CloudWatch Pricing](https://aws.amazon.com/cloudwatch/pricing/)\. Amazon ECS also provides monitoring metrics that are provided at no additional cost\. For more information, see [Amazon ECS CloudWatch metrics](cloudwatch-metrics.md)\. [AWS > ECS > Amazon ECS CloudWatch Container Insights<a name="cloudwatch-container-insights"></a> > Container Insights considerations<a name="cloudwatch-container-insights-considerations"></a>] The following should be considered when using CloudWatch Container Insights\.
> + CloudWatch Container Insights metrics only reflect the resources with running tasks during the specified time range\. For example, if you have a cluster with one service in it but that service has no tasks in a `RUNNING` state, there will be no metrics sent to CloudWatch\. If you have two

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 87. hibrido | granite4.1-8b — q157
**Pregunta:** Compare the scaling options of Azure Cosmos DB vs Azure Container Instances

**Claim:** Therefore, a direct comparison regarding scaling options between Azure Cosmos DB and ACI cannot be made from the given documentation.

**Mejor evidencia** (azure/Cosmos DB):

>  the container or from a set of containers at the back end. For example, you see that the used throughput is more than the provisioned throughput, and you're getting throttled. For more information, seeSet throughput for Azure Cosmos DB containers. [AZURE > Cosmos DB > Frequently asked questions about Azure Cosmos DB for Table > Other frequently asked questions > Can I scale up or scale down the throughput of my API for Table table?] Yes, you can use the Azure Cosmos DB portal's scale pane to scale the throughput. For more information, seeSet throughput.
> 
> Yes, if you don't override the TableThroughput via app.config and don't use a precreated container in Azure Cosmos DB, the service creates a table with throughput of 400.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 88. denso | mistral-7b-instruct — q027
**Pregunta:** What is AWS RDS and what are its main features?

**Claim:** Multi-AZ deployment: AWS RDS supports multi-AZ deployments for high availability and data durability.

**Mejor evidencia** (aws/RDS):

> [AWS > RDS > NOTICE] This repository is archived, read-only, and no longer updated. For more information, read [the announcement on the AWS News Blog](https://aws.amazon.com/blogs/aws/retiring-the-aws-documentation-on-github/).
> 
> You can find up-to-date AWS technical documentation on the [AWS Documentation website](https://docs.aws.amazon.com/), where you can also submit feedback and suggestions for improvement.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 89. denso | mistral-7b-instruct — q036
**Pregunta:** What is Azure Virtual Machines and what are its main capabilities?

**Claim:** Offering a Start/Stop VMs v2 feature that starts or stops virtual machines on user-defined schedules, provides insights through Azure Application Insights, and sends optional notifications using action groups .

**Mejor evidencia** (azure/Virtual Machines):

>  general purpose VM instances. While traditional Azure virtual machines provide fixed CPU performance, B-series virtual machines are the only VM type that use credits for CPU performance provisioning. B-series VMs utilize a CPU credit model to track how much CPU is consumed - the [AZURE > Virtual Machines > Sizes for virtual machines in Azure > List of VM size families by type > General purpose]  virtual machine accumulates CPU credits when a workload is operating below the base CPU performance threshold and uses credits when running above the base CPU performance threshold until all of its credits are consumed. Upon consuming all the CPU credits, a B-series virtual machine is throttled back to its base CPU performance until it accumulates the credits to CPU burst again.View the full B fam

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 90. hibrido | qwen3.5-9b — q119
**Pregunta:** How to deploy a workload on Google Cloud Cloud Run?

**Claim:** Must end with a letter or a.

**Mejor evidencia** (gcp/GKE):

>  following instructions in other
> tutorials:Go to the Cloud LoggingLogs storagepage.Go to Logs storageSelecthello-world-cluster-bucketand clickDelete.Go to the LoggingLog routerpage.Go to Log routerSelecthello-world-cluster-sinkand clickDelete. [GCP > GKE > Create a cluster and deploy a workload in the Google Cloud console > What's next] Explore your cluster and workloadto learn about the some of the key workload settings and resources that
> you deployed.Try our more in-depthLearning path: Scalable apps.Learn how to get started with real life cluster administration in ourCluster administration overview.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 91. denso | mistral-7b-instruct — q004
**Pregunta:** What is AWS EKS and what are its main features?

**Claim:** Streamlined Contribution Process: Faster pull request processing.

**Mejor evidencia** (aws/EC2):

>  scale up or down to handle changes in requirements or spikes in popularity, reducing your need to forecast traffic\.
> 
> For more information about cloud computing, see [What is cloud computing?](https://aws.amazon.com/what-is-cloud-computing/) [AWS > EC2 > What is Amazon EC2?<a name="concepts"></a> > Features of Amazon EC2<a name="ec2-features"></a>] Amazon EC2 provides the following features:
> + Virtual computing environments, known as *instances*
> + Preconfigured templates for your instances, known as *Amazon Machine Images \(AMIs\)*, that package the bits you need for your server \(including the operating system and additional software\)
> + Various configurations of CPU, memory, storage, and networking capacity for your instances, known as *instance types*
> + Secure login information for you

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 92. hibrido | granite4.1-8b — q039
**Pregunta:** What are the available SKUs for Azure Virtual Machines?

**Claim:** discusses standard and basic SKUs for public IP addresses used by Azure Bastion, but this does not directly relate to VM SKUs.

**Mejor evidencia** (azure/AKS):

>  SKUs offered and their corresponding detailed capacity limits, seeGeneral purpose virtual machine sizes.
> To help determine best fit for your workload between Azure Files and Azure NetApp Files, review the information provided in the articleAzure Files and Azure NetApp Files comparison. [AZURE > AKS > Storage options for applications in Azure Kubernetes Service (AKS) > Volumes > Azure Disk] UseAzure Diskto create a KubernetesDataDiskresource. Disks types include:
> Premium SSDs (recommended for most workloads)Ultra disksStandard SSDsStandard HDDs
> TipFor most production and development workloads, use Premium SSDs.
> Because an Azure Disk is mounted asReadWriteOnce, it's only available to a single node. For storage volumes accessible by pods on multiple nodes simultaneously, use Azure Files.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 93. lexico | gemma4-e4b — q031
**Pregunta:** What regions support Azure Entra ID?

**Claim:** Microsoft Azure operated by 21Vianet.

**Mejor evidencia** (azure/Blob Storage):

> [AZURE > Blob Storage > Authorize access to blobs using Microsoft Entra ID] Azure Storage supports using Microsoft Entra ID to authorize requests to blob data. With Microsoft Entra ID, you can use Azure role-based access control (Azure RBAC) to grant permissions to a security principal, which may be a user, group, or application service principal. The security principal is authenticated by Microsoft Entra ID to return an OAuth 2.0 token. The token can then be used to authorize a request against the Blob service.
> 
> Authorization with Microsoft Entra ID is available for all general-purpose and Blob storage accounts in all public regions and national clouds. Only storage accounts created with the Azure Resource Manager deployment model support Microsoft Entra authorization.
> 
> [!INCLUDE [storage

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 94. hibrido | granite4.1-8b — q074
**Pregunta:** How to deploy a containerized application on AWS IAM?

**Claim:** On the Configure task and container definitions page, scroll down and choose Configure via JSON..

**Mejor evidencia** (aws/ECS):

> -ecs.md) and that your AWS user has either the permissions specified in the `AdministratorAccess` or the [Amazon ECS first\-run wizard permissions](security_iam_id-based-policy-examples.md#first-run-permissions) IAM policy example\. [AWS > ECS > Getting started with the classic console using Amazon EC2<a name="getting-started-ecs-ec2"></a> > Step 1: Register a task definition<a name="getting-started-ec2-task-def"></a>] A task definition is like a blueprint for your application\. Each time that you launch a task in Amazon ECS, you specify a task definition\. The service then knows which Docker image to use for containers, how many containers to use in the task, and the resource allocation for each container\. For more information about task definitions, see [Amazon ECS task definitions](tas

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 95. hibrido | granite4.1-8b — q180
**Pregunta:** What are the main differences between AWS CloudFormation and Azure Resource Manager templates?

**Claim:** To compare AWS CloudFormation and Azure Resource Manager templates, we would need documentation that describes ARM templates from Azure.

**Mejor evidencia** (aws/Lambda):

>  [sample-apps](./sample-apps) - Sample applications that demonstrate features and use cases for the AWS Lambda service and managed runtimes.
> - [templates](./templates) - AWS CloudFormation templates for creating functions and VPC network resources. [AWS > Lambda > License Summary] The sample code within this repo is made available under a modified MIT license. See the [LICENSE](./LICENSE) file.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 96. denso | granite4.1-8b — q064
**Pregunta:** How to set up AWS EKS for a production workload?

**Claim:** None of these sections specifically address the setup or configuration of AWS EKS for production workloads.

**Mejor evidencia** (azure/AKS):

> loads or testing existing workloads. TheProduction Economypreset is best for serving production traffic in a cost-conscious way if your workloads can tolerate interruptions. Noncritical features are off by default, and the preset values can be modified at any time. [AZURE > AKS > Best practices for cost optimization in Azure Kubernetes Service (AKS) > Prepare the application environment > Consider multitenancy] AKS offer flexibility in how you run multitenant clusters and isolate resources. For friendly multitenancy, you can share clusters and infrastructure across teams and business units throughlogical isolation. KubernetesNamespacesform the logical isolation boundary for workloads and resources. Sharing infrastructure reduces cluster management overhead while also improving resource uti

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 97. lexico | granite4.1-8b — q065
**Pregunta:** How to set up AWS RDS for a production workload?

**Claim:** Action: Choose an appropriate database engine (e.g., MySQL, PostgreSQL) based on your application requirements.

**Mejor evidencia** (aws/EC2):

>  that you can use, you can skip this task\.
> 
> **Important**  
> If you use an existing RDS database, make sure that it is in the same VPC as your EC2 instance so that you can use the automatic connection feature\. [AWS > EC2 > Task 1: Create an RDS database – *optional*<a name="option1-task1-create-rds-database"></a> > Steps to create an RDS database<a name="option1-task1-create-rds-database-steps"></a>] Use the following steps to create an RDS database\.
> 
> To view an animation of these steps, see [View an animation: Create an RDS database](#task1-create-rds-database-animation)\.
> 
> **RDS database configuration**
> 
> The steps in this task configure the RDS database as follows:
> + Engine type: MySQL
> + Template: Free tier
> + DB instance identifier: **tutorial\-database\-1**
> + DB instance class: `db.t3

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 98. hibrido | qwen3.5-9b — q161
**Pregunta:** When should you choose Azure Virtual Network over Azure Container Instances?

**Claim:** What the documentation does mention:.

**Mejor evidencia** (azure/Virtual Network):

> [AZURE > Virtual Network > Add, change, or delete a virtual network subnet] Azure virtual network subnets are essential containers where all Azure resources are deployed within a virtual network. This comprehensive guide explains how to add, change, or delete virtual network subnets using the Azure portal, Azure CLI, or Azure PowerShell, helping you effectively manage your network infrastructure.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 99. lexico | mistral-7b-instruct — q085
**Pregunta:** How to configure networking for Azure Virtual Network?

**Claim:** Configure service endpoints on the destination service..

**Mejor evidencia** (azure/Azure Functions):

> outbound calls to private endpoints](#private-endpoints). For more details on using private endpoints with the storage account for your function app, visit [restrict your storage account to a virtual network](#restrict-your-storage-account-to-a-virtual-network). [AZURE > Azure Functions > Azure Functions networking options > Inbound networking features > Service endpoints] Using service endpoints, you can restrict many Azure services to selected virtual network subnets to provide a higher level of security. Regional virtual network integration enables your function app to reach Azure services that are secured with service endpoints. This configuration is supported on all [plans](functions-scale.md#networking-features) that support virtual network integration. Follow these steps to access a

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 100. hibrido | granite4.1-8b — q198
**Pregunta:** How does the concept of availability zones differ across AWS, Azure, and GCP?

**Claim:** Management: Azure automatically distributes VMs within an availability set across different zones to balance fault domains and update domains..

**Mejor evidencia** (aws/EC2):

>  Availability Zones](#availability-zones-describe)
> + [Launch instances in an Availability Zone](#using-regions-availability-zones-launching)
> + [Migrate an instance to another Availability Zone](#migrating-instance-availability-zone) [AWS > EC2 > Regions and Zones<a name="using-regions-availability-zones"></a> > Availability Zones<a name="concepts-availability-zones"></a> > AZ IDs<a name="az-ids"></a>] To ensure that resources are distributed across the Availability Zones for a Region, we independently map Availability Zones to codes for each AWS account\. For example, the Availability Zone `us-east-1a` for your AWS account might not be the same physical location as `us-east-1a` for another AWS account\.
> 
> To coordinate Availability Zones across accounts, you must use the *AZ ID*, which is a

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 101. lexico | granite4.1-8b — q162
**Pregunta:** What are the differences between Azure Functions and Azure Virtual Machines?

**Claim:** Azure Virtual Machines: Can be deployed within any Azure virtual network, providing extensive networking options such as private IP addresses, subnets, and network security groups for robust isolation and connectivity..

**Mejor evidencia** (azure/Virtual Machines):

>  your needs, this decision tree should help with typical scenarios:
> 
> For a video that covers some high level differences for the different disk types, and some ways for determining what impacts your workload requirements, seeBlock storage options with Azure Disk Storage and Elastic SAN. [AZURE > Virtual Machines > Azure managed disk types > Ultra Disks] Azure Ultra Disks are the highest-performing storage option for Azure virtual machines (VMs). You can change the performance parameters of an Ultra Disk without having to restart your VMs. Ultra Disks are suited for data-intensive workloads such as SAP HANA, top-tier databases, and transaction-heavy workloads.
> Ultra Disks must be used as data disks and can only be created as empty disks. When using Ultra Disks as data disks, use Premium sol

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 102. denso | granite4.1-8b — q096
**Pregunta:** How to scale Azure Blob Storage horizontally?

**Claim:** *Reference*: [Scalability targets for standard storage accounts](../common/scalability-targets-standard-account.md) mentions replication options that contribute to scalability..

**Mejor evidencia** (azure/Blob Storage):

> ](../../../includes/storage-scalability-intro-include.md)]
> 
> The service-level agreement (SLA) for Azure Storage accounts is available at [SLA for Storage Accounts](https://azure.microsoft.com/support/legal/sla/storage/v1_5/). [AZURE > Blob Storage > Scalability and performance targets for Blob storage > Scale targets for Blob storage] [!INCLUDE [storage-blob-scale-targets](../../../includes/storage-blob-scale-targets.md)]
> 
> - [Performance and scalability checklist for Blob storage](storage-performance-checklist.md)
> - [Scalability targets for standard storage accounts](../common/scalability-targets-standard-account.md)
> - [Scalability targets for premium block blob storage accounts](scalability-targets-premium-block-blobs.md)
> - [Scalability targets for the Azure Storage resource provider](../

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 103. hibrido | mistral-7b-instruct — q038
**Pregunta:** What is Azure AKS and what are its main capabilities?

**Claim:** When you create an AKS cluster, Azure automatically creates and configures a control plane for you at no cost.

**Mejor evidencia** (azure/AKS):

> Facebook
> x.com
> LinkedIn
> Email
> 
> Copy Markdown
> Print [AZURE > AKS > What is Azure Kubernetes Service (AKS)? > Overview of AKS] AKS reduces the complexity and operational overhead of managing Kubernetes by shifting that responsibility to Azure. When you create an AKS cluster, Azure automatically creates and configures a control plane for you at no cost. The Azure platform manages the AKS control plane, which is responsible for the Kubernetes objects and worker nodes that you deploy to run your applications. Azure takes care of critical operations like health monitoring and maintenance, and you only pay for the AKS nodes that run your applications.
> 
> NoteAKS isCNCF-certifiedand is compliant with SOC, ISO, PCI DSS, and HIPAA. For more information, see theMicrosoft Azure compliance overview.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 104. lexico | mistral-7b-instruct — q176
**Pregunta:** Compare container orchestration options across AWS ECS, Azure Container Instances, and Google Cloud Run

**Claim:** Integration: All three services integrate with their respective container registries (Amazon ECR for AWS, Docker Hub for AWS, and Container Registry for GCP) ..

**Mejor evidencia** (aws/ECS):

>  the default, expand **Task Placement**, and then choose from the following options\.
> 
>     For more information, see [Amazon ECS task placement](task-placement.md)\.
>    + **AZ Balanced Spread** \- Distribute tasks across Availability Zones and [AWS > ECS > Running a standalone task using the Amazon ECS console<a name="ecs_run_task-v2"></a>]  across container instances in the Availability Zone\.
>    + **AZ Balanced BinPack** \- Distribute tasks across Availability Zones and across container instances with the least available memory\.
>    + **BinPack** \- Distribute tasks based on the least available amount of CPU or memory\.
>    + **One Task Per Host** \- Place, at most, one task from the service on each container instance\.
>    + **Custom** \- Define your own task placement strategy\. 
> 
>    If 

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 105. lexico | qwen3.5-9b — q142
**Pregunta:** What are the differences between AWS RDS and AWS VPC?

**Claim:** VPC configuration requirements for RDS.

**Mejor evidencia** (aws/EC2):

>  the VPC, you can disable ClassicLink for the VPC\. [AWS > EC2 > ClassicLink<a name="vpc-classiclink"></a> > ClassicLink basics<a name="classiclink-basics"></a> > Use other AWS services in your VPC with ClassicLink<a name="classiclink-other-services"></a>] Linked EC2\-Classic instances can access the following AWS services in the VPC: Amazon Redshift, Amazon ElastiCache, Elastic Load Balancing, and Amazon RDS\. However, instances in the VPC cannot access the AWS services provisioned by the EC2\-Classic platform using ClassicLink\.
> 
> If you use Elastic Load Balancing, you can register your linked EC2\-Classic instances with the load balancer\. You must create your load balancer in the ClassicLink\-enabled VPC and enable the Availability Zone in which the instance runs\. If you terminate the 

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 106. denso | gemma4-e4b — q149
**Pregunta:** Compare AWS EKS and AWS IAM in terms of use cases and features

**Claim:** Features: It is an AWS service that can be used with no additional charge ..

**Mejor evidencia** (aws/ECS):

> [AWS > ECS > Identity and Access Management for Amazon Elastic Container Service<a name="security-iam"></a>] AWS Identity and Access Management \(IAM\) is an AWS service that helps an administrator securely control access to AWS resources\. IAM administrators control who can be *authenticated* \(signed in\) and *authorized* \(have permissions\) to use Amazon ECS resources\. IAM is an AWS service that you can use with no additional charge\.
> 
> **Topics**
> + [Audience](#security_iam_audience)
> + [Authenticating with identities](#security_iam_authentication)
> + [Managing access using policies](#security_iam_access-manage)
> + [How Amazon Elastic Container Service works with IAM](security_iam_service-with-iam.md)
> + [Identity\-based policy examples for Amazon Elastic Container Service](security_iam_id

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 107. denso | mistral-7b-instruct — q103
**Pregunta:** How to deploy a workload on Google Cloud Compute Engine?

**Claim:** To deploy a workload on Google Cloud Compute Engine, follow these steps based on the provided context:.

**Mejor evidencia** (gcp/Compute Engine):

> .Using load balancing for highly available applications.Using autoscaling for highly scalable applications.
> 
> Send feedback [GCP > Compute Engine > Instance groupsStay organized with collectionsSave and categorize content based on your preferences. > Try it for yourself] If you're new to Google Cloud, create an account to evaluate how
>       Compute Engine performs in real-world
>       scenarios. New customers also get $300 in free credits to run, test, and
>       deploy workloads.
> Try Compute Engine free

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 108. denso | granite4.1-8b — q132
**Pregunta:** What are the differences between AWS Lambda and AWS EC2?

**Claim:** AWS Lambda: Serverless compute service where you run code without provisioning or managing servers.

**Mejor evidencia** (aws/EC2):

>  scale up or down to handle changes in requirements or spikes in popularity, reducing your need to forecast traffic\.
> 
> For more information about cloud computing, see [What is cloud computing?](https://aws.amazon.com/what-is-cloud-computing/) [AWS > EC2 > What is Amazon EC2?<a name="concepts"></a> > Features of Amazon EC2<a name="ec2-features"></a>] Amazon EC2 provides the following features:
> + Virtual computing environments, known as *instances*
> + Preconfigured templates for your instances, known as *Amazon Machine Images \(AMIs\)*, that package the bits you need for your server \(including the operating system and additional software\)
> + Various configurations of CPU, memory, storage, and networking capacity for your instances, known as *instance types*
> + Secure login information for you

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 109. lexico | mistral-7b-instruct — q178
**Pregunta:** How does auto-scaling work differently in AWS EC2 vs Azure Virtual Machine Scale Sets?

**Claim:** Azure VMSS (Flexible orchestration mode): In Flexible orchestration mode, you can choose whether your instances are spread across multiple zones or fault domains within a single region.

**Mejor evidencia** (azure/Virtual Machines):

>  all serialized. But, a single  deployment that requires a reboot only rolls out one zone at a time to reduce risk. In general, updates that require a reboot are avoided when possible, and Azure attempts to use Live Migration or provide customers control. [AZURE > Virtual Machines > Maintenance for virtual machines in Azure > Maintenance that requires a reboot > Availability considerations during scheduled maintenance > Virtual machine scale sets] Virtual machine scale sets inFlexibleorchestration mode are an Azure compute resource allow you to combine the scalability of virtual machine scale sets in Uniform orchestration mode with the regional availability guarantees of availability sets.
> With Flexible orchestration, you can choose whether your instances are spread across multiple zones, 

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 110. hibrido | gemma4-e4b — q093
**Pregunta:** How to implement backup and disaster recovery for Azure AKS?

**Claim:** Data cloning: Cloning existing volumes to create new persistent volumes with the same data..

**Mejor evidencia** (azure/AKS):

>  the following scenarios:
> Backup and restore: Create point-in-time backups of stateful application data and restore when
> needed.Data cloning: Clone existing volumes to create new persistent volumes with the same data.Disaster recovery: Quickly recover from data loss or corruption. [AZURE > AKS > Azure storage CSI driver and volume provisioning > Learn about volume snapshots > Create a volume snapshot] NoteBefore proceeding, ensure that the application isn't writing data to the source disk.
> For an example of this capability, create avolume snapshot classwith thekubectl applycommand:kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/azuredisk-csi-driver/master/deploy/example/snapshot/storageclass-azuredisk-snapshot.yamlThe output of the command resembles the following example

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 111. lexico | gemma4-e4b — q136
**Pregunta:** What are the differences between AWS CloudWatch and AWS EKS?

**Claim:** | Provides documentation using AsciiDoc markup, detailing advanced formatting, cross-referencing, and contribution processes  .

**Mejor evidencia** (aws/EKS):

> .aws.amazon.com/eks/latest/userguide/contribute.html) in the Amazon EKS User Guide.
> 
> We look forward to your contributions with the new GitHub experience. The improved platform makes it easier than ever to help us enhance the Amazon EKS documentation. [AWS > EKS > License Summary] The documentation is made available under the Creative Commons Attribution-ShareAlike 4.0 International License. See the LICENSE file.
> 
> The sample code within this documentation is made available under a modified MIT license. See the LICENSE-SAMPLECODE file.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 112. hibrido | qwen3.5-9b — q093
**Pregunta:** How to implement backup and disaster recovery for Azure AKS?

**Claim:** Important Note: Before proceeding, ensure that the application isn't writing data to the source disk.

**Mejor evidencia** (azure/AKS):

>  the following scenarios:
> Backup and restore: Create point-in-time backups of stateful application data and restore when
> needed.Data cloning: Clone existing volumes to create new persistent volumes with the same data.Disaster recovery: Quickly recover from data loss or corruption. [AZURE > AKS > Azure storage CSI driver and volume provisioning > Learn about volume snapshots > Create a volume snapshot] NoteBefore proceeding, ensure that the application isn't writing data to the source disk.
> For an example of this capability, create avolume snapshot classwith thekubectl applycommand:kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/azuredisk-csi-driver/master/deploy/example/snapshot/storageclass-azuredisk-snapshot.yamlThe output of the command resembles the following example

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 113. hibrido | qwen3.5-9b — q102
**Pregunta:** How to set up logging and monitoring for Google Cloud Cloud SQL?

**Claim:** Cloud SQL metrics and monitoring setup.

**Mejor evidencia** (gcp/GKE):

>  setting up
> production-ready clusters.
> Some networking features that can be enabled with Google Cloud tools, such as
> network policy enforcement, also require further configuration using Kubernetes
> APIs.
> You can learn much more about GKE networking inNetwork
> overview. [GCP > GKE > Cluster administration overviewStay organized with collectionsSave and categorize content based on your preferences. > Cluster observability] Another important part of cluster administration is configuring and usingobservabilitytooling to understand the health of your infrastructure and
> applications, and maintain application availability and reliability. By default,
> GKE clusters are configured to do the following:
> Sendsystem logs, audit logs, and application logsto Cloud Logging.Sendsystem metricsto Cloud Monitori

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 114. lexico | gemma4-e4b — q187
**Pregunta:** Compare the machine learning platforms: AWS SageMaker vs Azure Machine Learning vs Google Vertex AI

**Claim:** <br>- Reduce prediction latency.

**Mejor evidencia** (gcp/Compute Engine):

> You can provision clusters by using the accelerator-optimized machine family
> with any of the following open-source or Google Cloud products. [GCP > Compute Engine > About GPUs on Google CloudStay organized with collectionsSave and categorize content based on your preferences. > Multiple provisioning options > Vertex AI] Vertex AI is a fully-managed machine learning (ML) platform that you
> can use to train and deploy ML models and AI applications. In Vertex AI
> applications, you can use GPU-accelerated VMs to improve performance in the
> following ways:
> Use GPU-enabled VMsin custom training GKE worker pools.Useopen source LLM models from the Vertex AI Model Garden.Reducepredictionlatency.Improve performance ofVertex AI Workbenchnotebook code.Improve performance of aColab Enterprise runtime.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 115. denso | gemma4-e4b — q029
**Pregunta:** What is AWS ECS and what are its main features?

**Claim:** Deployment Options: ECS Anywhere: This allows users to manage on-premises container workloads using the Amazon ECS console and AWS CLI ..

**Mejor evidencia** (aws/ECS):

> [AWS > ECS > AWS services integrated with Amazon ECS<a name="ecs-integrations"></a>] Amazon ECS works with other AWS services to provide additional solutions for your business challenges\. This topic identifies services that either use Amazon ECS to add functionality, or services that Amazon ECS uses to perform tasks\.
> 
> **Topics**
> + [Using Amazon ECR with Amazon ECS](ecr-repositories.md)
> + [Creating Amazon ECS resources with AWS CloudFormation](creating-resources-with-cloudformation.md)
> + [Amazon Elastic Container Service on AWS Outposts](ecs-on-outposts.md)
> + [Use App Mesh with Amazon ECS](gs-app-mesh.md)
> + [AWS Deep Learning Containers on Amazon ECS](deep-learning-containers.md)

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 116. denso | mistral-7b-instruct — q090
**Pregunta:** How to configure networking for Azure Cosmos DB?

**Claim:** [Configure Azure Private Link for an Azure Cosmos DB account > Create a private endpoint by using the Azure portal] (Source: azure/Cosmos DB/Configure Azure Private Link for an Azure Cosmos DB account > Create a private endpoint by using the Azure portal).

**Mejor evidencia** (azure/Cosmos DB):

>  account are in different subscriptions, make sure that the subscription that has virtual network also hasMicrosoft.DocumentDBresource provider registered. To register a resource provider, seeAzure resource providers and types.
> Here are the directions for registering a subscription with a resource provider. [AZURE > Cosmos DB > Configure access to Azure Cosmos DB from virtual networks > Configure a service endpoint by using the Azure portal > Configure a service endpoint for a new Azure virtual network and subnet] From theAll resourcespane, find the Azure Cosmos DB account that you want to secure.SelectNetworkingfrom the settings menu, and choose to allow access fromSelected networks.To grant access to a new Azure virtual network, underVirtual networks, selectAdd new virtual network.Provid

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 117. denso | granite4.1-8b — q195
**Pregunta:** Compare the CI/CD pipeline offerings: AWS CodePipeline vs Azure DevOps vs Google Cloud Build

**Claim:** Service: Google Cloud Build (Cloud Build).

**Mejor evidencia** (gcp/GKE):

> loads, you can find
> GKE-specific best practices and guidelines for CI/CD inBest
> practices for continuous integration and delivery to
> GKE,
> as well as tutorials for setting up CI/CD pipelines with specific tools and
> products. [GCP > GKE > Deploying workloadsStay organized with collectionsSave and categorize content based on your preferences. > What's next] Learn more about tools for working with GKE:GKE in the Google Cloud consoleGoogle Cloud CLI overviewInstallkubectland configure cluster accessProvision GKE resources with TerraformLearn how to simplify deployment from your IDE with Cloud Code in
> ourDeploy and update from an
> IDEtutorial.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 118. denso | gemma4-e4b — q148
**Pregunta:** Compare AWS VPC and AWS ECS in terms of use cases and features

**Claim:** This comparison focuses on the distinct roles of Amazon VPC (networking foundation) and Amazon ECS (container orchestration) within the AWS ecosystem.

**Mejor evidencia** (aws/ECS):

> ancing Application Load Balancers\. This provides service discovery for a microservice architecture\.
> + [Amazon Elastic Container Service \- Reference Architecture: Service Discovery](https://github.com/awslabs/ecs-refarch-service-discovery/): This Amazon [AWS > ECS > Common use cases in Amazon ECS<a name="common_use_cases"></a> > Additional resources<a name="additional-resources"></a>]  ECS reference architecture provides service discovery to containers using CloudWatch Events, Lambda, and Route 53 private hosted zones\. 
> + [Metrics and traces collection from Amazon ECS using AWS Distro for OpenTelemetry with dynamic service discovery](http://aws.amazon.com/blogs/containers/metrics-and-traces-collection-from-amazon-ecs-using-aws-distro-for-opentelemetry-with-dynamic-service-discovery/): T

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 119. denso | qwen3.5-9b — q077
**Pregunta:** How to monitor AWS Lambda with CloudWatch?

**Claim:** EC2 monitoring with CloudWatch.

**Mejor evidencia** (aws/ECS):

> AWS provides various tools that you can use to monitor Amazon ECS\. You can configure some of these tools to do the monitoring for you, while some of the tools require manual intervention\. We recommend that you automate monitoring tasks as much as possible\. [AWS > ECS > Monitoring tools<a name="monitoring-automated-manual"></a> > Automated monitoring tools<a name="monitoring-automated_tools"></a>] You can use the following automated monitoring tools to watch Amazon ECS and report when something is wrong:
> + Amazon CloudWatch alarms – Watch a single metric over a time period that you specify, and perform one or more actions based on the value of the metric relative to a given threshold over a number of time periods\. The action is a notification sent to an Amazon Simple Notification Servic

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 120. lexico | qwen3.5-9b — q159
**Pregunta:** Compare the scaling options of Azure Container Instances vs Azure Virtual Machines

**Claim:** Azure Container Instances information is not present in the provided documentation.

**Mejor evidencia** (azure/Virtual Machines):

> )âï¸âï¸VHDX formatâï¸â
> For more information, seeTrusted launch. [AZURE > Virtual Machines > Support for Generation 2 VMs on Azure > Features and capabilities > Generation 1 vs. Generation 2 features] FeatureGeneration 1Generation 2BootPCATUEFIDisk controllersIDESCSIVM sizesAll VM sizesSee available sizes
> 
> CapabilityGeneration 1Generation 2OS disk > 2 TBââï¸Custom disk/image/swap OSâï¸âï¸Virtual machine scale set supportâï¸âï¸Azure Site Recoveryâï¸âï¸Backup/restoreâï¸âï¸Azure Compute Galleryâï¸âï¸Azure disk encryptionâï¸âï¸Server-side encryptionâï¸âï¸

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 121. hibrido | granite4.1-8b — q136
**Pregunta:** What are the differences between AWS CloudWatch and AWS EKS?

**Claim:** How to configure logging and monitoring in EKS using CloudWatch..

**Mejor evidencia** (aws/ECS):

> AWS provides various tools that you can use to monitor Amazon ECS\. You can configure some of these tools to do the monitoring for you, while some of the tools require manual intervention\. We recommend that you automate monitoring tasks as much as possible\. [AWS > ECS > Monitoring tools<a name="monitoring-automated-manual"></a> > Automated monitoring tools<a name="monitoring-automated_tools"></a>] You can use the following automated monitoring tools to watch Amazon ECS and report when something is wrong:
> + Amazon CloudWatch alarms – Watch a single metric over a time period that you specify, and perform one or more actions based on the value of the metric relative to a given threshold over a number of time periods\. The action is a notification sent to an Amazon Simple Notification Servic

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 122. denso | granite4.1-8b — q065
**Pregunta:** How to set up AWS RDS for a production workload?

**Claim:** By following these steps, you can successfully set up Amazon RDS for a production workload, ensuring scalability, reliability, and optimal performance.

**Mejor evidencia** (aws/ECS):

> \.
> 
> 1. For **Production listener port**, choose the listener port and protocol of the listener that you created in [Creating an Application Load Balancer](create-application-load-balancer.md) \(if applicable\), or choose **create new** to [AWS > ECS > Configuring a load balancer for the blue/green deployment type<a name="service-create-loadbalancer-bluegreen"></a>]  create a new listener and then enter a port number and choose a port protocol for **Production listener protocol**\.
> 
> 1. \(Optional\) Select **Test listener** if you want to configure a listener port and protocol on your load balancer to test updates to your service before routing traffic to your new task set\. Complete the following step:
> 
>    1. For **Test listener port**, choose the listener port and protocol of the listener 

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 123. lexico | qwen3.5-9b — q122
**Pregunta:** How to implement auto-scaling with Google Cloud Cloud Storage?

**Claim:** Networking and forwarding rules for VMs.

**Mejor evidencia** (gcp/Compute Engine):

> . They cannot be overridden by other routes,
> even if another route has a higher priority. This is because Google Cloud
> considers destination specificity before priority when selecting a route. For
> more information about subnet IP ranges, see thesubnets overview. [GCP > Compute Engine > Networking overview for VMsStay organized with collectionsSave and categorize content based on your preferences. > Forwarding rules] While routes govern traffic leaving an instance, forwarding rules direct traffic
> to a Google Cloud resource in a VPC network based on IP
> address, protocol, and port. Some forwarding rules direct traffic from outside
> of Google Cloud to a destination in the network; other rules direct
> traffic from inside the network.
> You can configure forwarding rules for your instances to implem

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 124. lexico | granite4.1-8b — q073
**Pregunta:** How to deploy a containerized application on AWS CloudWatch?

**Claim:** If you need guidance on deploying a containerized application using AWS ECS or similar services, please provide more context or clarify the specific service you intend to use for deployment.

**Mejor evidencia** (gcp/GKE):

>  bulletinsSecurity patchingPricing
> 
> This course features a combination of lectures, demos, and hands-on labs to help you explore and deploy solution elements—including infrastructure components like pods, containers, deployments, and services—along with networks and application services. [GCP > GKE > Google Kubernetes Engine documentation > Kubernetes Qwik Start lab] Learn how to deploy a containerized application with Kubernetes Engine in less than 30 minutes.
> 
> This tutorial demonstrates how to use graphical processinng units (GPUs) on GKE to run large language models (LLMs) for AI/ML inference.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 125. hibrido | mistral-7b-instruct — q110
**Pregunta:** How to configure IAM permissions for Google Cloud Compute Engine?

**Claim:** In the body of the request, provide the updated IAM policy from the previous step.

**Mejor evidencia** (gcp/Compute Engine):

> manage access to custom imageswith IAM.Learn more aboutService accounts.Learn more aboutCompute Engine IAM roles.Learn more about the permissions that are included inpredefined
> Compute Engine IAM roles.Learn how to create and managecustom roles.
> 
> Send feedback [GCP > Compute Engine > Managing access to Compute Engine resourcesStay organized with collectionsSave and categorize content based on your preferences. > Before you begin] Review theIAM overview.Read the Compute Engineaccess control overview.Familiarize yourself withIAM roles for Compute Engine.If you haven't already, set upauthentication.
>   Authentication verifies your identity for access to Google Cloud services and APIs. To run
>   code or samples from a local development environment, you can authenticate to
>   Compute Engine by sel

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 126. lexico | granite4.1-8b — q080
**Pregunta:** How to set up monitoring and alerts for Azure AKS?

**Claim:** These can be analyzed using the Metrics Explorer or used to create metric alerts.

**Mejor evidencia** (azure/AKS):

> Some services in Azure have a built-in monitoring dashboard in the Azure portal that provides a starting point for monitoring your service. These dashboards are calledinsights, and you can find them in theInsights Hubof Azure Monitor in the Azure portal. [AZURE > AKS > Monitor Azure Kubernetes Service (AKS) > AKS monitoring data: metrics, logs, integrations] AKS generates the same kinds of monitoring data as other Azure resources as described inMonitor data from Azure resources. For detailed information on the metrics and logs created by AKS, see theAKS monitoring data reference.
> Other Azure services and featurescollect other data and enable other analysis options as shown in the following diagram and table.
> 
> SourceDescriptionPlatform metricsPlatform metricsare automatically collected for 

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 127. lexico | mistral-7b-instruct — q071
**Pregunta:** How to monitor AWS DynamoDB with CloudWatch?

**Claim:** Next, go to the Amazon CloudWatch console.

**Mejor evidencia** (aws/EC2):

> [AWS > EC2 > Monitor the lifecycle of snapshots and AMIs<a name="dlm-monitor-lifecycle"></a>] You can use the following features to monitor the lifecycle of your snapshots and AMIs\.
> 
> **Topics**
> + [Console and AWS CLI](#monitor-console-cli)
> + [AWS CloudTrail](#monitor-lifecycle-cloudtrail)
> + [Monitor your policies using CloudWatch Events](monitor-cloudwatch-events.md)
> + [Monitor your policies using Amazon CloudWatch](monitor-dlm-cw-metrics.md)

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 128. lexico | granite4.1-8b — q120
**Pregunta:** How to deploy a workload on Google Cloud GKE?

**Claim:** After deployment, navigate to the Deployment details page for [CODE]..

**Mejor evidencia** (gcp/GKE):

>  have minimum availability. This occurs because Autopilot
> deletes and then re-creates the nodes. Wait a few minutes, then clickrefreshRefreshto update the page.Wait until the deployment completes and you see theDeployment detailspage. [GCP > GKE > Create a cluster and deploy a workload in the Google Cloud console > View a live demo in your browser] In the Google Cloud console, go to
> theDeployment detailspage forhello-world-app:In the Google Cloud console, go to the GKEWorkloadspage.Go to WorkloadsIn theNamecolumn, click the name of the workload you deployed,hello-world-app.In theEndpointscolumn, click the IP address, which is publicly available.GKE opens a new browser tab and sends a request to your
> app. Dismiss any secure-site warnings, and you should seeHello, world!in the new browser ta

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 129. denso | qwen3.5-9b — q187
**Pregunta:** Compare the machine learning platforms: AWS SageMaker vs Azure Machine Learning vs Google Vertex AI

**Claim:** Best Practice: Choose Vertex AI for teams focused on MLOps and rapid time-to-value.

**Mejor evidencia** (gcp/GKE):

> -tenant
> platform that serves diverse needs.Low-latency online serving: For developers building generative AI
> applications, GKE with the Inference Gateway provides the
> optimized routing and autoscaling needed to deliver a responsive user experience
> while controlling costs. [GCP > GKE > Introduction to AI/ML workloads on GKEStay organized with collectionsSave and categorize content based on your preferences. > Choose the right platform for your AI/ML workload] Google Cloud offers a spectrum of AI infrastructure products to support your
> ML journey, from fully managed to fully configurable. Choosing the right
> platform depends on your specific needs for control, flexibility, and level of
> management.
> Best practice:Choose GKE when you need deep control, portability, and the
> ability to build a cus

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 130. lexico | granite4.1-8b — q117
**Pregunta:** How to implement auto-scaling with Google Cloud Cloud Run?

**Claim:** for hybrid deployment considerations..

**Mejor evidencia** (gcp/GKE):

>  can therefore easily be migrated between runtimes.
> Here's anexamplecomparing the YAML files of a Kubernetes deployment and a
> Cloud Run service.Both GKE and Cloud Run integrate seamlessly withCloud LoggingandCloud Monitoring, providing you with a
> central [GCP > GKE > GKE and Cloud RunStay organized with collectionsSave and categorize content based on your preferences.] ized view on the Google Cloud console to observe application metrics
> regardless of their platform. You can also useservice-level objectives
> (SLO) monitoringon both
> platforms, and view a unified display of the SLOs on the Cloud Monitoring
> dashboard.You can implement continuous delivery to either GKE resources
> or Cloud Run services by usingCloud Deploy. Or, if
> you prefer, simultaneously deploy your application to both GKE
> and 

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 131. hibrido | qwen3.5-9b — q162
**Pregunta:** What are the differences between Azure Functions and Azure Virtual Machines?

**Claim:** What the context covers:.

**Mejor evidencia** (azure/Azure Functions):

> [AZURE > Azure Functions > Azure Functions networking options > Connecting to Azure Services through a virtual network > Virtual network triggers (non-HTTP) > App Service plan and App Service Environment with virtual network triggers] When your function app runs in either an App Service plan or an App Service Environment, you can write functions that are triggered by resources secured by a virtual network. For your functions to get triggered correctly, your app must be connected to a virtual network with access to the resource defined in the trigger connection.
> 
> For example, assume you want to configure Azure Cosmos DB to accept traffic only from a virtual network. In this case, you must deploy your function app in an App Service plan that provides virtual network integration with that vir

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 132. denso | granite4.1-8b — q120
**Pregunta:** How to deploy a workload on Google Cloud GKE?

**Claim:** Wait for the API and related services to be enabled..

**Mejor evidencia** (gcp/GKE):

>  gaming, and financial services. Examples of workloads you can run include:
> AI and ML operationsData processing at scaleScalableonline games platformsReliable applications under heavy load
> For case studies by industry and application, refer toGoogle Cloud customers. [GCP > GKE > GKE overviewStay organized with collectionsSave and categorize content based on your preferences. > How GKE works] A GKE environment consists ofnodes, which areCompute Engine
> virtual machines (VMs), that are grouped together to form acluster. You
> package your apps (also calledworkloads) into containers. You deploy sets
> of containers asPodsto your nodes. You use the Kubernetes API to interact
> with your workloads, including administering, scaling, and monitoring.
> Kubernetes clusters have a set of management nodes cal

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 133. lexico | granite4.1-8b — q065
**Pregunta:** How to set up AWS RDS for a production workload?

**Claim:** Citation: *"For DB instance class, leave the default, which is db.t3.micro."*.

**Mejor evidencia** (aws/EC2):

>  that you can use, you can skip this task\.
> 
> **Important**  
> If you use an existing RDS database, make sure that it is in the same VPC as your EC2 instance so that you can use the automatic connection feature\. [AWS > EC2 > Task 1: Create an RDS database – *optional*<a name="option1-task1-create-rds-database"></a> > Steps to create an RDS database<a name="option1-task1-create-rds-database-steps"></a>] Use the following steps to create an RDS database\.
> 
> To view an animation of these steps, see [View an animation: Create an RDS database](#task1-create-rds-database-animation)\.
> 
> **RDS database configuration**
> 
> The steps in this task configure the RDS database as follows:
> + Engine type: MySQL
> + Template: Free tier
> + DB instance identifier: **tutorial\-database\-1**
> + DB instance class: `db.t3

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 134. hibrido | qwen3.5-9b — q136
**Pregunta:** What are the differences between AWS CloudWatch and AWS EKS?

**Claim:** Any information about EKS-specific metrics or monitoring.

**Mejor evidencia** (aws/ECS):

>  [CloudWatch Pricing](https://aws.amazon.com/cloudwatch/pricing/)\. Amazon ECS also provides monitoring metrics that are provided at no additional cost\. For more information, see [Amazon ECS CloudWatch metrics](cloudwatch-metrics.md)\. [AWS > ECS > Amazon ECS CloudWatch Container Insights<a name="cloudwatch-container-insights"></a> > Container Insights considerations<a name="cloudwatch-container-insights-considerations"></a>] The following should be considered when using CloudWatch Container Insights\.
> + CloudWatch Container Insights metrics only reflect the resources with running tasks during the specified time range\. For example, if you have a cluster with one service in it but that service has no tasks in a `RUNNING` state, there will be no metrics sent to CloudWatch\. If you have two

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 135. hibrido | granite4.1-8b — q109
**Pregunta:** How to implement auto-scaling with Google Cloud Cloud Build?

**Claim:** Note: For precise implementation details specific to your environment, refer to the official Google Cloud Build documentation or consult with a cloud architect familiar with GCP services.

**Mejor evidencia** (aws/ECS):

>  *Amazon EC2 Auto Scaling User Guide*\.
> + You have a VPC and security group created to use\. For more information, see [Create a virtual private cloud](get-set-up-for-amazon-ecs.md#create-a-vpc)\. [AWS > ECS > Tutorial: Using cluster auto scaling with the AWS Management Console and the Amazon ECS console<a name="tutorial-cluster-auto-scaling-console"></a> > Step 1: Create an Amazon ECS cluster<a name="console-tutorial-cluster"></a>] Use the following steps to create an Amazon ECS cluster\. 
> 
> Amazon ECS creates an Amazon EC2 Auto Scaling launch template and Auto Scaling group on your behalf as part of the AWS CloudFormation stack\. 
> 
> 1. Open the console at [https://console\.aws\.amazon\.com/ecs/v2](https://console.aws.amazon.com/ecs/v2)\.
> 
> 1. From the navigation bar, select the Region to us

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 136. lexico | qwen3.5-9b — q109
**Pregunta:** How to implement auto-scaling with Google Cloud Cloud Build?

**Claim:** Connecting GitHub repositories to Cloud Build.

**Mejor evidencia** (gcp/Cloud Functions):

> .
> Cloud Storage does provide a free tier: review the pricing
> document for further details.For Artifact Registry pricing, see thePricingpage.Because the build process is subject to billing, your project must
> have aCloud Billing Accountattached to it. [GCP > Cloud Functions > Build process overview > View your build image logs] A key benefit of having the build image process in your user project is access
> to build logs. You can use gcloud CLI or the Google Cloud console
> to reach the logs, which are available throughCloud Logging.
> gcloudDeploy your function using thegcloud functions deploycommand.The URL of the logs is shown as part of the response in your terminal
> window. For example:Deployingfunction(maytakeawhile-upto2minutes)...⠹
> **ForCloudBuildStackdriverLogs**,visit:
> https://console.clo

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 137. lexico | granite4.1-8b — q065
**Pregunta:** How to set up AWS RDS for a production workload?

**Claim:** Citation: *"The security groups that are automatically created also respect least privilege as they only allow the mutual connection for this workload on the database port by creating a workload‑specific security group pair."*.

**Mejor evidencia** (aws/EC2):

>  that you can use, you can skip this task\.
> 
> **Important**  
> If you use an existing RDS database, make sure that it is in the same VPC as your EC2 instance so that you can use the automatic connection feature\. [AWS > EC2 > Task 1: Create an RDS database – *optional*<a name="option1-task1-create-rds-database"></a> > Steps to create an RDS database<a name="option1-task1-create-rds-database-steps"></a>] Use the following steps to create an RDS database\.
> 
> To view an animation of these steps, see [View an animation: Create an RDS database](#task1-create-rds-database-animation)\.
> 
> **RDS database configuration**
> 
> The steps in this task configure the RDS database as follows:
> + Engine type: MySQL
> + Template: Free tier
> + DB instance identifier: **tutorial\-database\-1**
> + DB instance class: `db.t3

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 138. denso | qwen3.5-9b — q080
**Pregunta:** How to set up monitoring and alerts for Azure AKS?

**Claim:** Measure two types of information: Calculation of a value: Makes calculations based on numeric columns (e.g., CPU percentage).

**Mejor evidencia** (azure/AKS):

>  on:
> The resource provider's knowledge of important signals and thresholds for monitoring the resource.Data that tells what customers commonly alert on for this resource.
> NoteRecommended alert rules are available for:Virtual machinesAzure Kubernetes Service (AKS) resourcesLog Analytics workspaces [AZURE > AKS > Monitor Azure Kubernetes Service (AKS) > Alerts > Configure Prometheus metrics-based alerts] Download and configuration requirements: Alert rules are available as downloadable ARM templates or Bicep files. Before configuring alerts, ensure the managed service for Prometheus is enabled on your cluster and an Azure Monitor workspace is properly linked to your AKS cluster.
> When youenable collection of the managed service for Prometheus metricsfor your cluster, you can download a collec

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 139. hibrido | granite4.1-8b — q192
**Pregunta:** Compare identity federation approaches across AWS, Azure, and GCP

**Claim:** Inference for AWS and Azure:.

**Mejor evidencia** (gcp/GKE):

>  provides a single hosted solution for managing external IdPs across Google Cloud products. Unlike Identity Service for GKE, Workforce Identity Federation doesn't install components in your cluster.For details and migration instructions, seeUse external identity providers to authenticate to GKE. [GCP > GKE > GKE release notesStay organized with collectionsSave and categorize content based on your preferences. > February 18, 2025] Change(2025-R07) Version updatesGKE cluster versions have been updated.New versions available for upgrades and new clusters.The following Kubernetes versions are now available for new clusters and for
> opt-in control plane upgrades and node upgrades for existing clusters. For more
> information on versioning and upgrades, seeGKE versioning and supportandUpgrades.Rapi

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 140. denso | granite4.1-8b — q088
**Pregunta:** How to deploy an application using Azure Blob Storage?

**Claim:** Location: The Azure region (e.g., [CODE], [CODE]) where your resources will reside..

**Mejor evidencia** (azure/Blob Storage):

> ) installed, you can create a storage account and run the sample code with just a few commands. You can run the project in your local development environment, or in a [DevContainer](https://code.visualstudio.com/docs/devcontainers/containers). [AZURE > Blob Storage > Quickstart: Azure Blob Storage client library for Java SE > Setting up > Initialize the Azure Developer CLI template and deploy resources] From an empty directory, follow these steps to initialize the `azd` template, provision Azure resources, and get started with the code:
> 
> - Clone the quickstart repository assets from GitHub and initialize the template locally:
> 
>     
> [CODE_BLOCK_8]
> 
> 
>     You'll be prompted for the following information:
> 
>     - **Environment name**: This value is used as a prefix for all Azure resources creat

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 141. lexico | mistral-7b-instruct — q191
**Pregunta:** What are the key differences between AWS S3 lifecycle policies and Azure Blob Storage tiering?

**Claim:** In Azure, these policies can be used to transition blobs to delete blob at the end of their lifecycle or to transition blobs between tiers based on their use patterns ..

**Mejor evidencia** (aws/EC2):

>  lifecycle of local snapshots using Amazon Data Lifecycle Manager\. For more information see, [Automate snapshots on an Outpost](#dlm)\.
> 
> **6\. Can I create, use, or delete local snapshots if my Outpost loses connectivity to its [AWS > EC2 > Amazon EBS local snapshots on Outposts<a name="snapshots-outposts"></a> > Frequently asked questions<a name="faq"></a>]  Region?**  
> No\. The Outpost must have connectivity with its Region as the Region provides the access, authorization, logging, and monitoring services that are critical for your snapshots' health\. If there is no connectivity, you can't create new local snapshots, create volumes or launch instances from existing local snapshots, or delete local snapshots\.
> 
> **7\. How quickly is Amazon S3 storage capacity made available after deleting

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 142. hibrido | granite4.1-8b — q103
**Pregunta:** How to deploy a workload on Google Cloud Compute Engine?

**Claim:** Ensure the instance's firewall rules allow necessary traffic (e.g., HTTP, HTTPS)..

**Mejor evidencia** (gcp/GKE):

>  enable the Kubernetes Engine API:
> Visit theKubernetes Engine pagein the Google Cloud console.Create or select a project.Wait for the API and related services to be enabled.
>     This can take several minutes.Verify that billing is enabled for your Google Cloud project. [GCP > GKE > Create a cluster and deploy a workload in the Google Cloud console > Before you begin > Required roles] Make sure that you have the following role or roles on the project:
>           
>           Compute Admin, Kubernetes Engine Admin, Service Account User
> Check for the rolesIn the Google Cloud console, go to theIAMpage.Go to IAMSelect the project.In thePrincipalcolumn, find all rows that identify you or a group that
>                 you're included in. To learn which groups you're included in, contact your
>         

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 143. denso | mistral-7b-instruct — q145
**Pregunta:** Compare the pricing models of AWS S3 vs AWS EKS

**Claim:** It is suitable for production workloads requiring 99.9-99.95% API server uptime, workloads requiring 24-month Long Term Support (LTS) Kubernetes version support, regulated environments requiring extended maintenance, and enterprise-grade or production workloads ..

**Mejor evidencia** (azure/AKS):

> Standard tier, and thePremium tier.
> SKU and tier relationship:
> Base SKU clusters: Can use any of the three pricing tiers (Free, Standard, or Premium).Automatic SKU clusters: Must use the Standard tier (automatically selected during cluster creation). [AZURE > AKS > Free, Standard, and Premium pricing tiers for Azure Kubernetes Service (AKS) cluster management > AKS pricing tiers comparison] The following table compares the Free, Standard, and Premium pricing tiers for AKS cluster management:
> TierWhen to useSupported cluster typesPricingFeature comparisonFreeâ¢ Development/testing environments.â¢ Learning and evaluation scenarios.â¢ Non-production workloads.â¢ Development clusters or small scale testing environments.â¢ Clusters with fewer than 10 nodes.â¢ Free cluster management.â¢ P

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 144. denso | granite4.1-8b — q089
**Pregunta:** How to configure networking for Azure Blob Storage?

**Claim:** Go to Networking settings..

**Mejor evidencia** (azure/Blob Storage):

>  subresource select `dfs`. Select Virtual network and select virtual network and Subnet. Select DNS. Select Yes for Integrate with private DNS. Select the Subscription and Resource Group for your private link DNS. Select Next, Next, and select Create. [AZURE > Blob Storage > Troubleshoot issues in BlobFuse > Common mount problems > Failed to initialize new pipeline [configuration error in azstorage [account name not provided]]] Make sure the configuration file has `azstorage` section in your configuration file.
> 
> The [BlobFuse base configuration file](https://github.com/Azure/azure-storage-fuse/blob/main/setup/baseConfig.yaml) contains a list of all settings and a brief explanation of each setting. Use the [sample file cache configuration file](https://github.com/Azure/azure-storage-fuse/bl

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 145. lexico | gemma4-e4b — q193
**Pregunta:** How do encryption at rest implementations differ between AWS KMS, Azure Key Vault, and GCP Cloud KMS?

**Claim:** Scope of Encryption: When an encrypted EBS volume is created and attached to a supported instance type, the following data types are encrypted: Data at rest inside the volume..

**Mejor evidencia** (aws/EC2):

>  AMI\. This is because EBS\-backed AMIs include snapshots of EBS volumes that can be encrypted as described\. For more information, see [Use encryption with EBS\-backed AMIs](AMIEncryption.md)\. [AWS > EC2 > Amazon EBS encryption<a name="EBSEncryption"></a> > Rotating AWS KMS keys<a name="kms-key-rotation"></a>] Cryptographic best practices discourage extensive reuse of encryption keys\. To create new cryptographic material for your KMS key, you can create new KMS key, and then change your applications or aliases to use the new KMS key\. Or, you can enable automatic key rotation for an existing KMS key\.
> 
> When you enable automatic key rotation for a KMS key, AWS KMS generates new cryptographic material for the KMS key every year\. AWS KMS saves all previous versions of the cryptographic ma

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 146. lexico | qwen3.5-9b — q119
**Pregunta:** How to deploy a workload on Google Cloud Cloud Run?

**Claim:** GKE cluster creation in Google Cloud console.

**Mejor evidencia** (gcp/GKE):

>  how to deploy a containerized application with Kubernetes Engine in less than 30 minutes.
> 
> This tutorial demonstrates how to use graphical processinng units (GPUs) on GKE to run large language models (LLMs) for AI/ML inference. [GCP > GKE > Google Kubernetes Engine documentation > Create a cluster and deploy a workload in the Google Cloud console] Learn how to create a Kubernetes cluster and deploy a 'hello world' web app in Google Cloud console.
> 
> This tutorial shows how to run a web application behind an external HTTP(S) load balancer by configuring the Ingress resource.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 147. denso | mistral-7b-instruct — q082
**Pregunta:** How to configure networking for Azure App Service?

**Claim:** In the Virtual Network Integration page, under Outbound traffic configuration, select Not configured..

**Mejor evidencia** (azure/Virtual Network):

>  network](../app-service/overview-vnet-integration.md?toc=%2fazure%2fvirtual-network%2ftoc.json)
> * [Set up Azure App Service access restrictions](../app-service/app-service-ip-restrictions.md) [AZURE > Virtual Network > Azure Virtual Network frequently asked questions (FAQ) > Azure services that connect to virtual networks > Can I deploy Cloud Services with web and worker roles (PaaS) in a virtual network?] Yes. You can (optionally) deploy Cloud Services role instances in virtual networks. To do so, you specify the virtual network name and the role/subnet mappings in the network configuration section of your service configuration. You don't need to update any of your binaries.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 148. lexico | mistral-7b-instruct — q082
**Pregunta:** How to configure networking for Azure App Service?

**Claim:** During the creation process, ensure to select the option for "Virtual network integration" and choose an existing virtual network or create a new one.

**Mejor evidencia** (azure/Azure Functions):

> [AZURE > Azure Functions > Azure Functions networking options > Connecting to Azure Services through a virtual network > Virtual network triggers (non-HTTP) > App Service plan and App Service Environment with virtual network triggers] When your function app runs in either an App Service plan or an App Service Environment, you can write functions that are triggered by resources secured by a virtual network. For your functions to get triggered correctly, your app must be connected to a virtual network with access to the resource defined in the trigger connection.
> 
> For example, assume you want to configure Azure Cosmos DB to accept traffic only from a virtual network. In this case, you must deploy your function app in an App Service plan that provides virtual network integration with that vir

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 149. denso | qwen3.5-9b — q127
**Pregunta:** When should you use AWS IAM instead of AWS EKS?

**Claim:** IAM is described as an AWS service that helps administrators securely control access to AWS resources, including authentication and authorization for AWS resources.

**Mejor evidencia** (aws/ECS):

> [AWS > ECS > How Amazon Elastic Container Service works with IAM<a name="security_iam_service-with-iam"></a>] Before you use IAM to manage access to Amazon ECS, learn what IAM features are available to use with Amazon ECS\.
> 
> 
> 
> 
> 
> 
> **IAM features you can use with Amazon Elastic Container Service**  
> 
> | IAM feature | Amazon ECS support | 
> | --- | --- | 
> |  [Identity\-based policies](#security_iam_service-with-iam-id-based-policies)  |    Yes  | 
> |  [Resource\-based policies](#security_iam_service-with-iam-resource-based-policies)  |    No   | 
> |  [Policy actions](#security_iam_service-with-iam-id-based-policies-actions)  |    Yes  | 
> |  [Policy resources](#security_iam_service-with-iam-id-based-policies-resources)  |    Partial  | 
> |  [Policy condition keys](#security_iam_service-with-iam-id-

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

## 150. lexico | granite4.1-8b — q107
**Pregunta:** How to set up logging and monitoring for Google Cloud BigQuery?

**Claim:** Enable Stackdriver Logging (Cloud Logging).

**Mejor evidencia** (gcp/GKE):

>  Scheduler, and Controller Manager.These logs are stored in Cloud Logging and can be queried in the Cloud Logging Log Explorer or Cloud Logging API. These logs can also be sent to Google Cloud Storage, BigQuery, or Pub/Sub using the Log Router. [GCP > GKE > GKE release notesStay organized with collectionsSave and categorize content based on your preferences. > November 22, 2022] IssueGKE version 1.21.14-gke.9500 has an issue where Pods in certain conditions might get stuck terminating indefinitely, due to a Linux kernel bug. The version has been removed and is no longer available for new clusters. If your node pools are running 1.21.14-gke.9500 and experience the issue, we recommenddowngrading the node poolto 1.21.14-gke.8500.

**Juicio humano:** ______ (correcto / incorrecto / dudoso)  |  **Comentario:** ______

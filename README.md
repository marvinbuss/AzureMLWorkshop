# Azure Machine Learning - Hands-on Workshop

The repo contains a collection of samples and challenge-based tranings for the Azure Machine Learning (AML) service.
While some of the samples are completly ready-to-go, the challenges present prepared notebooks with sections in them, which need to be completed by the participants.

While the repo also contains a sample solution for the challenges, participants are highly encouraged to try to find the solution on their own as much as possible. For this, the notebooks do contain links and hints.

## Getting started

## Prerequisites

The only prerequisite for this workshop is access to an Azure subscription with some budget on it. The overall Azure cost for the workshop should not exceed a few dollars if all the resources are being disposed again right afterwards.

Apart from that, everything will be executed in a standard web browser and no local installations etc. are needed.

### Provision Azure Machine Learning workspace

To run through the challenges, you first need to provision an Azure Machine Learning workspace within the Azure Portal (https://portal.azure.com). If you need help with this step, see [here](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-1st-experiment-sdk-setup) for a walkthrough.

### Provision Notebook VM

While you can easily run the workshop in a local jupyter environment, it is easier to use a prepared machine. The AML service offers the Notebook VMs for this.

1) To create a notebook VM, first access your AML workspace that created before from within the Azure portal.
2) Open the new Machine Learning studio UI experience by clicking on the blue "Launch now" button
3) In the newly opened browser window, click on "Compute", make sure you are in the "Notebook VM" pane and hit "+ New" and create a new VM. You can accept the default VM size.
4) The VM should be ready after just a few minutes.
5) Once the VM is provisioned, you will see the links to JupyterLab, Jupyter and R-Studio. Click on "Jupyter" to get started.

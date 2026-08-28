# Helm chart scope

This chart deploys only the MorphCloud-LLM **control-plane scaffold**. It does not provision the paper's AWS/GCP GPU nodes or recreate the missing vLLM data plane.

Build the local control-plane image first:

```bash
docker build -f docker/Dockerfile.control-plane -t morphcloud-llm-control-plane:local .
```

Then load/tag that image for your Kubernetes environment and install the chart:

```bash
helm install morphcloud ./helm
```

`values-aws.yaml` and `values-gcp.yaml` document paper-aligned hardware/storage parameters. They are reference configuration records, not cloud-provisioning templates. Cloud account identifiers, buckets, node groups, IAM, networking, and the GPU data-plane image must come from the original deployment environment.

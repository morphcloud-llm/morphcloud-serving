# GCP configuration
`../../helm/values-gcp.yaml` records paper-aligned GCP experiment parameters. It is a reference values file, not a complete GCP provisioner and is not automatically consumed by the minimal control-plane chart. Replace project-specific bucket, region/zone, IAM, node-group, and image settings using the original deployment artifacts before production use.

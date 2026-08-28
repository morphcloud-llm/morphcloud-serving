# AWS configuration
`../../helm/values-aws.yaml` records paper-aligned AWS experiment parameters. It is a reference values file, not a complete AWS provisioner and is not automatically consumed by the minimal control-plane chart. Replace account-specific bucket, region/AZ, IAM, node-group, and image settings using the original deployment artifacts before production use.

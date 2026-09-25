# Remote state uses a partial backend configuration so that no storage account
# names or keys are committed. Supply the rest at init time, one state per
# environment:
#
#   terraform init \
#     -backend-config="resource_group_name=$TFSTATE_RESOURCE_GROUP" \
#     -backend-config="storage_account_name=$TFSTATE_STORAGE_ACCOUNT" \
#     -backend-config="container_name=tfstate" \
#     -backend-config="key=client-research-agent/${ENVIRONMENT}.tfstate" \
#     -backend-config="use_azuread_auth=true"
#
# On AWS replace the block below with `backend "s3" {}` and pass
# bucket/key/region/use_lockfile=true the same way.
#
# Apply order per environment (prod first, because prod owns the catalog):
#   1. terraform apply -var-file=envs/<env>.tfvars
#   2. python infrastructure/unity_catalog/apply_ddl.py --environment <env> \
#        --agent-sp "$(terraform output -raw service_principal_application_id)"
#   3. terraform apply -var-file=envs/<env>.tfvars \
#        -var=create_vector_index=true -var=tables_provisioned=true
terraform {
  backend "azurerm" {}
}

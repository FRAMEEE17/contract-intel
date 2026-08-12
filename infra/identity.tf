# Keyless auth: grant the API's system-assigned identity data-plane RBAC on the
# Azure services, so the deployed app needs no keys. Key Vault holds the one
# remaining non-Azure secret (the Groq eval key), read via the same identity.

data "azurerm_client_config" "current" {}

data "azurerm_cognitive_account" "openai" {
  name                = "boomboxcontract2026"
  resource_group_name = "contract-intel-rg"
}

data "azurerm_search_service" "search" {
  name                = "smartsearchcontract"
  resource_group_name = "contract-intel-rg"
}

resource "azurerm_role_assignment" "api_openai" {
  scope                = data.azurerm_cognitive_account.openai.id
  role_definition_name = "Cognitive Services OpenAI User"
  principal_id         = azurerm_container_app.api.identity[0].principal_id
}

resource "azurerm_role_assignment" "api_search_data" {
  scope                = data.azurerm_search_service.search.id
  role_definition_name = "Search Index Data Contributor"
  principal_id         = azurerm_container_app.api.identity[0].principal_id
}

resource "azurerm_role_assignment" "api_search_service" {
  scope                = data.azurerm_search_service.search.id
  role_definition_name = "Search Service Contributor"
  principal_id         = azurerm_container_app.api.identity[0].principal_id
}

resource "azurerm_role_assignment" "api_acr_pull" {
  scope                = azurerm_container_registry.this.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_container_app.api.identity[0].principal_id
}

resource "azurerm_key_vault" "this" {
  name                      = "cintelkv${substr(md5(var.subscription_id), 0, 6)}"
  resource_group_name       = azurerm_resource_group.this.name
  location                  = azurerm_resource_group.this.location
  tenant_id                 = data.azurerm_client_config.current.tenant_id
  sku_name                  = "standard"
  rbac_authorization_enabled = true
}

resource "azurerm_role_assignment" "api_kv" {
  scope                = azurerm_key_vault.this.id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_container_app.api.identity[0].principal_id
}

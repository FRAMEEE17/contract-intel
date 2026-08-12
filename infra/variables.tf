variable "subscription_id" {
  type        = string
  description = "Azure subscription ID to deploy into."
}

variable "location" {
  type    = string
  default = "southeastasia"
}

variable "resource_group_name" {
  type    = string
  default = "contract-intel-aca-rg"
}

variable "acr_name" {
  type        = string
  description = "Globally-unique Azure Container Registry name (alphanumeric)."
  default     = "contractintelacr"
}

variable "image_tag" {
  type    = string
  default = "latest"
}

variable "azure_openai_endpoint" {
  type = string
}

variable "azure_openai_deployment" {
  type    = string
  default = "gpt-5-mini"
}

variable "azure_openai_api_version" {
  type    = string
  default = "2024-12-01-preview"
}

variable "azure_openai_api_key" {
  type      = string
  sensitive = true
}

variable "azure_search_endpoint" {
  type = string
}

variable "azure_search_key" {
  type      = string
  sensitive = true
}

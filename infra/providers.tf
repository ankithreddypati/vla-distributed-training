terraform {
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
    runpod = {
      source  = "decentralized-infrastructure/runpod"
      version = "1.0.1"
    }
    nebius = {
      source  = "terraform-provider.storage.eu-north1.nebius.cloud/nebius/nebius"
      version = ">= 0.5.55"
    }
  }
}

provider "digitalocean" {}

provider "runpod" {}

provider "nebius" {}
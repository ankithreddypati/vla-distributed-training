resource "digitalocean_droplet" "gpu" {
  image     = "gpu-h100x1-base"
  name      = "vla-gpu-node"
  region    = "nyc2"
  size      = "gpu-h200x1-141gb"
  ssh_keys  = var.ssh_key_ids
  monitoring = true
}
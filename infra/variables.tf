variable "ssh_key_ids" {
  type        = list(number)
  description = "SSH key IDs from DigitalOcean → Settings → Security (use numbers, not fingerprints)."
}

variable "node_count" {
  type        = number
  default     = 1   
  description = "Number of GPU nodes for distributed training"
}
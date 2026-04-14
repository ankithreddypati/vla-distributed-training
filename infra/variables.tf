variable "ssh_key_ids" {
  type        = list(number)
  description = "SSH key IDs from DigitalOcean → Settings → Security (use numbers, not fingerprints)."
}


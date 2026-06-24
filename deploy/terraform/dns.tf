# yapper.yuchia.dev -> the EC2, proxied through Cloudflare (edge terminates TLS; the origin
# stays plain HTTP via Traefik, matching the other *.yuchia.dev services).
resource "cloudflare_record" "yapper" {
  zone_id = var.cloudflare_zone_id
  name    = "yapper"
  type    = "A"
  content = var.ec2_public_ip
  proxied = true
  ttl     = 1 # 1 = automatic (required when proxied)
  comment = "Yapper commentary platform (k3s on hubstream)"
}

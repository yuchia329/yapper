# Durable artifact store on AWS S3 (no local-disk risk on the tiny box). Retention is a
# lifecycle rule — THIS is the "clean artifacts on a schedule" mechanism (no CronJob/code).
resource "aws_s3_bucket" "artifacts" {
  bucket = var.s3_bucket
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    id     = "expire-old-artifacts"
    status = "Enabled"
    filter {} # whole bucket: sources/* uploads + */recap_final.mp4 outputs

    expiration {
      days = var.artifact_retention_days
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

# The browser PUTs uploads + GETs playback straight to S3 via presigned URLs, so the bucket
# must allow CORS from the app origin.
resource "aws_s3_bucket_cors_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  cors_rule {
    allowed_methods = ["GET", "PUT", "HEAD"]
    allowed_origins = ["https://yapper.yuchia.dev"]
    allowed_headers = ["*"]
    expose_headers  = ["ETag"]
    max_age_seconds = 3000
  }
}

# A dedicated IAM user scoped to just this bucket; its key is injected into the app Secret.
# (Prefer an EC2 instance role if you wire one up later; static keys are fine for one app.)
resource "aws_iam_user" "yapper" {
  name = "yapper-app"
}

resource "aws_iam_access_key" "yapper" {
  user = aws_iam_user.yapper.name
}

resource "aws_iam_user_policy" "yapper_s3" {
  name = "yapper-s3-access"
  user = aws_iam_user.yapper.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = aws_s3_bucket.artifacts.arn
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
        Resource = "${aws_s3_bucket.artifacts.arn}/*"
      },
    ]
  })
}

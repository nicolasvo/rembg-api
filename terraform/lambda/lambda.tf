resource "aws_cloudwatch_log_group" "rembg" {
  name = "/aws/lambda/${aws_lambda_function.rembg.function_name}"

  retention_in_days = 7
}

resource "aws_lambda_function" "rembg" {
  function_name = "rembg"
  memory_size   = 6000
  timeout       = 300
  package_type  = "Image"
  architectures = ["x86_64"]
  image_uri     = "${data.terraform_remote_state.ecr.outputs.repository_url_rembg}:${var.image_tag_rembg}"

  role = aws_iam_role.lambda.arn
}

resource "aws_lambda_function_url" "rembg" {
  function_name      = aws_lambda_function.rembg.function_name
  authorization_type = "NONE"
}

data "aws_iam_policy_document" "lambda" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda" {
  name               = "rembg"
  assume_role_policy = data.aws_iam_policy_document.lambda.json
  managed_policy_arns = [
    "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
  ]
}

# ============================================
# K8s PredictScale - Dev Environment
# ============================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.12"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.25"
    }
  }

  backend "s3" {
    bucket = "predictscale-terraform-state"
    key    = "dev/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "k8s-predictscale"
      Environment = "dev"
      ManagedBy   = "terraform"
    }
  }
}

# ------------------------------------------------------------------
# VPC
# ------------------------------------------------------------------

module "vpc" {
  source = "../../modules/vpc"

  project_name = var.project_name
  vpc_cidr     = var.vpc_cidr

  tags = {
    Environment = "dev"
  }
}

# ------------------------------------------------------------------
# EKS
# ------------------------------------------------------------------

module "eks" {
  source = "../../modules/eks"

  project_name       = var.project_name
  private_subnet_ids = module.vpc.private_subnet_ids
  kubernetes_version = var.kubernetes_version

  node_instance_types = ["t3.medium"]
  node_desired_size   = 2
  node_min_size       = 1
  node_max_size       = 4

  tags = {
    Environment = "dev"
  }
}

# ------------------------------------------------------------------
# Kubernetes & Helm providers (configured after EKS is created)
# ------------------------------------------------------------------

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority)

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority)

    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
    }
  }
}

# ------------------------------------------------------------------
# Monitoring
# ------------------------------------------------------------------

module "monitoring" {
  source = "../../modules/monitoring"

  grafana_admin_password = var.grafana_admin_password

  depends_on = [module.eks]
}

"""Deploy the test agent to Vertex AI Agent Engine.

Usage:
    # Authenticate first
    gcloud auth login
    gcloud auth application-default login

    # Build the wheel (if not already built)
    cd policy-plugin && python -m build

    # Upload wheel to GCS (Agent Engine build needs HTTP access)
    gsutil cp dist/rubrik_agent_cloud_policy_plugin-*.whl gs://<bucket>/packages/
    gsutil acl ch -u AllUsers:R gs://<bucket>/packages/rubrik_agent_cloud_policy_plugin-*.whl

    # Deploy
    RUBRIK_WEBHOOK_URL=https://webhooks.your-dataplane.example.com \
    GOOGLE_API_KEY=<gemini-key> \
    python -m examples.test_agent.deploy --wheel-url https://storage.googleapis.com/<bucket>/packages/rubrik_agent_cloud_policy_plugin-0.1.1-py3-none-any.whl
"""

import argparse
import logging
import os

import vertexai
from vertexai import agent_engines

from examples.test_agent.agent import root_agent
from rubrik_agent_cloud_policy.adk import RubrikPolicyPlugin

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DEFAULT_PROJECT = "your-gcp-project"
DEFAULT_LOCATION = "us-central1"
DEFAULT_BUCKET = f"gs://{DEFAULT_PROJECT}-adk-staging"


def deploy(project: str, location: str, bucket: str, wheel_url: str) -> str:
    webhook_url = os.environ.get("RUBRIK_WEBHOOK_URL", "")
    api_key = os.environ.get("RUBRIK_API_KEY", "")
    google_api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not webhook_url:
        raise SystemExit(
            "RUBRIK_WEBHOOK_URL must be set. Example:\n"
            "  export RUBRIK_WEBHOOK_URL=https://webhook.your-dataplane.example.com"
        )

    vertexai.init(project=project, location=location, staging_bucket=bucket)

    adk_app = agent_engines.AdkApp(
        agent=root_agent,
        plugins=[RubrikPolicyPlugin()],
    )

    log.info("Deploying to %s/%s ...", project, location)

    env_vars = {"RUBRIK_WEBHOOK_URL": webhook_url}
    if api_key:
        env_vars["RUBRIK_API_KEY"] = api_key
    if google_api_key:
        env_vars["GOOGLE_API_KEY"] = google_api_key

    # Build requirements — use wheel URL if provided (package not yet on PyPI)
    requirements = [
        "google-cloud-aiplatform>=1.142.0",
        "google-genai>=1.51.0",
    ]
    if wheel_url:
        requirements.insert(0, f"rubrik-agent-cloud-policy-plugin[adk] @ {wheel_url}")
    else:
        requirements.insert(0, "rubrik-agent-cloud-policy-plugin[adk]")

    remote_app = agent_engines.create(
        agent_engine=adk_app,
        display_name="rubrik-agent-cloud-policy-test-agent",
        description="Test agent for validating rubrik-agent-cloud-policy-plugin tool blocking",
        requirements=requirements,
        env_vars=env_vars,
    )

    log.info("Deployed: %s", remote_app.resource_name)
    return remote_app.resource_name


def main():
    parser = argparse.ArgumentParser(description="Deploy test agent to Agent Engine")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--wheel-url", default="", help="HTTPS URL to the rubrik-agent-cloud-policy-plugin wheel (for pre-PyPI deploys)")
    args = parser.parse_args()

    resource_name = deploy(args.project, args.location, args.bucket, args.wheel_url)
    print(f"\nAgent deployed: {resource_name}")
    print(f"View at: https://console.cloud.google.com/vertex-ai/agents/agent-engines?project={args.project}")


if __name__ == "__main__":
    main()

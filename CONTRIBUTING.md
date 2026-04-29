# Contributing

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Setup

```bash
cd policy-plugin
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Running Tests

```bash
cd policy-plugin
pytest tests/ -v
```

## Building Locally

```bash
cd policy-plugin
pip install build
python -m build
```

This produces both a wheel (`.whl`) and source distribution (`.tar.gz`) in `dist/`.

## Releasing

Releases are fully automated via GitHub Actions. To publish a new version:

1. Bump `version` in `policy-plugin/pyproject.toml`
2. Merge to `main`

The workflow automatically:
- Detects the version change
- Builds the wheel and sdist
- Creates a git tag (`rubrik-agent-cloud-policy-plugin/v<version>`)
- Creates a GitHub Release with the build artifacts attached
- Publishes to PyPI

No manual tagging or uploading is needed.

### PyPI trusted publishing setup (one-time)

The PyPI publish step uses [trusted publishing](https://docs.pypi.org/trusted-publishers/), which requires:

1. On [pypi.org](https://pypi.org/manage/project/rubrik-agent-cloud-policy-plugin/settings/publishing/): add a trusted publisher
   with owner `rubrikinc`, repository `rubrik-agent-cloud`, workflow `policy-plugin-release.yml`, and environment `pypi`.
2. In the GitHub repo settings: create an environment named `pypi` (Settings > Environments > New environment).

### Deploying to Vertex AI Agent Engine

For deploying the plugin with an ADK agent to Agent Engine, see
[the deploy example](policy-plugin/examples/test_agent/deploy.py) and the
"Deploy to Vertex AI Agent Engine" section in the
[policy-plugin README](policy-plugin/README.md#deploy-to-vertex-ai-agent-engine).

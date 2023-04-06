"""Tasks of the project."""

# %% IMPORTS

from invoke import task
from invoke.context import Context

# %% CONFIGS

KFP_VERSION = "2.0.0b13"
KFP_PLATFORM = "platform-agnostic-emissary"

# %% TASKS

@task
def install(c: Context) -> None:
    """Install the project."""
    c.run("poetry install")


@task
def apply(c: Context, platform: str = KFP_PLATFORM, version: str = KFP_VERSION) -> None:
    """Apply Kubeflow manifests."""
    c.run(f'kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref={version}"')
    c.run('kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io')
    c.run(f'kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/{platform}?ref={version}"')
    print('Press CTRL+C to stop watching the pods.')
    c.run('kubectl get pods -n kubeflow --watch')


@task
def serve(c, port: int = 8443) -> None:
    """Serve Kubeflow Pipelines UI."""
    c.run(f'kubectl port-forward -n kubeflow svc/ml-pipeline-ui {port}:80')

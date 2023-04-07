"""Short demo of Kubeflow Pipelines from the online quickstart:
https://www.kubeflow.org/docs/components/pipelines/v2/quickstart/
"""
# %% IMPORTS

import kfp
import kfp.dsl

# %% CONFIGS

KFP_PIPELINE = "quickstart.yaml"
KFP_ENDPOINT = "http://localhost:8443/pipeline"

# %% SERVICES

client = kfp.client.Client(host=KFP_ENDPOINT)
compiler = kfp.compiler.Compiler()

# %% COMPONENTS

@kfp.dsl.component
def addition_component(num1: int, num2: int) -> int:
    return num1 + num2

@kfp.dsl.container_component
def say_hello(a: int, b: int, c: int):
    return kfp.dsl.ContainerSpec(image='alpine', command=['echo'], args=[f'Hello a={a}, b={b}, c={c}'])

# %% PIPELINES

@kfp.dsl.pipeline(name="addition-pipeline")
def my_pipeline(a: int, b: int, c: int):
    say_hello(a=a, b=b, c=c)
    add_task_1 = addition_component(num1=a, num2=b)
    add_task_2 = addition_component(num1=add_task_1.output, num2=c)

# %% COMPILATIONS

compiler.compile(pipeline_func=my_pipeline, package_path=KFP_PIPELINE)

# %% EXECUTIONS

# ! ERROR: does not work (probably a problem with the SDK)
# run = client.create_run_from_pipeline_func(
#     my_pipeline,
#     arguments={
#         'a': 1,
#         'b': 2
#     },
# )
run = client.create_run_from_pipeline_package(
    KFP_PIPELINE,
    arguments={
        "a": 1,
        "b": 2,
        "c": 3,
    },
)

# %% MONITORING

print(f"Run ID: {run.run_id}")

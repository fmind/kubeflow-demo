"""Predict the type of a wine from its measurements."""

# %% IMPORTS

import kfp
from kfp import dsl
from typing import NamedTuple, List

# %% CONFIGS

KFP_PIPELINE = "wine.yaml"
KFP_ENDPOINT = "http://localhost:8443/pipeline"

# %% SERVICES

client = kfp.client.Client(host=KFP_ENDPOINT)
compiler = kfp.compiler.Compiler()

# %% COMPONENTS

@dsl.component(packages_to_install=["pandas", "scikit-learn"])
def load_data_target(
    output_data: dsl.Output[dsl.Dataset],
    output_target: dsl.Output[dsl.Dataset],
) -> NamedTuple("Outputs", [("data_shape", List[int]), ("target_shape", List[int])]):
    """Load the wine datasets from sklearn."""
    from sklearn import datasets
    from typing import NamedTuple, List

    data, target = datasets.load_wine(return_X_y=True, as_frame=True)
    data.to_csv(output_data.path, index=False)
    target.to_csv(output_target.path, index=False)
    # this verbose structure is required, as explained in the documentation:
    # https://www.kubeflow.org/docs/components/pipelines/v2/author-a-pipeline/component-io/#python-components
    output = NamedTuple(
        "Outputs",
        [
            ("data_shape", List[int]),
            ("target_shape", List[int]),
        ],
    )
    # kubeflow doesn't support tuples -> cast to list
    return output(list(data.shape), list(target.shape))


@dsl.component(packages_to_install=["pandas"])
def validate_data(
    input_data: dsl.Input[dsl.Dataset], output_stats: dsl.Output[dsl.Dataset]
):
    """Validate the data and report its statistics."""
    import pandas as pd

    data = pd.read_csv(input_data.path)
    stats = data.describe(include="all")
    stats.to_csv(output_stats.path, index=False)
    assert data.shape[1] == 13, "Data should have 13 columns!"
    assert data.isnull().sum().sum() == 0, "Data contains null values!"


@dsl.component(packages_to_install=["pandas", "scikit-learn"])
def split_data_target(
    input_data: dsl.Input[dsl.Dataset],
    input_target: dsl.Input[dsl.Dataset],
    output_data_train: dsl.Output[dsl.Dataset],
    output_target_train: dsl.Output[dsl.Dataset],
    output_data_test: dsl.Output[dsl.Dataset],
    output_target_test: dsl.Output[dsl.Dataset],
    test_size: float,
    random_state: int,
):
    """Split the data and target into train and test sets."""
    import pandas as pd
    from sklearn import model_selection

    data = pd.read_csv(input_data.path)
    target = pd.read_csv(input_target.path)
    data_train, data_test, target_train, target_test = model_selection.train_test_split(
        data,
        target,
        test_size=test_size,
        random_state=random_state,
    )
    data_test.to_csv(output_data_test.path, index=False)
    data_train.to_csv(output_data_train.path, index=False)
    target_test.to_csv(output_target_test.path, index=False)
    target_train.to_csv(output_target_train.path, index=False)


@dsl.component(packages_to_install=["joblib", "scikit-learn"])
def create_model(
    output_model: dsl.Output[dsl.Model],
    n_neighbors: int,
):
    """Create a model (including a scaler and classifier)."""
    import joblib as jl
    from sklearn import pipeline, preprocessing, neighbors

    classifier = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    scaler = preprocessing.StandardScaler()
    steps = [
        ("scaler", scaler),
        ("classifier", classifier),
    ]
    model = pipeline.Pipeline(steps)
    jl.dump(model, output_model.path)


@dsl.component(packages_to_install=["joblib", "pandas", "scikit-learn"])
def train_model(
    input_model: dsl.Input[dsl.Model],
    input_data_train: dsl.Input[dsl.Dataset],
    input_target_train: dsl.Input[dsl.Dataset],
    output_model: dsl.Output[dsl.Model],
):
    """Fit the model to the training data."""
    import joblib as jl
    import pandas as pd

    model = jl.load(input_model.path)
    data_train = pd.read_csv(input_data_train.path)
    target_train = pd.read_csv(input_target_train.path)
    model.fit(data_train, target_train)
    jl.dump(model, output_model.path)


@dsl.component(packages_to_install=["pandas", "scikit-learn"])
def evaluate_model(
    input_model: dsl.Input[dsl.Model],
    input_data_test: dsl.Input[dsl.Dataset],
    input_target_test: dsl.Input[dsl.Dataset],
) -> float:
    """Evaluate the model on the test data."""
    import joblib as jl
    import pandas as pd

    model = jl.load(input_model.path)
    data_test = pd.read_csv(input_data_test.path)
    target_test = pd.read_csv(input_target_test.path)
    score = model.score(data_test, target_test)

    return float(score)


# %% PIPELINES

@dsl.pipeline(
    name="wine-classification",
    description="Predict the type of wine from its measurements.",
)
def wine_classification(
    n_neighbors: int,
    random_state: int,
    test_size: float,
) -> dsl.Model:
    """Define the pipeline for training a wine classifier."""
    load_data_target_task = load_data_target()
    validate_data_task = validate_data(
        input_data=load_data_target_task.outputs["output_data"],
    )
    split_data_target_task = split_data_target(
        input_data=load_data_target_task.outputs["output_data"],
        input_target=load_data_target_task.outputs["output_target"],
        test_size=test_size,
        random_state=random_state,
    )
    create_model_task = create_model(n_neighbors=n_neighbors)
    train_model_task = train_model(
        input_model=create_model_task.outputs["output_model"],
        input_data_train=split_data_target_task.outputs["output_data_train"],
        input_target_train=split_data_target_task.outputs["output_target_train"],
    )
    evaluate_model_task = evaluate_model(
        input_model=train_model_task.outputs["output_model"],
        input_data_test=split_data_target_task.outputs["output_data_test"],
        input_target_test=split_data_target_task.outputs["output_target_test"],
    )
    return train_model_task.outputs["output_model"]


# %% COMPILATIONS

compiler.compile(
    package_path=KFP_PIPELINE,
    pipeline_func=wine_classification,
)

# %% EXECUTIONS

run = client.create_run_from_pipeline_package(
    KFP_PIPELINE,
    arguments={
        "n_neighbors": 3,
        "test_size": 0.25,
        "random_state": 42,
    },
)

# %% MONITORING

print(f"Run ID: {run.run_id}")

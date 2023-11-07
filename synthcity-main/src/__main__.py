from synthcity.plugins import Plugins
from synthcity.benchmark import Benchmarks
from synthcity.plugins.core.dataloader import GenericDataLoader

from sklearn.datasets import load_diabetes


def main():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X["target"] = y

    loader = GenericDataLoader(X, target_column="target", sensitive_columns=["sex"])
    # loader.dataframe()

    syn_model = Plugins().get("dpgan")

    syn_model.fit(loader)

    syn_model.generate(count = 10).dataframe()

    print('done')

    score = Benchmarks.evaluate(
        [("CTGAN", "ctgan", {})],
        loader,
        synthetic_size=500,
        repeats=2,
        synthetic_reuse_if_exists=False
    )

    Benchmarks.print(score)

    score = Benchmarks.evaluate(
        [("DPCTGAN", "dpctgan", {"epsilon": 4})],
        loader,
        synthetic_size=500,
        repeats=2,
        synthetic_reuse_if_exists=False
    )

    Benchmarks.print(score)


if __name__ == '__main__':
    main()

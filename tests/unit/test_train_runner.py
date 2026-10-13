from exps.train_runner import _print_pairwise_cv_metrics


def test_print_pairwise_cv_metrics_shows_model_and_feature_baselines(capsys):
    _print_pairwise_cv_metrics(
        {
            "cross_validation": {
                "mean_accuracy": 0.8,
                "mean_precision": 0.9,
                "mean_prediction_coverage": 0.75,
                "feature_metrics": {
                    "title": {
                        "mean_accuracy": 0.7,
                        "mean_precision": 0.8,
                        "mean_prediction_coverage": 0.5,
                    },
                    "description": {
                        "mean_accuracy": 0.6,
                        "mean_precision": None,
                        "mean_prediction_coverage": 0.0,
                    },
                },
            }
        }
    )

    output = capsys.readouterr().out
    assert "aggregate model: 0.800 / 0.900 / 0.750" in output
    assert "title: 0.700 / 0.800 / 0.500" in output
    assert "description: 0.600 / n/a / 0.000" in output

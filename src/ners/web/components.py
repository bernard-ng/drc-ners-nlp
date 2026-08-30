"""Streamlit components for visual experiment work."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import streamlit as st

from ners.config import ResearchConfig
from ners.research import (
    MODEL_REGISTRY,
    ExperimentBuilder,
    ExperimentResult,
    ExperimentRunner,
    ExperimentStatus,
    ExperimentTracker,
)
from ners.web.dataset import DatasetSnapshot, inspect_dataset
from ners.web.view_models import (
    confusion_matrix_frame,
    experiment_results_frame,
    feature_importance_frame,
    metric_frame,
    model_availability_frame,
)


_EXPERIMENT_TYPES = ("baseline", "advanced", "feature_study", "tuning")


def render_overview(config: ResearchConfig) -> None:
    """Render model availability and comparable tracked results."""

    st.header("Experiment comparison")
    tracker = ExperimentTracker(config)
    results = tracker.list()
    frame = experiment_results_frame(results)

    completed = sum(result.status == ExperimentStatus.COMPLETED for result in results)
    failed = sum(result.status == ExperimentStatus.FAILED for result in results)
    available = len(MODEL_REGISTRY.names(available_only=True))
    best = tracker.best("macro_f1")
    summary_columns = st.columns(4)
    summary_columns[0].metric("Tracked runs", len(results))
    summary_columns[1].metric("Completed", completed)
    summary_columns[2].metric("Available models", f"{available}/{len(MODEL_REGISTRY)}")
    summary_columns[3].metric(
        "Best test macro-F1",
        _format_score(best.test_metrics.get("macro_f1") if best else None),
    )
    if failed:
        st.warning(f"{failed} tracked experiment(s) failed. Open Results for details.")

    if frame.is_empty():
        st.info("The tracker has no experiments yet. Use Run experiment to create one.")
    else:
        filter_columns = st.columns(2)
        status_options = frame.get_column("status").unique().sort().to_list()
        model_options = frame.get_column("model").unique().sort().to_list()
        statuses = filter_columns[0].multiselect(
            "Status", status_options, default=status_options, key="overview_status"
        )
        models = filter_columns[1].multiselect(
            "Model", model_options, default=model_options, key="overview_model"
        )
        filtered = frame.filter(
            pl.col("status").is_in(statuses) & pl.col("model").is_in(models)
        )
        st.dataframe(
            filtered.select(
                "name",
                "model",
                "status",
                "test_macro_f1",
                "test_balanced_accuracy",
                "sample_fraction",
                "train_rows",
                "test_rows",
                "duration_seconds",
                "started_at",
            ),
            hide_index=True,
            width="stretch",
            column_config=_result_column_config(),
        )

        chart = (
            filtered.filter(
                (pl.col("status") == ExperimentStatus.COMPLETED.value)
                & pl.col("test_macro_f1").is_not_null()
            )
            .with_columns(
                pl.concat_str(
                    pl.col("model"),
                    pl.lit(" · "),
                    pl.col("started_at").dt.strftime("%Y-%m-%d %H:%M"),
                ).alias("run")
            )
            .sort("test_macro_f1", descending=True)
            .head(15)
        )
        if not chart.is_empty():
            st.subheader("Test macro-F1")
            st.bar_chart(
                chart,
                x="run",
                y="test_macro_f1",
                x_label="Experiment",
                y_label="Macro-F1",
                horizontal=True,
                sort="-test_macro_f1",
                height=max(260, chart.height * 34),
            )

    st.subheader("Architecture availability")
    st.dataframe(
        model_availability_frame(MODEL_REGISTRY),
        hide_index=True,
        width="stretch",
        column_config={
            "available": st.column_config.CheckboxColumn("Available"),
            "reason": st.column_config.TextColumn("Local dependency status", width="large"),
        },
    )


def render_dataset(config: ResearchConfig) -> None:
    """Show counts, schema, and a bounded preview from names.csv."""

    st.header("Published dataset")
    st.caption(
        "Read-only inspection of names.csv. This view does not clean, normalize, split, "
        "edit, or export the corpus."
    )
    path = config.dataset_path
    if not path.is_file():
        st.error(f"Dataset not found at {path}")
        return

    try:
        snapshot = _cached_dataset_snapshot(str(path.resolve()), path.stat().st_mtime_ns)
    except (OSError, TypeError, ValueError) as error:
        st.error(f"Could not inspect the dataset: {error}")
        return

    summary_columns = st.columns(4)
    summary_columns[0].metric("Rows", f"{snapshot.row_count:,}")
    summary_columns[1].metric("File size", _format_bytes(snapshot.file_size_bytes))
    summary_columns[2].metric("Empty names", f"{snapshot.empty_name_count:,}")
    summary_columns[3].metric("Invalid labels", f"{snapshot.invalid_label_count:,}")
    if snapshot.invalid_label_count:
        st.error("Training will reject labels outside the published f/m contract.")
    elif snapshot.empty_name_count:
        st.warning(
            f"All labels are valid. Training will skip {snapshot.empty_name_count:,} "
            "empty names in memory."
        )
    else:
        st.success("The published name and label fields pass the interface checks.")

    left, right = st.columns((3, 2))
    with left:
        st.subheader("Sex distribution")
        st.bar_chart(
            snapshot.label_distribution,
            x="sex",
            y="rows",
            x_label="Sex",
            y_label="Rows",
            color="#4c78a8",
            height=280,
        )
        st.dataframe(
            snapshot.label_distribution,
            hide_index=True,
            width="stretch",
            column_config={
                "share": st.column_config.NumberColumn("Share", format="percent"),
            },
        )
    with right:
        st.subheader("CSV schema")
        schema_frame = pl.DataFrame(
            {
                "column": [name for name, _ in snapshot.schema],
                "type": [dtype for _, dtype in snapshot.schema],
                "model_field": [name in {"name", "sex"} for name, _ in snapshot.schema],
            }
        )
        st.dataframe(schema_frame, hide_index=True, width="stretch", height=320)

    if st.toggle("Show the first 20 published rows", value=False):
        st.caption("The table shows only name and sex, at most 20 rows.")
        st.dataframe(snapshot.preview, hide_index=True, width="stretch", height=440)


def render_experiment_launcher(config: ResearchConfig) -> None:
    """Render a single-template experiment launcher."""

    st.header("Run an experiment")
    st.caption(
        "Train one template with the deterministic name-group split used by CLI experiments. "
        "This page stays busy during training. Use the CLI suite for unattended multi-model "
        "runs."
    )
    if not config.dataset_path.is_file():
        st.error(f"Dataset not found at {config.dataset_path}")
        return
    if not config.templates_path.is_file():
        st.error(f"Experiment templates not found at {config.templates_path}")
        return

    builder = ExperimentBuilder(config)
    try:
        templates_by_type = {
            experiment_type: builder.templates(experiment_type)
            for experiment_type in _EXPERIMENT_TYPES
        }
    except (OSError, TypeError, ValueError) as error:
        st.error(f"Could not load experiment templates: {error}")
        return

    available_types = [key for key, templates in templates_by_type.items() if templates]
    if not available_types:
        st.info("config/research_templates.yaml has no experiment templates.")
        return

    experiment_type = st.selectbox(
        "Template group",
        available_types,
        format_func=lambda value: value.replace("_", " ").title(),
    )
    templates = templates_by_type[experiment_type]
    template_names = [str(template["name"]) for template in templates]
    selected_name = st.selectbox("Experiment template", template_names)
    template = next(template for template in templates if template["name"] == selected_name)
    model_type = str(template["model_type"])
    try:
        spec = MODEL_REGISTRY[model_type]
    except KeyError:
        st.error(f"Template references an unregistered model: {model_type}")
        return

    detail_columns = st.columns(3)
    detail_columns[0].metric("Architecture", model_type)
    detail_columns[1].metric("Family", spec.family.value)
    detail_columns[2].metric("Local status", "Available" if spec.available else "Unavailable")
    st.write(str(template.get("description", "")))
    if not spec.available:
        st.warning(spec.availability_error or "The model dependency is unavailable.")
    with st.expander("Template parameters"):
        st.json(template.get("model_params", {}))

    with st.form("run_experiment"):
        form_columns = st.columns(3)
        sample_fraction = form_columns[0].number_input(
            "Dataset fraction",
            min_value=0.0001,
            max_value=1.0,
            value=config.sample_fraction,
            step=0.01,
            format="%.4f",
        )
        test_fraction = form_columns[1].number_input(
            "Held-out fraction",
            min_value=0.001,
            max_value=0.999,
            value=config.test_fraction,
            step=0.01,
            format="%.3f",
        )
        chunk_size = form_columns[2].number_input(
            "CSV chunk size",
            min_value=1_000,
            max_value=1_000_000,
            value=config.chunk_size,
            step=10_000,
        )
        submitted = st.form_submit_button(
            "Run experiment",
            type="primary",
            disabled=not spec.available,
        )

    if not submitted:
        return

    run_config = ResearchConfig(
        dataset_path=config.dataset_path,
        templates_path=config.templates_path,
        models_dir=config.models_dir,
        outputs_dir=config.outputs_dir,
        chunk_size=int(chunk_size),
        sample_fraction=float(sample_fraction),
        test_fraction=float(test_fraction),
    )
    try:
        experiment = builder.from_template(
            template,
            sample_fraction=float(sample_fraction),
            test_fraction=float(test_fraction),
        )
        runner = ExperimentRunner(run_config)
        with st.spinner(f"Training {model_type}. Keep this page open until it completes."):
            experiment_id = runner.run(experiment)
        result = runner.tracker.get(experiment_id)
    except Exception as error:
        st.error(f"Experiment failed: {error}")
        return

    st.success(f"Experiment completed: {experiment_id}")
    if result is not None:
        render_result(result, compact=True)


def render_results(config: ResearchConfig) -> None:
    """Render the metrics and artifacts of one tracked experiment."""

    st.header("Result explorer")
    tracker = ExperimentTracker(config)
    results = tracker.list()
    if not results:
        st.info("No tracked results are available yet.")
        return

    by_id = {result.experiment_id: result for result in results}
    selected_id = st.selectbox(
        "Experiment",
        list(by_id),
        format_func=lambda value: _result_label(by_id[value]),
    )
    render_result(by_id[selected_id])


def render_result(result: ExperimentResult, *, compact: bool = False) -> None:
    """Render one experiment result consistently after a run or from history."""

    if not compact:
        st.subheader(result.config.name)
    status_columns = st.columns(4)
    status_columns[0].metric("Status", result.status.value.title())
    status_columns[1].metric("Model", result.config.model_type)
    status_columns[2].metric("Train rows", f"{result.train_size:,}")
    status_columns[3].metric("Test rows", f"{result.test_size:,}")

    if result.error_message:
        st.error(result.error_message)
    metrics = metric_frame(result)
    if not metrics.is_empty():
        st.subheader("Metrics")
        st.dataframe(
            metrics,
            hide_index=True,
            width="stretch",
            column_config={
                column: st.column_config.NumberColumn(
                    column.replace("_", " ").title(), format="%.4f"
                )
                for column in ("train", "test", "cross_validation")
            },
        )

    confusion = confusion_matrix_frame(result)
    importance = feature_importance_frame(result)
    if not confusion.is_empty() or not importance.is_empty():
        chart_columns = st.columns(2)
        with chart_columns[0]:
            if not confusion.is_empty():
                st.subheader("Confusion matrix")
                st.bar_chart(
                    confusion,
                    x="actual",
                    y="rows",
                    color="predicted",
                    x_label="Actual sex",
                    y_label="Rows",
                    stack=False,
                    height=340,
                )
        with chart_columns[1]:
            if not importance.is_empty():
                st.subheader(_feature_score_title(result))
                st.caption(_feature_score_note(result))
                st.bar_chart(
                    importance,
                    x="feature",
                    y="importance",
                    horizontal=True,
                    sort="-importance",
                    height=340,
                )

    if result.class_distribution:
        distribution = pl.DataFrame(
            {
                "sex": list(result.class_distribution),
                "rows": list(result.class_distribution.values()),
            }
        )
        st.subheader("Training sex distribution")
        st.bar_chart(distribution, x="sex", y="rows", height=240)

    with st.expander("Reproducibility details"):
        st.json(result.config.to_dict())
        st.write(f"Experiment ID: `{result.experiment_id}`")
        st.write(f"Started: {result.start_time.isoformat()}")
        st.write(f"Ended: {result.end_time.isoformat() if result.end_time else 'not recorded'}")
        st.write(f"Model artifact: `{result.model_path or 'not available'}`")


@st.cache_data(show_spinner="Scanning names.csv with Polars...")
def _cached_dataset_snapshot(path: str, _modified_ns: int) -> DatasetSnapshot:
    return inspect_dataset(Path(path), preview_rows=20)


def _format_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _format_bytes(size: int) -> str:
    value = float(size)
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _result_label(result: ExperimentResult) -> str:
    started = result.start_time.strftime("%Y-%m-%d %H:%M")
    return f"{result.config.name}, {started}, {result.status.value}"


def _feature_score_title(result: ExperimentResult) -> str:
    if result.config.model_type == "logistic_regression":
        return "Largest absolute coefficients"
    return "Largest estimator feature scores"


def _feature_score_note(result: ExperimentResult) -> str:
    if result.config.model_type == "logistic_regression":
        return "Absolute coefficient magnitude. The stored value does not show direction."
    return "The estimator defines this score. Do not compare it across model families."


def _result_column_config() -> dict[str, Any]:
    return {
        "test_macro_f1": st.column_config.NumberColumn("Test macro-F1", format="%.4f"),
        "test_balanced_accuracy": st.column_config.NumberColumn(
            "Test balanced accuracy", format="%.4f"
        ),
        "sample_fraction": st.column_config.NumberColumn("Dataset fraction", format="percent"),
        "duration_seconds": st.column_config.NumberColumn("Duration (s)", format="%.1f"),
        "started_at": st.column_config.DatetimeColumn("Started", format="YYYY-MM-DD HH:mm"),
    }

"""Main table sort settings menu for dashboard v4."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView

from slurminator.config import DashboardTableSortSettings


class TableSortFormScreen(ModalScreen[None]):
    """Edit runtime table sort preferences for the current dashboard session."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    _METRICS = ("primary", "secondary")
    _VALUES = ("current", "best")
    _DIRECTIONS = ("auto", "asc", "desc")

    def __init__(self) -> None:
        super().__init__()
        self.metric = "primary"
        self.value = "current"
        self.direction = "auto"
        self.preserve_state_groups = True
        self.preserve_dataset_groups = False

    def on_mount(self) -> None:
        """Focus the list after loading current app settings."""
        current = _current_table_sort(getattr(self.app, "table_sort", None))
        self.metric = current.metric
        self.value = current.value
        self.direction = current.direction
        self.preserve_state_groups = current.preserve_state_groups
        self.preserve_dataset_groups = current.preserve_dataset_groups
        self._refresh_labels()
        self.query_one("#table-sort-options", ListView).focus()

    def compose(self) -> ComposeResult:
        """Render the table sort controls as a keyboard-friendly list."""
        yield Container(
            Label("Table sort", id="table-sort-title"),
            Label(
                "Up/down selects; Enter changes a setting. YAML controls the startup default.",
                id="table-sort-message",
            ),
            ListView(
                ListItem(Label(self._metric_label()), id="table-sort-metric"),
                ListItem(Label(self._value_label()), id="table-sort-value"),
                ListItem(Label(self._direction_label()), id="table-sort-direction"),
                ListItem(Label(self._state_group_label()), id="table-sort-state-groups"),
                ListItem(Label(self._dataset_group_label()), id="table-sort-dataset-groups"),
                ListItem(Label("Apply sort"), id="apply-table-sort"),
                ListItem(Label("Return"), id="return-table-sort"),
                id="table-sort-options",
            ),
            id="table-sort-form",
        )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Cycle selected values or apply the menu."""
        item_id = event.item.id
        if item_id == "table-sort-metric":
            self.metric = _next_value(self.metric, self._METRICS)
        elif item_id == "table-sort-value":
            self.value = _next_value(self.value, self._VALUES)
        elif item_id == "table-sort-direction":
            self.direction = _next_value(self.direction, self._DIRECTIONS)
        elif item_id == "table-sort-state-groups":
            self.preserve_state_groups = not self.preserve_state_groups
        elif item_id == "table-sort-dataset-groups":
            self.preserve_dataset_groups = not self.preserve_dataset_groups
        elif item_id == "apply-table-sort":
            self._apply_sort()
            return
        elif item_id == "return-table-sort":
            self.app.pop_screen()
            return
        self._refresh_labels()

    def _apply_sort(self) -> None:
        settings = DashboardTableSortSettings(
            metric=self.metric,
            value=self.value,
            direction=self.direction,
            preserve_state_groups=self.preserve_state_groups,
            preserve_dataset_groups=self.preserve_dataset_groups,
        )
        apply_table_sort = getattr(self.app, "apply_table_sort", None)
        if callable(apply_table_sort):
            apply_table_sort(settings)
        else:
            self.app.table_sort = settings
        self.app.pop_screen()

    def _refresh_labels(self) -> None:
        self._set_item_label("table-sort-metric", self._metric_label())
        self._set_item_label("table-sort-value", self._value_label())
        self._set_item_label("table-sort-direction", self._direction_label())
        self._set_item_label("table-sort-state-groups", self._state_group_label())
        self._set_item_label("table-sort-dataset-groups", self._dataset_group_label())

    def _set_item_label(self, item_id: str, label: str) -> None:
        self.query_one(f"#{item_id}", ListItem).query_one(Label).update(label)

    def _metric_label(self) -> str:
        return f"Metric: {self.metric}"

    def _value_label(self) -> str:
        return f"Value: {self.value}"

    def _direction_label(self) -> str:
        return f"Direction: {self.direction}"

    def _state_group_label(self) -> str:
        return _toggle_label("State groups", self.preserve_state_groups)

    def _dataset_group_label(self) -> str:
        return _toggle_label("Dataset groups", self.preserve_dataset_groups)


def _current_table_sort(settings: object | None) -> DashboardTableSortSettings:
    if isinstance(settings, DashboardTableSortSettings):
        return settings
    return DashboardTableSortSettings(
        metric=_choice_attr(settings, "metric", "primary", {"primary", "secondary"}),
        value=_choice_attr(settings, "value", "current", {"current", "best"}),
        direction=_choice_attr(settings, "direction", "auto", {"auto", "asc", "desc"}),
        preserve_state_groups=_bool_attr(settings, "preserve_state_groups", True),
        preserve_dataset_groups=_bool_attr(settings, "preserve_dataset_groups", False),
    )


def _next_value(value: str, choices: tuple[str, ...]) -> str:
    try:
        index = choices.index(value)
    except ValueError:
        return choices[0]
    return choices[(index + 1) % len(choices)]


def _choice_attr(settings: object | None, name: str, default: str, choices: set[str]) -> str:
    value = getattr(settings, name, default)
    text = str(value).strip().lower() if value is not None else default
    return text if text in choices else default


def _bool_attr(settings: object | None, name: str, default: bool) -> bool:
    value = getattr(settings, name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _toggle_label(label: str, enabled: bool) -> str:
    return f"{label}: {'on' if enabled else 'off'}"


__all__ = ["TableSortFormScreen"]

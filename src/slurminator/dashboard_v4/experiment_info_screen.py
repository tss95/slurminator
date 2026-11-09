"""Experiment-list context overlay for dashboard v4."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static


class ExperimentInfoScreen(ModalScreen[None]):
    """Show the experiment title and a scrollable plain-text description."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    DEFAULT_CSS = """
    ExperimentInfoScreen {
        align: center middle;
    }

    #experiment-info {
        width: 80%;
        max-width: 88;
        height: 70%;
        min-height: 10;
        max-height: 28;
        border: solid $accent;
        background: $surface;
        padding: 1 2;
    }

    #experiment-info-title {
        width: 100%;
        text-style: bold;
        margin-bottom: 1;
    }

    #experiment-info-scroll {
        height: 1fr;
        min-height: 1;
    }

    #experiment-info-description {
        width: 100%;
    }

    #close-experiment-info {
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        """Render optional context with an explicit message when it is absent."""
        info = self.app.get_experiment_info()
        yield Container(
            VerticalScroll(
                Label(info.name or "Experiment info", markup=False, id="experiment-info-title"),
                Static(
                    info.description or "No experiment description provided.",
                    markup=False,
                    id="experiment-info-description",
                ),
                id="experiment-info-scroll",
            ),
            Button("Close", id="close-experiment-info"),
            id="experiment-info",
        )

    def on_mount(self) -> None:
        """Focus the description so arrow and page keys can scroll it."""
        self.query_one("#experiment-info-scroll", VerticalScroll).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Return to the global menu when Close is pressed."""
        if event.button.id == "close-experiment-info":
            self.app.pop_screen()

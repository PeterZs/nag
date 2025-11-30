from typing import Any, Dict, Set
from tools.event.event import Event, EventArgs
from dataclasses import dataclass, field


@dataclass
class EpochChangedEventArgs(EventArgs):
    """Event arguments for the epoch changed event."""

    epoch: int = field(default=0)
    """The new epoch."""

    old_epoch: int = field(default=0)
    """The old epoch."""

    memo: Set["EpochStateMixin"] = field(default_factory=set)
    """Memo for the event. Marks already visited objects."""


class EpochStateMixin():
    """A mixin for objects that have an epoch."""

    _epoch: int
    """The current epoch."""

    _epoch_changed: Event[EpochChangedEventArgs]
    """Event that is triggered when the epoch changes."""

    @property
    def epoch(self) -> int:
        """The current epoch."""
        return self._epoch

    @epoch.setter
    def epoch(self, value: int) -> None:
        if value != self._epoch:
            self._on_epoch_changed({}, EpochChangedEventArgs(
                epoch=value, old_epoch=self._epoch))

    def __init__(self, *args, epoch: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._epoch = epoch
        self._epoch_changed = Event[EpochChangedEventArgs](self)

    def _on_epoch_changed(self, ctx: Dict[str, Any], args: EpochChangedEventArgs) -> None:
        """Called when the epoch changes."""
        if self in args.memo:
            return
        args.memo.add(self)
        self._epoch = args.epoch
        self.on_epoch_change(args.epoch)
        self._epoch_changed.notify(args)

    def on_epoch_change(self, epoch: int) -> None:
        """Called when the epoch changes."""
        pass

    def notify_on_epoch_change(self, listener: "EpochStateMixin") -> None:
        """Notify the listener when the epoch changes."""
        self._epoch_changed.attach(listener._on_epoch_changed)

    def remove_on_epoch_change(self, listener: "EpochStateMixin") -> None:
        """Remove the listener from the epoch changed event."""
        self._epoch_changed.remove(listener._on_epoch_changed)

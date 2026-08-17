"""Exchange trading rules: tick size, lot size and minimum notional.

Prices and quantities are quantised with :class:`~decimal.Decimal`, not float.
A venue's rules are exact decimal constraints — a 0.01 tick, a 0.001 lot — and
binary floating point cannot represent either, so rounding in float leaves
values that look right and are rejected by the venue. Quantisation happens once,
at the order boundary.

Indicator and P&L arithmetic stays in float deliberately. Decimal there would
cost speed for no benefit: nobody rejects a moving average for being 1e-17 off,
and the exactness that matters is in what gets submitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_HALF_UP, ROUND_UP, Decimal, InvalidOperation


def _decimal(value: float | str | Decimal) -> Decimal:
    """Convert to Decimal via ``str`` so 0.1 stays 0.1 and not 0.1000...055."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


@dataclass(frozen=True)
class InstrumentSpec:
    """The venue's rules for one instrument.

    tick_size: prices must be a whole multiple of this.
    lot_size: quantities must be a whole multiple of this (the step).
    min_quantity: smallest order the venue accepts.
    min_notional: smallest order value the venue accepts.
    max_quantity: largest single order, if the venue caps it.
    """

    symbol: str = ""
    tick_size: Decimal = Decimal("0.01")
    lot_size: Decimal = Decimal("0.000001")
    min_quantity: Decimal = Decimal("0")
    min_notional: Decimal = Decimal("0")
    max_quantity: Decimal | None = None

    def __post_init__(self) -> None:
        if self.tick_size <= 0:
            raise ValueError("tick_size must be positive")
        if self.lot_size <= 0:
            raise ValueError("lot_size must be positive")
        if self.min_quantity < 0 or self.min_notional < 0:
            raise ValueError("minimums cannot be negative")
        if self.max_quantity is not None and self.max_quantity <= 0:
            raise ValueError("max_quantity must be positive when set")

    @classmethod
    def create(
        cls,
        symbol: str = "",
        tick_size: float | str = "0.01",
        lot_size: float | str = "0.000001",
        min_quantity: float | str = "0",
        min_notional: float | str = "0",
        max_quantity: float | str | None = None,
    ) -> "InstrumentSpec":
        """Build a spec from plain numbers, converting exactly."""
        return cls(
            symbol=symbol,
            tick_size=_decimal(tick_size),
            lot_size=_decimal(lot_size),
            min_quantity=_decimal(min_quantity),
            min_notional=_decimal(min_notional),
            max_quantity=None if max_quantity is None else _decimal(max_quantity),
        )

    # -- quantisation --------------------------------------------------------

    def round_price(self, price: float | Decimal) -> Decimal:
        """Snap a price to the nearest tick."""
        value = _decimal(price)
        try:
            ticks = (value / self.tick_size).quantize(Decimal(1), rounding=ROUND_HALF_UP)
        except InvalidOperation as exc:  # pragma: no cover - absurd inputs only
            raise ValueError(f"cannot quantise price {price!r}") from exc
        return ticks * self.tick_size

    def round_price_down(self, price: float | Decimal) -> Decimal:
        """Snap a price down to a tick, so a clamped buy limit stays under it."""
        value = _decimal(price)
        return (value / self.tick_size).to_integral_value(rounding=ROUND_DOWN) * self.tick_size

    def round_price_up(self, price: float | Decimal) -> Decimal:
        """Snap a price up to a tick, so a clamped sell limit stays above it."""
        value = _decimal(price)
        return (value / self.tick_size).to_integral_value(rounding=ROUND_UP) * self.tick_size

    def round_quantity(self, quantity: float | Decimal) -> Decimal:
        """Snap a quantity down to a whole lot.

        Always rounds *down*, never up: rounding up turns an order the account
        can just afford into one it cannot, and turns a full exit into one that
        leaves a residual short.
        """
        value = _decimal(quantity)
        sign = -1 if value < 0 else 1
        magnitude = abs(value)
        lots = (magnitude / self.lot_size).to_integral_value(rounding=ROUND_DOWN)
        rounded = lots * self.lot_size
        if self.max_quantity is not None and rounded > self.max_quantity:
            rounded = self.max_quantity
        return sign * rounded

    def is_tradable(self, quantity: float | Decimal, price: float | Decimal) -> tuple[bool, str]:
        """Whether a quantised order clears the venue's minimums."""
        size = abs(_decimal(quantity))
        if size == 0:
            return False, "quantity rounds to zero at this lot size"
        if size < self.min_quantity:
            return False, f"quantity {size} below minimum {self.min_quantity}"
        notional = size * _decimal(price)
        if notional < self.min_notional:
            return False, f"notional {notional} below minimum {self.min_notional}"
        return True, ""


#: Sub-penny quantities and penny prices: a reasonable stand-in for US equities
#: at a broker offering fractional shares.
EQUITY = InstrumentSpec.create(tick_size="0.01", lot_size="0.000001")

#: Whole shares only, which is what most venues actually enforce.
WHOLE_SHARE = InstrumentSpec.create(tick_size="0.01", lot_size="1", min_quantity="1")

#: Roughly Binance's BTCUSDT filters.
CRYPTO = InstrumentSpec.create(
    tick_size="0.01", lot_size="0.00001", min_quantity="0.00001", min_notional="10"
)

SPECS: dict[str, InstrumentSpec] = {
    "equity": EQUITY,
    "whole-share": WHOLE_SHARE,
    "crypto": CRYPTO,
}


def build_spec(name: str, symbol: str = "") -> InstrumentSpec:
    """Look up a named spec preset."""
    try:
        spec = SPECS[name]
    except KeyError:
        available = ", ".join(sorted(SPECS))
        raise KeyError(f"unknown instrument spec {name!r}; available: {available}") from None
    return InstrumentSpec(
        symbol=symbol or spec.symbol,
        tick_size=spec.tick_size,
        lot_size=spec.lot_size,
        min_quantity=spec.min_quantity,
        min_notional=spec.min_notional,
        max_quantity=spec.max_quantity,
    )

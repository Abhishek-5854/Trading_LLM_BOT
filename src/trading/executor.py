"""
Paper trading executor with risk management and logging.

Features:
- Records all trades with timestamp, symbol, side, size, price, and P&L
- Applies position limits and drawdown controls
- Simulates slippage and transaction fees
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional
import json
from pathlib import Path


@dataclass
class Trade:
    """Single trade record."""
    timestamp: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    size: float
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    slippage: float = 0.0  # pips
    fee: float = 0.0  # fee per trade


class PaperExecutor:
    """
    Paper trading simulator with position tracking and risk controls.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_position_size: float = 100000.0,
        max_daily_loss: float = 200.0,
        max_trades_per_hour: int = 10,
        slippage_pips: float = 1.0,
        fee_per_trade: float = 2.0,  # typical 2-5 pips for FX
    ):
        """
        Initialize paper executor.
        
        Args:
            initial_capital: starting account balance
            max_position_size: max notional value per position
            max_daily_loss: max loss (in currency units) per day before stopping
            max_trades_per_hour: max trades executed in a rolling hour window
            slippage_pips: slippage applied to each trade (pips)
            fee_per_trade: transaction cost per trade (pips)
        """
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_trades_per_hour = max_trades_per_hour
        self.slippage_pips = slippage_pips
        self.fee_per_trade = fee_per_trade
        
        self.trades: List[Trade] = []
        self.current_balance = initial_capital
        self.open_positions = {}  # symbol -> list of trades
        
        Path("logs").mkdir(exist_ok=True)
    
    def can_execute(self, symbol: str, side: str, size: float) -> tuple[bool, str]:
        """
        Check if trade can be executed given risk constraints.
        
        Returns:
            (can_execute, reason_if_blocked)
        """
        notional_value = size * 1.0  # assume ~1.0 per pip notional
        
        # Check max position size
        if notional_value > self.max_position_size:
            return False, f"Size {size} exceeds max position {self.max_position_size}"
        
        # Check daily loss (simple check: compare current balance to initial)
        daily_loss = self.initial_capital - self.current_balance
        if daily_loss > self.max_daily_loss:
            return False, f"Daily loss {daily_loss} exceeds limit {self.max_daily_loss}"
        
        return True, ""
    
    def execute(
        self,
        symbol: str,
        side: str,
        size: float,
        current_price: float,
        timestamp: Optional[str] = None,
    ) -> Optional[Trade]:
        """
        Execute a paper trade.
        
        Args:
            symbol: trading pair (e.g., "XAUUSD")
            side: 'BUY' or 'SELL'
            size: notional position size
            current_price: entry price in market
            timestamp: trade timestamp (defaults to now)
        
        Returns:
            Trade object if execution succeeded, None otherwise
        """
        if not timestamp:
            timestamp = datetime.now().isoformat()
        
        can_do, reason = self.can_execute(symbol, side, size)
        if not can_do:
            print(f"Trade blocked: {reason}")
            return None
        
        # Apply slippage
        if side == "BUY":
            entry_price = current_price + self.slippage_pips * 0.0001  # convert pips to price
        else:
            entry_price = current_price - self.slippage_pips * 0.0001
        
        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            slippage=self.slippage_pips,
            fee=self.fee_per_trade,
        )
        
        self.trades.append(trade)
        
        # Track position
        if symbol not in self.open_positions:
            self.open_positions[symbol] = []
        self.open_positions[symbol].append(trade)
        
        print(f"[{timestamp}] Executed {side} {size} {symbol} @ {entry_price:.4f}")
        
        return trade
    
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        timestamp: Optional[str] = None,
    ) -> Optional[float]:
        """
        Close all open positions for a symbol.
        
        Returns:
            Total P&L from closing
        """
        if not timestamp:
            timestamp = datetime.now().isoformat()
        
        if symbol not in self.open_positions or not self.open_positions[symbol]:
            return 0.0
        
        total_pnl = 0.0
        for trade in self.open_positions[symbol]:
            if trade.side == "BUY":
                pnl = (exit_price - trade.entry_price) * trade.size - trade.fee * 0.0001 * trade.size
            else:
                pnl = (trade.entry_price - exit_price) * trade.size - trade.fee * 0.0001 * trade.size
            
            trade.exit_price = exit_price
            trade.pnl = pnl
            total_pnl += pnl
            
            print(f"[{timestamp}] Closed {trade.side} {symbol} @ {exit_price:.4f}, P&L: {pnl:.2f}")
        
        self.current_balance += total_pnl
        self.open_positions[symbol] = []
        
        return total_pnl
    
    def get_statistics(self) -> dict:
        """
        Compute P&L statistics.
        """
        closed_trades = [t for t in self.trades if t.pnl is not None]
        
        if not closed_trades:
            return {
                "total_trades": len(self.trades),
                "closed_trades": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
            }
        
        wins = [t.pnl for t in closed_trades if t.pnl > 0]
        losses = [t.pnl for t in closed_trades if t.pnl < 0]
        
        return {
            "total_trades": len(self.trades),
            "closed_trades": len(closed_trades),
            "total_pnl": sum(t.pnl for t in closed_trades),
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": len(wins) / len(closed_trades) if closed_trades else 0.0,
            "avg_win": sum(wins) / len(wins) if wins else 0.0,
            "avg_loss": sum(losses) / len(losses) if losses else 0.0,
            "current_balance": self.current_balance,
            "return_pct": (self.current_balance - self.initial_capital) / self.initial_capital * 100,
        }
    
    def log_trades(self, filename: str = "logs/trades.jsonl"):
        """Save trades to JSONL for analysis."""
        with open(filename, "w") as f:
            for trade in self.trades:
                f.write(json.dumps(asdict(trade)) + "\n")
        print(f"Logged {len(self.trades)} trades to {filename}")
    
    def history(self):
        """Return trade history."""
        return self.trades


if __name__ == "__main__":
    # Demo: execute a few trades
    ex = PaperExecutor(initial_capital=10000.0)
    
    print("=== Paper Trading Demo ===\n")
    
    # BUY
    ex.execute("XAUUSD", "BUY", size=100, current_price=2050.0)
    
    # SELL (close)
    ex.close_position("XAUUSD", exit_price=2060.0)
    
    # Print stats
    stats = ex.get_statistics()
    print(f"\nTrading Statistics:")
    for key, val in stats.items():
        print(f"  {key}: {val}")
    
    # Log trades
    ex.log_trades()

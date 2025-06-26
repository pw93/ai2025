from dataclasses import dataclass
from collections import deque
from enum import Enum
from typing import List, Tuple


class BuySell(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Trade:
    price: float
    qty: int
    trade_type: BuySell


def calculate_profit_loss(
    trades: List[Trade],
    current_price: float
) -> Tuple[float, float, List[Trade]]:
    """
Calculate realized (closed) and unrealized (open) profit/loss from a sequence of trades.

This function processes trades in chronological order and matches buys and sells 
using FIFO logic to compute realized profit/loss (closed P/L) and keeps track of 
unmatched open positions to compute unrealized profit/loss (open P/L).

Args:
    trades (List[Trade]): A list of trades, each with price, quantity, and trade_type (BUY or SELL),
                          ordered chronologically.
    current_price (float): The current market price used to evaluate unrealized P/L on open positions.

Returns:
    Tuple containing:
        - closed_profit (float): Total realized profit or loss from matched buy/sell pairs.
        - open_profit (float): Total unrealized profit or loss from unmatched open positions.
        - open_positions (List[Trade]): List of remaining open trades (buys or sells) after matching,
                                        preserving their chronological order.

Notes:
    - Matching uses FIFO: oldest buys are matched first with incoming sells and vice versa.
    - After matching, any leftover quantity from a trade is kept in the respective queue (buy or sell).
    - Unrealized profit/loss is calculated on the leftover open positions using the current market price.
"""

    buy_queue = deque()
    sell_queue = deque()
    closed_profit = 0.0
    open_profit = 0.0

    for trade in trades:
        if trade.trade_type == BuySell.BUY:
            qty_to_match = trade.qty
            while qty_to_match > 0 and sell_queue:
                sell = sell_queue[0]
                matched_qty = min(qty_to_match, sell.qty)
                closed_profit += (sell.price - trade.price) * matched_qty
                qty_to_match -= matched_qty

                if matched_qty == sell.qty:
                    sell_queue.popleft()
                else:
                    sell_queue[0] = Trade(sell.price, sell.qty - matched_qty, BuySell.SELL)

            if qty_to_match > 0:
                buy_queue.append(Trade(trade.price, qty_to_match, BuySell.BUY))

        else:  # BuySell.SELL
            qty_to_match = trade.qty
            while qty_to_match > 0 and buy_queue:
                buy = buy_queue[0]
                matched_qty = min(qty_to_match, buy.qty)
                closed_profit += (trade.price - buy.price) * matched_qty
                qty_to_match -= matched_qty

                if matched_qty == buy.qty:
                    buy_queue.popleft()
                else:
                    buy_queue[0] = Trade(buy.price, buy.qty - matched_qty, BuySell.BUY)

            if qty_to_match > 0:
                sell_queue.append(Trade(trade.price, qty_to_match, BuySell.SELL))

    for trade in buy_queue:
        open_profit += (current_price - trade.price) * trade.qty
    for trade in sell_queue:
        open_profit += (trade.price - current_price) * trade.qty

    open_positions = list(buy_queue) + list(sell_queue) #it keeps order because only one buy_queue or sell_queue left.

    return closed_profit, open_profit, open_positions


def test_calculate_profit_loss():
    trades = [
        Trade(price=85, qty=12, trade_type=BuySell.SELL),
        Trade(price=105, qty=2, trade_type=BuySell.SELL),
        Trade(price=100, qty=1, trade_type=BuySell.BUY),
        Trade(price=102, qty=2, trade_type=BuySell.BUY),
        Trade(price=108, qty=12, trade_type=BuySell.BUY),
        
    ]

    current_price = 106

    closed_pl, open_pl, open_positions = calculate_profit_loss(trades, current_price)

    print(f"Closed P/L: {closed_pl:.2f}")
    print(f"Open P/L: {open_pl:.2f}")
    print("Open Positions:")
    for pos in open_positions:
        print(f"  {pos.trade_type.value.capitalize()} - Price: {pos.price}, Qty: {pos.qty}")


if __name__ == "__main__":
    test_calculate_profit_loss()
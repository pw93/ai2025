# -*- coding: utf-8 -*-
import yfinance as yf
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple
from collections import deque
from logger import logi, loge, logw
import matplotlib.pyplot as plt


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


class BTData:
    def __init__(self):
        self.ymd=None
        self.hms=None
        self.o=None
        self.h=None
        self.l=None
        self.c=None
        self.v=None
    def __str__(self):
        return (f"Date: {self.ymd}, Time: {self.hms}, Open: {self.o}, High: {self.h}, "
                f"Low: {self.l}, Close: {self.c}, Volume: {self.v}")

class BTStrategy:
    def __init__(self):
        print('MyStrategy init')
        self.pt_now = 0
        self.pt_win = []
        self.pt_win_realtime = []

        self.trades_history = []
        self.trades_current = []

    def cal_pos(self):
        pos=0
        for trade in self.trades_current:
            if trade.trade_type == BuySell.BUY:
                pos+=trade.qty
            elif trade.trade_type == BuySell.SELL:
                pos-=trade.qty
        #logi(pos, self.trades_current)
        return pos


    def run_open_top(self, data, o):
        self.pt_now = o
        self.run_at_open(data, o)

    def run_close_top(self, data):
        self.pt_now = data[-1].c
        self.run_at_close(data)
        pl_close, pl_open, _ = calculate_profit_loss(self.trades_current, self.pt_now)        
        if len(self.pt_win)==0:
            self.pt_win_realtime.append(pl_open)
        else:
            self.pt_win_realtime.append(self.pt_win[-1] + pl_open)





    def buy_sell(self, price, pos):
        if pos==0:
            logw('buy_sell pos is zero')
            return

        logi(price,pos)

        if pos>0:
            qty = pos
            trade_type = BuySell.BUY
        else:
            qty = -pos
            trade_type = BuySell.SELL

        td = Trade(price,qty,trade_type)
        self.trades_history.append(td)
        self.trades_current.append(td)

        pl_close, pl_open, self.trades_current = calculate_profit_loss(self.trades_current, self.pt_now)        
        if not self.pt_win:
            self.pt_win.append(pl_close)
        else:
            self.pt_win.append(self.pt_win[-1] + pl_close)


    def buy_to_one(self):
        del_pos = 1 - self.cal_pos()

        if del_pos!=0:
            logi(del_pos)
            self.buy_sell(self.pt_now, del_pos)

    def buy(self, qty):
        self.buy_sell(self.pt_now, qty)

    def sell(self, qty):
        self.buy_sell(self.pt_now, -qty)


    def sell_to_one(self):

        del_pos = -1 - self.cal_pos()
        if del_pos!=0:
            logi(del_pos)
            self.buy_sell(self.pt_now, del_pos)

    def clear_bs(self):
        del_pos = -self.cal_pos()
        self.buy_sell(self.pt_now, del_pos)




    # These should be implemented by subclasses
    def run_at_open(self, data, o):
        pass

    def run_at_close(self, data):
        pass







class BT:

    def __init__(self):

        pass

    def run(self, strategy, data):
        pt_win=0
        sz = len(data)
        for i in range(sz):
            #logi(i)
            d_open=data[:i]
            d_close = data[:i+1]

            strategy.run_open_top(d_open,data[i].o)
            strategy.run_close_top(d_close)



class MyStrategy(BTStrategy):
    def __init__(self):
        print("MyStrategy")
        super().__init__()

        self.c=[]
        pass

    def run_at_open(self, data, o):
        pass

    def run_at_close(self, data):
        self.c.append(data[-1].c)

        if len(self.c) >= 10:
            ma5 = sum(self.c[-2:]) / 2
            ma10 = sum(self.c[-10:]) / 10

            if ma5 > ma10:
                self.buy_to_one()
                #logi("buy")
            else:
                self.sell_to_one()
                #logi("sell")


class MyStrategy2(BTStrategy):
    def __init__(self):
        print("MyStrategy")
        super().__init__()

        self.c=[]
        pass

    def run_at_open(self, data, o):
        pass

    def run_at_close(self, data):
        self.buy_to_one()





import dill
def write_dill(fname, data):
    with open(fname, 'wb') as f:
        dill.dump(data, f)

def read_dill(fname):
    """Deserialize data from a dill file."""
    with open(fname, 'rb') as f:
        return dill.load(f)

fname = 'c:\\temp\\a.pickle'
if 1:
    df = yf.Ticker("SPY").history(start="2024-01-01", end="2025-06-23", auto_adjust=True)
    write_dill(fname,df)
else:
    df = read_dill(fname)

o_arr=df[['Open']]
c_arr=df[['Close']]
sz = len(df)


from datetime import datetime

def dt_to_ymd(dt):
    y, m, d = dt.year, dt.month, dt.day
    ymd = y * 10000 + m * 100 + d
    return ymd

def dt_to_hms(dt):
    h, m, s = dt.hour, dt.minute, dt.second
    hms = h * 10000 + m * 100 + s
    return hms

def ymd_to_dt(ymd):
    y = ymd // 10000
    m = (ymd % 10000) // 100
    d = ymd % 100
    return datetime(y, m, d)

def hms_to_dt(hms):
    h = hms // 10000
    m = (hms % 10000) // 100
    s = hms % 100
    return h, m, s  # typically returned as a tuple

def ymd_hms_to_dt(ymd, hms):
    y = ymd // 10000
    m = (ymd % 10000) // 100
    d = ymd % 100
    h = hms // 10000
    mi = (hms % 10000) // 100
    s = hms % 100
    return datetime(y, m, d, h, mi, s)

data = []

for idx, row in df.iterrows():
    d = BTData()
    dt = idx.tz_localize(None)
    ymd = dt_to_ymd(dt)
    hms = dt_to_hms(dt)
    d.ymd = ymd
    d.hms = hms
    d.o = row['Open']
    d.h = row['High']
    d.l = row['Low']
    d.c = row['Close']
    d.v = row['Volume']
    data.append(d)





strategy = MyStrategy()
bt = BT()
bt.run(strategy, data)  # this will properly run your backtest using your dataset

if 1:
    plt.plot(strategy.pt_win_realtime)
    plt.title("Realtime P&L")
    plt.xlabel("Time (ticks)")
    plt.ylabel("Profit / Loss")
    plt.grid(True)
    plt.show()


strategy = MyStrategy2()
bt = BT()
bt.run(strategy, data)  # this will properly run your backtest using your dataset

plt.plot(strategy.pt_win_realtime)
plt.title("Realtime P&L")
plt.xlabel("Time (ticks)")
plt.ylabel("Profit / Loss")
plt.grid(True)
plt.show()
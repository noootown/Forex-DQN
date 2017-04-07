from helper import getUTCTimeStrFromMs
from constants import BUY, SELL, SHOW_HAND

class Account:
  def __init__(
    self,
    leverage = 50,
    balance = 100000,
    cliOutput = False,
  ):
    self.leverage = leverage
    self.balance = balance
    self.position = 0
    self.price = 0
    self.units = 0
    self.showHand = False # False means hasn't show hand, True means show hand
    self.openTime = 0

    self.cliOutput = cliOutput

    self.itemHeader = [
      'type',
      'price',
      'closePrice',
      'units',
      'pl',
      'balance',
      'openTime',
      'closeTime',
    ]

  def order(
    self,
    price,
    action,
    time,
  ):
    if (action['type'] != BUY and action['type'] != SELL) \
      or self.balance < 100 \
      or self.showHand: # to discourage repeat order
      return 0

    posNeg = 1 if action['type'] == BUY else -1

    if action['unit'] == SHOW_HAND:
      unit = int(self.balance * self.leverage / price)
      self.showHand = True
    else:
      unit = action['unit']

    self.position += unit * price * posNeg
    self.units += unit * posNeg

    self.price = self.position / self.units
    if self.openTime == 0:
      self.openTime = time

    return 0

  def closePosition(
    self,
    price,
    time,
  ):
    if self.units == 0:
      return 0

    pl = \
      (price - self.price) * self.units \
        if self.units > 0 \
        else (self.price - price) * -self.units

    balance = self.balance

    if self.cliOutput:
      print([
        'B' if self.units > 0 else 'S',
        '%.5f' % self.price,
        '%.5f' % price,
        abs(self.units),
        '%.2f' % pl,
        '%.2f' % (balance + pl),
        getUTCTimeStrFromMs(self.openTime),
        getUTCTimeStrFromMs(time),
      ])

    self.balance += pl
    self.pl = 0
    self.position = 0
    self.price = 0
    self.units = 0
    self.showHand = False
    self.openTime = 0

    return pl * 100 / balance

  def stop(self):
    return 0

  @staticmethod
  def getCloseHeader():
    return Account().itemHeader

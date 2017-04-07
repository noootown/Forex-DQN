import numpy as np

class Forex:
  def __init__(
    self,
    midPriceMap,
    count,
  ):
    self.midPriceMap = midPriceMap
    self.midPriceArr = None
    self.midPricePtr = 0

    self.date = ''
    self.count = count

    self.featureLen = 0

  def setDate(self, date):
    self.date = date

    self.midPriceArr = self.midPriceMap[date]
    self.midPricePtr = len(self.midPriceArr) - 1
    self.featureLen = self.count * (len(self.midPriceArr[0]) - 2)

  def getTime(self):
    # number time is at last column
    startTime = int(self.midPriceArr[-1, -1]) + 5 * 60 * 1000 * self.count
    endTime = int(startTime + 5 * 86400 * 1000) # number
    return startTime, endTime

  def getPrice(self, time):
    while self.midPricePtr >= 0 and time >= self.midPriceArr[self.midPricePtr, -1]:
      self.midPricePtr -= 1

    self.midPricePtr += 1

    observation = \
      np.reshape(self.midPriceArr[self.midPricePtr + 1 : self.midPricePtr + self.count + 1, 0:-2],
                 self.featureLen)

    price = self.midPriceArr[self.midPricePtr, 0]

    return price, observation

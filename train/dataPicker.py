from helper import getMsByTimeStr
import numpy as np

def dataContentPicker(
  data,
  dataType,
):
  if dataType == 'mid':
    if data['complete']:
      return [
        data['mid']['o'],
        data['mid']['h'],
        data['mid']['l'],
        data['mid']['c'],
      ]
    else:
      return [0, 0, 0, 0]

def dataPicker(
  data,
  dataType = 'mid',
):
  arr = dataContentPicker(data, dataType)
  return np.concatenate((
    arr if type(arr).__module__ == 'numpy' else np.array(arr),
    [1 if data['complete'] else 0, getMsByTimeStr(data['time'])]
  ),axis = 0)

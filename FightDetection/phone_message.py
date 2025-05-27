import requests
import json

URL = 'https://www.sms4india.com/api/v1/sendCampaign'

def sendPostRequest(reqUrl, apiKey, secretKey, useType, phoneNo, senderId, textMessage):
  req_params = {
  'apikey':apiKey,
  'secret':secretKey,
  'usetype':useType,
  'phone': phoneNo,
  'message':textMessage,
  'senderid':senderId
  }
  return requests.post(reqUrl, req_params)

def main():
    response = sendPostRequest(URL, 'type_your_api_key', 'type_your_secret_key', 'stage', '9925335903', 'Meet', 'message-text' )

if __name__ == '__main__':
    print (response.text)


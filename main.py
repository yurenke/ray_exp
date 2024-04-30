from configparser import RawConfigParser
import os, json
import redis
import time, logging

import asyncio
import asyncio
# import async_timeout
import aioredis

from asgiref.sync import async_to_sync, sync_to_async
from trainmodel import ChineseChatModel, ChineseNicknameModel
import requests

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

config = RawConfigParser()
config.read(BASE_DIR+'/setting.ini')

logging_level = logging.INFO
logging.basicConfig(format='[%(levelname)s]%(asctime)s %(message)s', datefmt='(%m/%d) %I:%M:%S %p :: ', level=logging_level)

def main():
    logging.info('=============  Async Train Service Activated. subscribe to redis PubSub: [ training_request channel ] =============')

    r = redis.Redis(host='localhost', port=config.get('CHANNEL', 'CHANNEL_PORT'), db=0)

    sub = r.pubsub()
    sub.subscribe("training_request")

    while True:
        msg = sub.get_message()
        if msg:
            logging.info(f"new message: {msg['data']}")
            if isinstance(msg.get('data'), bytes):
                command_dict_str = msg['data'].decode()
                command_dict = json.loads(command_dict_str)
                command = command_dict['command']
                if command == 'train_chinese_chat':                    
                    logging.info('============ received chinese chat model training request ===========')
                    try:
                        model = ChineseChatModel()
                        model.fit_model()

                        # notify django training is done
                        r = requests.get('http://127.0.0.1:8000/api/complete/train_chinese_chat')

                        if r.status_code == requests.codes.ok:
                            logging.info("chinese chat model train complete notification OK")
                        else:
                            logging.info('chinese chat model train complete notification error')

                    except Exception as err:
                        logging.error(str(err))
                        requests.get('http://127.0.0.1:8000/api/complete/train_chinese_chat')

                elif command == 'test_chinese_chat':
                    logging.info('============ received chinese chat model testing request ===========')
                    origin = command_dict['origin']
                    try:
                        model = ChineseChatModel()
                        model.test_by_origin(origin=origin)

                        # notify django training is done
                        r = requests.get('http://127.0.0.1:8000/api/complete/test_chinese_chat')

                        if r.status_code == requests.codes.ok:
                            logging.info("chinese chat model test complete notification OK")
                        else:
                            logging.info('chinese chat model test complete notification error')
                    except Exception as err:
                        logging.error(str(err))
                        requests.get('http://127.0.0.1:8000/api/complete/test_chinese_chat')

                elif command == 'train_chinese_nickname':                    
                    logging.info('============ received chinese nickname model training request ===========')
                    try:
                        model = ChineseNicknameModel()
                        model.fit_model()

                        # notify django training is done
                        r = requests.get('http://127.0.0.1:8000/api/complete/train_chinese_nickname')

                        if r.status_code == requests.codes.ok:
                            logging.info("chinese nickname model train complete notification OK")
                        else:
                            logging.info('chinese nickname model train complete notification error')

                    except Exception as err:
                        logging.error(str(err))
                        requests.get('http://127.0.0.1:8000/api/complete/train_chinese_nickname')

                elif command == 'test_chinese_nickname':
                    logging.info('============ received chinese nickname model testing request ===========')
                    origin = command_dict['origin']
                    try:
                        model = ChineseNicknameModel()
                        model.test_by_origin(origin=origin)

                        # notify django training is done
                        r = requests.get('http://127.0.0.1:8000/api/complete/test_chinese_nickname')

                        if r.status_code == requests.codes.ok:
                            logging.info("chinese nickname model test complete notification OK")
                        else:
                            logging.info('chinese nickname model test complete notification error')
                    except Exception as err:
                        logging.error(str(err))
                        requests.get('http://127.0.0.1:8000/api/complete/test_chinese_nickname')
        else:
            # logging.info('no msg! sleep for 5 secs')
        
            time.sleep(5)  # be nice to the system :)
        

if __name__ == '__main__':
    main()

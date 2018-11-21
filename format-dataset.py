import re
import json
import urllib.parse


# match '=', '&', '<', '>', '(', ')', '.', ' '
reg = re.compile(r'[=&<>(). ]')
def split_payload(str):
    return reg.split(str)

def decode_payload(str):
    # double decoding
    str = urllib.parse.unquote_plus(str)
    str = urllib.parse.unquote_plus(str)
    return str

def parse_dataset(filepath):
    text = ''
    with open(filepath) as f:
        for line in f:
            if ('GET' in line) or ('POST' in line) or ('PUT' in line):
                if text != '':
                    yield text
                    text = ''
            text = text + line
        yield text

def parse_raw_http(str):
    arr = str.split('\n')
    method, url, _ = arr[0].split(' ')
    u = urllib.parse.urlparse(url)

    payload = ''

    if method == 'GET':
        payload = u.query
    elif method == 'POST' or method == 'PUT':
        for line in reversed(arr):
            if line == '':
                continue
            else:
                payload = line
                break

    # To be confirmed
    words = [method, u.netloc, *u.path.split(r'/')[1:], *split_payload(decode_payload(payload))]

    return {'payload': payload, 'words': words}



if __name__ == '__main__':
    # with open('./norm-train.jsonl', 'w') as f:
    #     for text in parse_dataset('./static/normalTrafficTraining.txt'):
    #         req = parse_raw_http(text)
    #         req['label'] = 'norm'
    #         f.write('{}\n'.format(json.dumps(req)))
            
    # with open('./norm-test.jsonl', 'w') as f:
    #     for text in parse_dataset('./static/normalTrafficTest.txt'):
    #         req = parse_raw_http(text)
    #         req['label'] = 'norm'
    #         f.write('{}\n'.format(json.dumps(req)))
            
    # with open('./anom-test.jsonl', 'w') as f:
    #     for text in parse_dataset('./static/anomalousTrafficTest.txt'):
    #         req = parse_raw_http(text)
    #         req['label'] = 'anom'
    #         f.write('{}\n'.format(json.dumps(req)))
            
    with open('./static2/norm-train.jsonl', 'w') as f:
        for i, text in enumerate(parse_dataset('./static/normalTrafficTraining.txt')):
            req = parse_raw_http(text)
            req['label'] = 'norm{}'.format(i)
            f.write('{}\n'.format(json.dumps(req)))
            
    with open('./static2/norm-test.jsonl', 'w') as f:
        for i, text in enumerate(parse_dataset('./static/normalTrafficTest.txt')):
            req = parse_raw_http(text)
            req['label'] = 'norm{}'.format(i)
            f.write('{}\n'.format(json.dumps(req)))
            
    with open('./static2/anom-test.jsonl', 'w') as f:
        for i, text in enumerate(parse_dataset('./static/anomalousTrafficTest.txt')):
            req = parse_raw_http(text)
            req['label'] = 'anom{}'.format(i)
            f.write('{}\n'.format(json.dumps(req)))
            
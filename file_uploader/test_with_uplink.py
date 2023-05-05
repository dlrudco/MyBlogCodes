import anvil
from anvil import server

import cv2

@server.callable
def circle_drawer(file):
    file = file.get_bytes()
    img = cv2.imdecode(file, cv2.IMREAD_COLOR)
    cv2.circle(img, (100, 100), 50, (0, 0, 255), -1)
    _, img_encoded = cv2.imencode('.jpg', img)
    return img_encoded.tobytes()

if __name__ == "__main__":
    uplink_key = open("credentials.txt", "r").read()
    key = uplink_key.split(':')[1]
    
    anvil.server.connect(key)
    anvil.server.wait_forever()
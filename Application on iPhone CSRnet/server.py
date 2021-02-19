import tornado
from tornado.options import define, options
import tornado.ioloop
import tornado.options
import tornado.httpserver
import tornado.web
import os, json, numpy as np, cv2
from model import CSRNet
from PIL import Image
from io import BytesIO
import base64, time
import matplotlib.pyplot as plt
import socket

# Get the host name
hostname = socket.gethostname()
# Get IP address
host = socket.gethostbyname(hostname)

define("port", default=8000, help="Server port", type=int)

Model = CSRNet()


def create_img(img):
    # Convert PIL image to image matrix that can be recognized by neural network

    im = np.array(img)

    im = im / 255.0

    im[:, :, 0] = (im[:, :, 0] - 0.485) / 0.229
    im[:, :, 1] = (im[:, :, 1] - 0.456) / 0.224
    im[:, :, 2] = (im[:, :, 2] - 0.406) / 0.225

    im = np.expand_dims(im, axis=0)
    return im


class Index(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")  # Domain names can be written in this place
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

    def options(self):
        self.write('{"errorCode":"00","errorMessage","success"}')

    def post(self):
        img = Image.open(
            BytesIO(
                self.request.files.get("img")[0].get('body'))).convert('RGB')
        if img.size[0] > 1500 or img.size[1] > 1500:
            rate = 1500 / img.size[0]
            img = img.resize((1500, int(rate * img.size[1])), Image.ANTIALIAS)

        resize_img = img.resize((1024, 768), Image.ANTIALIAS)
        Model.load_weights("A.hdf5")

        img = create_img(img)
        predictA = np.squeeze(Model.predict(img))
        print(predictA.shape)
        plt.imsave("static/A.hot.jpg", predictA)

        Model.load_weights("B.hdf5")

        img = create_img(resize_img)
        predictB = np.squeeze(Model.predict(img))
        print(predictB.shape)
        plt.imsave("static/B.hot.jpg", predictB)

        self.write(
            json.dumps({
                "code": 200,
                "A": "http://{}:{}/static/A.hot.jpg?t={}".format(host, options.port, str(time.time())),
                "B": "http://{}:{}/static/B.hot.jpg?t={}".format(host, options.port, str(time.time())),
                "A_SUM": int(np.sum(predictA)),
                "B_SUM": int(np.sum(predictB)),
            }))


def main():
    tornado.options.parse_command_line()

    http_server = tornado.httpserver.HTTPServer(
        tornado.web.Application(
            handlers=((r"/", Index),),
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            debug=True,
        ))
    print("Server run on: http://{}:{}".format(host, options.port))
    print("Press ctrl + C to quit")
    http_server.listen(options.port)

    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()

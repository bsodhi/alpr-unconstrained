import os
import time
import json
import logging
import cStringIO
import requests

from tornado.ioloop import IOLoop
from tornado.queues import Queue
from tornado.web import RequestHandler
from tornado.web import Application
from tornado import gen
from tornado.concurrent import Future

import sys
import cv2
import numpy as np
import traceback
import keras

import darknet.python.darknet as dn

from os.path import splitext, basename
from glob import glob
from darknet.python.darknet import detect
from src.label import dknet_label_conversion, Shape, writeShapes
from src.utils import nms, im2single
from src.keras_utils import load_model, detect_lp

import argparse

# Should be initialized right after server startup
WPOD_PATH = os.path.join(os.getcwd(), "data/lp-detector/wpod-net_update1.h5")


def lp_detect(task_dir):
    lp_text = None
    try:
        img_path = os.path.join(task_dir, "input_frame.jpg")
        logging.info("Processing {0}".format(img_path))

        lp_threshold = .5
        wpod_net = load_model(WPOD_PATH)
        logging.info("Searching for license plates using WPOD-NET")
        bname = splitext(basename(img_path))[0]
        Ivehicle = cv2.imread(img_path)

        ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
        side = int(ratio * 288.)
        bound_dim = min(side + (side % (2**4)), 608)
        logging.info("Bound dim: {0}, ratio: {1}".format(bound_dim, ratio))

        Llp, LlpImgs, _ = detect_lp(wpod_net, im2single(Ivehicle), bound_dim,
                                    2**4, (240, 80), lp_threshold)

        if len(LlpImgs):
            Ilp = LlpImgs[0]
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
            out_file = os.path.join(task_dir, "lp_{0}.jpg".format(bname))
            cv2.imwrite(out_file, Ilp * 255.)
            logging.info(
                "Licence plate written to {0}. Starting OCR.".format(out_file))
            lp_text = ocr(task_dir, file_name_pattern="lp_*.jpg")
        else:
            logging.warning("No plates found in {0}".format(img_path))

    except Exception as ex:
        logging.exception("Error occurred when extracting licence plate.")
    return lp_text


def ocr(task_dir, file_name_pattern="box_*.jpg"):
    text = None
    try:
        ocr_threshold = .4
        ocr_weights = 'data/ocr/ocr-net.weights'
        ocr_netcfg = 'data/ocr/ocr-net.cfg'
        ocr_dataset = 'data/ocr/ocr-net.data'
        ocr_net = dn.load_net(ocr_netcfg, ocr_weights, 0)
        ocr_meta = dn.load_meta(ocr_dataset)
        imgs_paths = sorted(glob("{0}/{1}".format(task_dir,
                                                  file_name_pattern)))
        if len(imgs_paths) < 1:
            raise Exception("No boxes found in {}".format(task_dir))
        logging.info("Performing OCR on {0} .jpg files from {1}".format(
            len(imgs_paths), task_dir))

        output = cStringIO.StringIO()
        for i, img_path in enumerate(imgs_paths):
            try:
                logging.info("Scanning {}".format(img_path))
                bname = basename(splitext(img_path)[0])

                R, (width, height) = detect(ocr_net,
                                            ocr_meta,
                                            img_path,
                                            thresh=ocr_threshold,
                                            nms=None)

                if len(R):
                    L = dknet_label_conversion(R, width, height)
                    L = nms(L, .45)

                    L.sort(key=lambda x: x.tl()[0])
                    lp_str = ''.join([chr(l.cl()) for l in L])
                    logging.info("Found '{0}' in {1}.".format(
                        lp_str, img_path))
                    output.write(lp_str)
                    output.write("\n")

                else:
                    logging.info('No characters found.')
            except Exception as ex:
                logging.exception(
                    "Error occurred when performing OCR on {}. Continuing to next image."
                    .format(img_path))

        text = output.getvalue()
        output.close()
    except:
        logging.exception("Fatal error when performing OCR.")
    logging.info("Found text {}".format(text))

    return text


Q = Queue(maxsize=1000)


def _send_callback(cb_url, task_type, status, file_path, text):
    data = {"task_type": task_type, "status": status,
            "file_path": file_path, "text": text}
    jr = requests.post(cb_url, json=data, verify=False).json()
    logging.info("Sent callback: {0}. Remote response: {1}".format(data, jr))
    return jr


@gen.coroutine
def process_request():
    while True:
        json_args = yield Q.get()
        try:
            tt = json_args["task_type"].upper()
            cb_url = json_args["callback_url"]
            td = json_args["task_dir"]
            file_path = os.path.join(td, "input_frame.jpg")
            logging.info("Dequeued request: "+str(json_args))

            if tt == "OCR":
                text = ocr(td)
            elif tt == "ALPR":
                text = lp_detect(td)
            else:
                raise Exception("Unsupported task type: {0}".format(tt))

            jr = _send_callback(cb_url, tt, "OK", file_path, text)
            fut = Future()
            fut.set_result(jr["body"])
            yield fut

            if jr and jr["status"] == "OK":
                Q.task_done()
            else:
                logging.error("Callback failed. Remote reply: "+str(jr))

        except Exception as ex:
            logging.exception("Error occurred when processing request.")
            try:
                _send_callback(cb_url, tt, "ERROR", file_path,
                               "Failed to process request. "+str(ex))
            except Exception as ex2:
                logging.exception("Could not callback to {}".format(cb_url))


class ApiHandler(RequestHandler):
    def prepare(self):
        if self.request.headers.get("Content-Type", "").\
            startswith("application/json"):
            self.json_args = json.loads(self.request.body)
        else:
            self.json_args = None

    def validate_req(self):
        if not self.json_args:
            raise Exception("Only JSON requests supported.")
        if "task_dir" not in self.json_args or \
            "task_type" not in self.json_args or \
                "callback_url" not in self.json_args:
            raise Exception(
                "Expected JSON payload with 'task_dir', \
                'task_type' and 'callback_url'")

    @gen.coroutine
    def post(self):
        self.validate_req()
        try:
            tt = self.json_args["task_type"].lower()
            cb_url = self.json_args["callback_url"]
            td = self.json_args["task_dir"]
            logging.info("Received request: "+str(self.json_args))

            yield Q.put(self.json_args)
            self.write({"status": "OK", "body": "Processing"})
            logging.info("Request queued for processing.")
        except Exception as ex:
            logging.exception("Error occurred when handling request.")
            self.write(
                {"status": "ERROR",
                 "body": "Failed to handle request. " + str(ex)})

    get = post


def make_app():
    return Application([
        (r"/api", ApiHandler),
    ])


if __name__ == "__main__":
    logging.basicConfig(filename='server.log',
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s:: %(message)s',
                        datefmt='%d-%m-%Y@%I:%M:%S %p')

    parser = argparse.ArgumentParser()
    parser.add_argument("-p",
                        "--port",
                        type=int,
                        default=8888,
                        dest="port",
                        help="Server port number.")
    parser.add_argument("-w",
                        "--wpod-path",
                        type=str,
                        dest="wpod_path",
                        help="WPOD model path.")

    args = parser.parse_args()
    if args.wpod_path:
        WPOD_PATH = args.wpod_path

    print("Using WPOD model from {0}".format(WPOD_PATH))

    app = make_app()
    app.listen(args.port)

    # Must start this one first
    print("Starting request processor couroutine on IOLoop.")
    IOLoop.current().spawn_callback(process_request)

    print("Starting server at port {0}".format(args.port))
    sys.stdout.flush()
    IOLoop.current().start()

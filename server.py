import io
import flask
from flask import Flask, make_response
from translator.utils import cv2_to_pil, pil_to_cv2
from translator.pipelines import FullConversion
from PIL import Image

app = Flask(__name__)

converter = FullConversion()


@app.route("/clean", methods=["POST"])
def clean():
    global converter
    imagefile = flask.request.files.get("image")
    image_cv2 = pil_to_cv2(Image.open(io.BytesIO(imagefile.read())))
    converted = converter([image_cv2])[0]
    converted_pil = cv2_to_pil(converted)
    img_byte_arr = io.BytesIO()
    converted_pil.save(img_byte_arr, format="PNG")
    # Create response given the bytes
    response = flask.make_response(img_byte_arr.getvalue())
    response.headers.set("Content-Type", "image/png")
    return response


app.run()

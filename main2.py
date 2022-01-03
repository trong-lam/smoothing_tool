from flask import Flask, flash, request, redirect, url_for, render_template, jsonify, send_from_directory,send_file
import urllib.request
import os
from werkzeug.utils import secure_filename
import json
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np
import copy
from testing_contours import centerlize_contour_image, preprocess_img
from tool_copy import is_inside_polygon,smoothing_line, is_inside_contour_and_get_local_line,convert_color_img,show_line_with_diff_color
from normalize import Normalize
from collections import defaultdict
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
normalize_obj = Normalize()


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('via4.html')

""" Saving image to the static/uploads in form of normalized image
    params:
    --filename: Name of target image

"""
@app.route('/static/uploads/<filename>', methods=['POST', 'GET'])
def recieve_img(filename):
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            nparr = np.fromstring(file.read(), np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
            normalized_pred_img = normalize_obj.preprocess_img(img_np)
            # = convert_color_img(normalized_pred_img, 'x')
            cv2.imwrite(path, normalized_pred_img)
            return send_from_directory(app.config['UPLOAD_FOLDER'],filename, as_attachment=False )


""" Recieving the gaussian value and make the change on the local line
    params:
    --id_img: ID of target image
    --region_id: ID of specific region in image
    --highlight: whether just show image or make a change
"""
@app.route('/<highlight>/gaussian/<filename>/<id_img>/<region_id>', methods=['POST', 'GET'])
def upload_image(id_img,region_id, filename, highlight):
    effect = "gaussian" # may be change later
    path = os.path.join(os.getcwd(), 'dataOfEffect.json')

    if request.method == 'GET':
        with open(path, 'r+') as jsonFile:
            #check if file is empty
            if os.path.getsize(path) == 0:
                content = {
                   id_img:{
                      effect:[
                         {
                            "global_rate":0,
                            "local_rate":0,
                            "long_rate":0,
                             "only_x":"True",
                             "only_y":"True",
                         }
                      ]
                   }
                }
                jsonFile.write(str(json.dumps(content,indent=2)))
                return content[id_img][effect][0]
            else:
                all_data = json.loads(jsonFile.read())
                array_of_region_data = all_data[id_img][effect]
                if int(region_id) + 1 > len(all_data[id_img][effect]):
                    array_of_region_data.append({"global_rate":0,
                                                 "local_rate": 0,
                                                 "long_rate": 0,
                                                 "only_x": "True",
                                                 "only_y": "True",
                                                 })
                    jsonFile.seek(0)
                    json.dump(all_data,jsonFile,indent=2)
            return jsonify(array_of_region_data[int(region_id)])

        return "Error reading file"

    if request.method == 'POST':
        with open(path, 'r+') as jsonFile:

            #check if
            all_data = json.loads(jsonFile.read())
            #print("all_data",all_data)
            request_data = request.get_json()
            highlight = True if highlight.lower() == "true" else False

            # only show the highlight line
            if not highlight:
                array_of_region_data = all_data[id_img][effect]
                data_of_region_id = array_of_region_data[int(region_id)]

                #update data in json file

                data_of_region_id['global_rate'] = request_data['global_rate']
                data_of_region_id['local_rate'] = request_data['local_rate']
                data_of_region_id['long_rate'] = request_data['long_rate']
                data_of_region_id['only_x'] = request_data['only_x']
                data_of_region_id['only_y'] = request_data['only_y']

                # smoothing
                glob = int(request_data['global_rate'])
                local = int(request_data['local_rate'])
                long = int(request_data['long_rate']) / 100
                only_x = True if request_data['only_x'].lower() == "true" else False
                only_y = True if request_data['only_y'].lower() == "true" else False

                # print("request_data['only_x']",request_data['only_x'])
                # print("long", long)
                # print("local", local)
                # print("glob", glob)
                # print("only_x", only_x, type(only_x))
                # print("only_y", only_y, type(only_y))

            else:
                glob = None
                local = None
                long = None
                only_x = None
                only_y = None

            all_points_x = request_data['attr']['all_points_x']
            all_points_y = request_data['attr']['all_points_y']

            jsonFile.seek(0)
            json.dump(all_data, jsonFile, indent=2)
            jsonFile.truncate()

            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            all_contours,_,normalized_shape = normalize_obj.get_attributes()
            blank = np.zeros(normalized_shape[:2], dtype=np.uint8)
            result_image = convert_color_img(blank, 'x')
            highlight_contour = []

            for index_of_cnt in range(len(all_contours)):
                result, mul_range, has_line_break = is_inside_contour_and_get_local_line(all_points_x,
                                                                        all_points_y,
                                                                        all_contours[index_of_cnt])

                if not result:
                    result_image = cv2.drawContours(result_image, all_contours, index_of_cnt, (255, 0, 255), 1)
                else:
                    highlight_contour.append([index_of_cnt, mul_range])
                    # blank1 = np.zeros(normalized_shape, dtype=np.uint8)
                    # blank1 = convert_color_img(blank1, 'x')
                    # show_line_with_diff_color(blank1,  all_contours[index_of_cnt])
                    #cv2.drawContours(result_image, all_contours, index_of_cnt, (255, 0, 255), 1)


            #print("highlight_contour: ",highlight_contour)
            count = 0
            for hcnt in highlight_contour:
                index, mul_range = hcnt
                #print("hcnt: ", hcnt)
                #print("mul_range: ", mul_range)
                global_contours = all_contours[index].copy()
                result_image, g_contours = smoothing_line(result_image, global_contours,
                                                          mul_range, False,
                                                          only_x, only_y,
                                                          local, glob,
                                                          long, normalized_shape, highlight)
                # plt.imshow(result_image)
                # plt.show()
                normalize_obj.update(result_image)

                # for rax in mul_range:
                #     start_cnt, end_cnt = rax
                #     line = global_contours[start_cnt:end_cnt]
                #     ranging = [start_cnt, end_cnt]
                #     result_image, g_contours = smoothing_line(result_image,global_contours,
                #                                               mul_range,False,
                #                                               only_x, only_y,
                #                                               local, glob,
                #                                               long, normalized_shape, highlight)
                #     normalize_obj.update(result_image)
                #     count +=1

            cv2.imwrite(path, result_image)
            return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False)

        #return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False)

            #return data_of_region_id

        return "Error reading file"









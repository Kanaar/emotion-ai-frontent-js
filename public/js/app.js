
def get_faces_coordinates(canvas):
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    return faceCascade.detectMultiScale(
        canvas,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(55, 55)
    )

def get_face_frames(canvas, faces_coordinates):
    for (x, y, w, h) in faces_coordinates:
        return canvas[y:y + h, x:x + w]

def call_api(snapshot):
    np.save('snapshot', snapshot)
    url='http://127.0.0.1:8000/predict_array'
    # url='https://emotion-ai-cr-3l4porcyga-ew.a.run.app/predict_array'
    files = {'files': open('snapshot.npy', 'rb')}
    response = requests.post(url, files=files)
    os.remove('snapshot.npy')
    return response.json()

def take_snaphot(canvas, faces_coordinates):
    image = get_face_frames(canvas, faces_coordinates)
    snapshot = cv2.resize(image, dsize=(48, 48), interpolation=cv2.INTER_CUBIC)
    return snapshot

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

def display(window, canvas, faces_coordinates, rectangles=False, text=""):
    if rectangles:
        for (x, y, w, h) in faces_coordinates:
            canvas = cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(canvas, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        window.image(canvas)
    else:
        window.image(canvas)

def render_frame(window, canvas, faces_coordinates, prediction, output_field, boxed):
    if boxed:
        display(window, canvas, faces_coordinates, rectangles=True, text=prediction)
    else:
        display(window, canvas, faces_coordinates)
        output_field.empty()
        output_field.markdown(f"Emotional state: {prediction}")

def resize_import(image):
    size_adjust = 480 / image.shape[0]
    if size_adjust < 1:
        return cv2.resize(image, (0,0), fx=size_adjust, fy=size_adjust)
    return image

def img_to_narray(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    return image
